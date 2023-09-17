import copy
import json
import os
import platform
import sys
import time
import torch

from datetime import datetime

import utils.arg_parser

from models.model_loader import ModelLoader
from train.trainer_loader import TrainerLoader
from utils.data.data_prep import DataPreparation
from utils.logger import Logger
from utils.misc import get_split_str

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

def train_and_eval(args, model, dataset, data_loader, val_dataset, val_data_loader, logger, device, job_path, eval_filename, fold):
    # Get trainer
    trainer_creator = getattr(TrainerLoader, args.model)
    trainer = trainer_creator(args, model, dataset, data_loader, logger, device)

    if args.train:
        evaluator = trainer_creator(args, model, val_dataset, val_data_loader, logger, device)
        evaluator.train = False

    if args.train:
        print("Training ...", flush=True)
    else:
        print("Evaluating ...", flush=True)
        vars(args)["num_epochs"] = 1

    # Start training/evaluation
    max_score = 0
    file = open(eval_filename, "a")
    today = datetime.now()
    currentDateTime = today.strftime("%b-%d-%Y-%H-%M-%S")
    file.write(str(currentDateTime))
    file.write("\nModel is {}\n".format(args.model))
    file.close()
    while trainer.curr_epoch < args.num_epochs:
        if args.train:
            file = open(eval_filename, "a")
            file.write("\nEpoch {}:\n".format(trainer.curr_epoch))
            file.close()

            trainer.train_epoch()

            # Eval & Checkpoint
            checkpoint_name = "ckpt-fold{}-e{}".format(fold, trainer.curr_epoch) if fold else "ckpt-e{}".format(trainer.curr_epoch)
            checkpoint_path = os.path.join(job_path, checkpoint_name)

            model.eval()
            result = evaluator.train_epoch()

            if evaluator.REQ_EVAL:
                score = val_dataset.eval(result, checkpoint_path)
            else:
                score = result
            model.train()

            logger.scalar_summary("score", score, trainer.curr_epoch)

            # TODO: Eval model
            # Save the models
            checkpoint = {"epoch": trainer.curr_epoch,
                          "max_score": max_score,
                          "optimizer" : trainer.optimizer.state_dict()}
            checkpoint_path += ".pth"
            torch.save(model.state_dict(), checkpoint_path)
            torch.save(checkpoint, os.path.join(job_path, "training_checkpoint.pth"))

            if score > max_score:
                print("The current model of epoch {} with a score of {} is the best.".format(trainer.curr_epoch, score))
                file = open(eval_filename, "a")
                file.write("\nThe current model of epoch {} with a score of {} is the best.\n".format(trainer.curr_epoch, score))
                file.close()
                max_score = score
                link_name = "best-ckpt.pth"
                link_path = os.path.join(job_path, link_name)

                if os.path.islink(link_path):
                    os.unlink(link_path)

                # Opening a directory is not possible on Windows, but that is not
                # a problem since Windows does not need to fsync the directory in
                # order to persist metadata.
                # Code adapted from https://github.com/NicolasLM/bplustree/commit/97de3d04022169f4ab0abd3d2ce97e5580bdff76
                if platform.system() == "Windows":
                    dir_fd = None
                else:
                    dir_fd = os.open(os.path.dirname(link_path), os.O_RDONLY)
                os.symlink(os.path.basename(checkpoint_path), link_name, dir_fd=dir_fd)

                if platform.system() != "Windows":
                    os.close(dir_fd)

        else:
            result = trainer.train_epoch()
            if trainer.REQ_EVAL:
                results_dirname = "results-val"
                os.makedirs(results_dirname, exist_ok=True)
                results_save_name = os.path.join(results_dirname, job_string)
                score = dataset.eval(result, results_save_name)
                print("The model has an evaluation score of {}.".format(score))
                file = open(eval_filename, "a")
                file.write("\nThe model has an evaluation score of {}.\n".format(score))
                file.close()

    print("\nThe best score is: {}\n".format(max_score))

    if not args.train and args.model == "sc":
        with open("results.json", "a") as file:
            json.dump(result, file)


if __name__ == "__main__":
    start_time = time.time()

    # Parse arguments
    args = utils.arg_parser.get_args()
    # Print arguments
    utils.arg_parser.print_args(args)

    if args.transfer_learning and not args.weights_ckpt:
        sys.exit("Be sure to pass pretrained weights for Transfer Learning.")

    device = torch.device("cuda:{}".format(args.cuda_device) if
            torch.cuda.is_available() and not args.disable_cuda else "cpu")

    job_string = time.strftime("{}-{}-D%Y-%m-%d-T%H-%M-%S-G{}".format(args.model, args.dataset, args.cuda_device))
    job_path = os.path.join(args.checkpoint_path, job_string)
    print("The current checkpoint is:", job_string)

    eval_filename = "evaluation_{}_{}.txt".format(args.model, args.dataset)

    # Create new checkpoint directory
    os.makedirs(job_path, exist_ok=True)

    # Save job arguments
    with open(os.path.join(job_path, "config.json"), "w") as f:
        json.dump(vars(args), f)

    # Data preparation
    print("Preparing Data ...", flush=True)
    split = get_split_str(args.train, bool(args.eval_ckpt), args.dataset)
    data_prep = DataPreparation(args.dataset, args.data_path)
    dataset, data_loader = data_prep.get_dataset_and_loader(split, args.pretrained_model,
            batch_size=args.batch_size, num_workers=args.num_workers)

    if args.train:
        val_dataset, val_data_loader = data_prep.get_dataset_and_loader("val",
                args.pretrained_model, batch_size=args.batch_size, num_workers=args.num_workers)

    # TODO: If eval + checkpoint load validation set

    # Get model lrcn, gve or sc from models/model_loader.py (depends on args)
    print("\nLoading Model ...", flush=True)
    ml = ModelLoader(args, dataset)
    model = getattr(ml, args.model)()
    print(model, "\n", flush=True)

    if args.transfer_learning and args.weights_ckpt:
        print("Initialize Transfer Learning")

    # TODO: Remove and handle with checkpoints
    if not args.train:
        print("Loading Model Weights ...", flush=True)
        evaluation_state_dict = torch.load(args.eval_ckpt)
        model_dict = model.state_dict(full_dict=True)
        model_dict.update(evaluation_state_dict)
        model.load_state_dict(model_dict)
        model.eval()

    if args.train:
        val_dataset.set_label_usage(dataset.return_labels)

    # Create logger
    logger = Logger(os.path.join(job_path, "logs"))

    if args.cross_validation:
        # Create copies of the datasets to use them for cross-validation
        original_dataset = copy.deepcopy(dataset)
        original_val_dataset = copy.deepcopy(val_dataset)

        n_splits = 5
        X, y = list(dataset.class_labels.keys()), list(dataset.class_labels.values())

        kf = KFold(n_splits=n_splits) if args.cross_validation == "kFold" else StratifiedKFold(n_splits=n_splits)
        split = kf.split(X) if args.cross_validation == "kFold" else kf.split(X, y)

        print("Use the same model for each cross-validation fold.\n") if args.cv_same_model else print("Use a new model for each cross-validation fold.\n")

        for fold, (train_index, val_index) in enumerate(split, 1):
            print("Fold {}:".format(fold))
            X_train = list(np.array(X)[train_index])
            y_train = list(np.array(y)[train_index])
            X_val = list(np.array(X)[val_index])
            y_val = list(np.array(y)[val_index])

            train_set = dict(zip(X_train, y_train))
            val_set = dict(zip(X_val, y_val))
            dataset.class_labels = train_set
            val_dataset.class_labels = val_set

            dataset.set_coco_anns_and_imgs(original_dataset)
            val_dataset.set_coco_anns_and_imgs(original_val_dataset)

            if not args.cv_same_model:
                ml = ModelLoader(args, dataset)
                model = getattr(ml, args.model)()

            train_and_eval(args, model, dataset, data_loader, val_dataset, val_data_loader, logger, device, job_path, eval_filename, fold)
    else:
        train_and_eval(args, model, dataset, data_loader, val_dataset, val_data_loader, logger, device, job_path, eval_filename, None)

    elapsed_time = time.time() - start_time
    time_string = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    days = int(elapsed_time / 86400)
    print("\nElapsed time:", days, "Days", time_string)

    file = open(eval_filename, "a")
    file.write("\nElapsed time: {} Days {}\n\n".format(days, time_string))
    file.close()
