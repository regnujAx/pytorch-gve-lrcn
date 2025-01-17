import torch
import torch.nn as nn
import numpy as np

class SCTrainer:

    REQ_EVAL = False

    def __init__(self, args, model, dataset, data_loader, logger, device, checkpoint=None):
        self.model = model
        self.dataset = dataset
        self.data_loader = data_loader
        self.train = args.train
        self.logger = logger
        self.device = device
        self.eval_filename = "evaluation_{}_{}.txt".format(args.model, args.dataset)

        model.to(self.device)

        # TODO: Implement checkpoint recovery
        if checkpoint is None:
            self.criterion = nn.CrossEntropyLoss()
            self.params = filter(lambda p: p.requires_grad, model.parameters())
            self.optimizer = torch.optim.Adam(self.params, lr=args.learning_rate)
            self.total_steps = len(data_loader)
            self.num_epochs = args.num_epochs
            self.log_step = args.log_step
            self.curr_epoch = 0

    def train_epoch(self):
        # Result is list of losses during training
        # and generated captions during evaluation
        result = []

        for i, (images, word_inputs, word_targets, lengths, ids, labels) in enumerate(self.data_loader):
            # Mapping of class labels for correct calculations
            classes = list(set(map(int, self.dataset.class_labels.values())))
            labels_list = labels.tolist()
            mapped_labels = []
            for index in labels_list:
                mapped_labels.append(classes.index(index + 1))

            labels_tensor = torch.Tensor(mapped_labels)
            labels = labels_tensor.type(torch.LongTensor)

            # Prepare mini-batch dataset
            if self.train:
                word_targets = word_targets.to(self.device)
                labels = labels.to(self.device)

                loss = self.train_step(word_targets, labels, lengths)
                result.append(loss.data.item())

                step = self.curr_epoch * self.total_steps + i + 1
                self.logger.scalar_summary('batch_loss', loss.data.item(), step)

            else:
                word_targets= word_targets.to(self.device)
                labels = labels.to(self.device)
                score = self.eval_step(word_targets, labels, lengths)
                result.append(score)

            # TODO: Add proper logging
            # Print log info
            if i % self.log_step == 0:
                print("Epoch [{}/{}], Step [{}/{}]".format(self.curr_epoch,
                    self.num_epochs, i, self.total_steps), end='', flush=True)
                if self.train:
                    print(", Loss: {:.4f}, Perplexity: {:5.4f}".format(loss.data.item(),
                                np.exp(loss.data.item())), end='', flush=True)
                    file = open(self.eval_filename, 'a')
                    file.write("\nEpoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}".format(
                                self.curr_epoch, self.num_epochs, i, self.total_steps, loss.data.item(),
                                np.exp(loss.data.item())))
                    file.close()
                print()

        self.curr_epoch += 1

        if self.train:
            self.logger.scalar_summary('epoch_loss', np.mean(result), self.curr_epoch)
        else:
            result = np.sum(result, axis=0)
            result = result[1] / result[0]
            print("Evaluation Accuracy: {}".format(result), flush=True)
            file = open(self.eval_filename, 'a')
            file.write("\nEvaluation Accuracy: {}".format(result))
            file.close()

        return result

    def train_step(self, word_targets, class_labels, lengths):
        # Forward, Backward and Optimize
        self.model.zero_grad()
        outputs = self.model(word_targets, lengths)
        loss = self.criterion(outputs, class_labels)
        loss.backward()
        self.optimizer.step()

        return loss

    def eval_step(self, word_targets, class_labels, lengths):
        outputs = self.model(word_targets, lengths)
        _, predicted = torch.max(outputs.data, 1)

        return [class_labels.size(0), (predicted == class_labels).sum().item()]
