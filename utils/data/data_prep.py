import os
import torch

from .coco_dataset import CocoDataset
from .cub_dataset import CubDataset
from .cub_dataset_1 import CubDataset1
from .cub_dataset_2 import CubDataset2
from utils.transform import get_transform

class DataPreparation:
    def __init__(self, dataset_name='coco', data_path='./data'):
        if dataset_name == 'coco':
            self.DatasetClass = CocoDataset
        elif dataset_name == 'cub':
            self.DatasetClass = CubDataset
        elif dataset_name == 'cub1':
            self.DatasetClass = CubDataset1
        elif dataset_name == 'cub2':
            self.DatasetClass = CubDataset2
        self.data_path = os.path.join(data_path, self.DatasetClass.dataset_prefix)

    def get_dataset(self, split='train', vision_model=None, vocab=None,
            tokens=None):
        transform = get_transform(vision_model, split)
        dataset = self.DatasetClass(root=self.data_path,
                                    split=split,
                                    vocab=vocab,
                                    tokenized_captions=tokens,
                                    transform=transform)
        self.dataset = dataset
        return self.dataset

    def get_loader(self, dataset, batch_size=128, num_workers=4):
        assert isinstance(dataset, self.DatasetClass)

        if dataset.split == 'train':
            shuffle = True
        else:
            shuffle = False

        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  collate_fn=dataset.collate_fn)

        return data_loader

    def get_dataset_and_loader(self, split='train', vision_model=None,
            vocab=None, tokens=None, batch_size=128, num_workers=4):
        dataset = self.get_dataset(split, vision_model, vocab, tokens)
        loader = self.get_loader(dataset, batch_size, num_workers)
        return dataset, loader
