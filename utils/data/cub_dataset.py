import os
import pickle
import torch

from .coco_dataset import CocoDataset

# Adapted from
# https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning
class CubDataset(CocoDataset):

    """CUB Custom Dataset compatible with torch.utils.data.DataLoader."""

    dataset_prefix = 'cub'
    image_path = ''
    image_features_path = 'CUB_feature_dict.p'
    caption_path = 'descriptions_bird.{}.fg.json'
    vocab_file_name = 'cub_vocab.pkl'
    tokens_file_name = 'cub_tokens_{}.pkl'
    class_labels_path = 'CUB_label_dict.p'

    # Available data splits (must contain 'train')
    DATA_SPLITS = set(['train', 'val', 'test'])

    def __init__(self, root, split='train', vocab=None, tokenized_captions=None,
            transform=None, use_image_features=True):
        super().__init__(root, split, vocab, tokenized_captions, transform)

        cls = self.__class__
        self.img_features_path= os.path.join(self.root, cls.image_features_path)

        if use_image_features:
            self.load_img_features(self.img_features_path)
            self.input_size = next(iter(self.img_features.values())).shape[0]

    def load_img_features(self, img_features_path):
        with open(img_features_path, 'rb') as f:
            feature_dict = pickle.load(f, encoding='latin1')
        self.img_features = feature_dict

    def load_class_labels(self, class_labels_path):
        with open(class_labels_path, 'rb') as f:
            label_dict = pickle.load(f, encoding='latin1')

        self.num_classes = len(set(label_dict.values()))
        self.class_labels = label_dict

    def get_image(self, img_id):
        if self.img_features is not None:
            image = self.img_features[img_id]
            image = torch.Tensor(image)
        else:
            image = super().get_image(img_id)
        return image

    def get_class_label(self, img_id):
        class_label = torch.LongTensor([int(self.class_labels[img_id])-1])
        return class_label
