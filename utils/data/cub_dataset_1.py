from .cub_dataset import CubDataset

class CubDataset1(CubDataset):

    """Splitted CUB Dataset (first part)"""

    image_features_path = 'CUB_feature_dict_1.p'
    caption_path = 'descriptions_bird.{}_1.fg.json'
    tokens_file_name = 'cub_tokens_{}_1.pkl'
    class_labels_path = 'CUB_label_dict_1.p'
