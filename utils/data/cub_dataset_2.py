from .cub_dataset import CubDataset

class CubDataset2(CubDataset):

    """Splitted CUB Dataset (second part)"""

    image_features_path = 'CUB_feature_dict_2.p'
    caption_path = 'descriptions_bird.{}_2.fg.json'
    tokens_file_name = 'cub_tokens_{}_2.pkl'
    class_labels_path = 'CUB_label_dict_2.p'
