### This script splits the CUB dataset from Hendricks et al. in a specific ratio.

import argparse
import os
import pandas as pd
import pickle


files = ['train', 'test', 'val']
pickle_files = ['CUB_feature_dict', 'CUB_label_dict']


parser = argparse.ArgumentParser()

parser.add_argument('--classes', type=str, default='./classes.txt',
                    help='path to the original classes.txt of the CUB_200_2011 dataset [default is ./classes.txt]')
parser.add_argument('--data', type=str, default='./data/cub',
                    help='name of the directory that contains the data files from Hendricks et al. [default is ./data/cub]')
parser.add_argument('--dir', type=str, default='./data/cub',
                    help='name of the directory that will contain the split data files [default is ./data/cub]')
parser.add_argument('--ratio', type=float, default=0.5,
                    help='ratio of splitting between 0 and 1 (e.g., use 0.25 if you want to use only a quarter) [default is 0.5]')

args = parser.parse_args()
class_file = args.classes
data_dir = args.data
dir = args.dir
ratio = args.ratio

if not os.path.exists(class_file):
    print(f'File {class_file} is not in this directory. You need the file in the same directory like this script.')
    quit()

if not (ratio > 0 and ratio < 1):
    print(f'A ratio of {ratio} is not valid. Please use a ratio greater than 0 or less than 1.')
    quit()

if not os.path.exists(dir):
   os.makedirs(dir)


# Split the classes with a ratio
classes = pd.read_csv(class_file, header=None, sep=' ')
num_classes = len(classes)

cut = int(ratio * num_classes)

split_1 = classes.sample(cut)
split_1 = split_1.sort_index()

split_2 = classes.drop(split_1.index)

split_1.to_csv(os.path.join(dir, 'split_1.txt'), header=None, index=None, sep=' ')
split_2.to_csv(os.path.join(dir, 'split_2.txt'), header=None, index=None, sep=' ')


# Split the train, test and val sets
classes_1 = tuple(list(split_1[1]))

for file in files:
    file_name = f'{file}.txt'
    file_path = os.path.join(data_dir, file_name)
    print(f'The current file is {file_path}.', flush=True)

    data_1 = []
    data_2 = []
    df = pd.read_csv(file_path, header=None)

    for data in df[0]:
        if data.startswith(classes_1):
            data_1.append(data)
        else:
            data_2.append(data)

    with open(os.path.join(dir, f'{file}_1.txt'), 'w') as f:
        f.write('\n'.join(data_1))
    with open(os.path.join(dir, f'{file}_2.txt'), 'w') as f:
        f.write('\n'.join(data_2))


# Split the CUB feature dict and CUB label dict
for file in pickle_files:
    file_name = f'{file}.p'
    file_path = os.path.join(data_dir, file_name)
    print(f'The current file is {file_path}.', flush=True)

    objects_1 = {}
    objects_2 = {}
    obj = pd.read_pickle(file_path)
    for key, value in obj.items():
        if key.startswith(classes_1):
            objects_1[key] = value
        else:
            objects_2[key] = value

    with open(os.path.join(dir, f'{file}_1.p'), 'wb') as f:
        pickle.dump(objects_1, f)
    with open(os.path.join(dir, f'{file}_2.p'), 'wb') as f:
        pickle.dump(objects_2, f)
