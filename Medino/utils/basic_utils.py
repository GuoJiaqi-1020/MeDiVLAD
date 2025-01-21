import argparse
import glob
import os
import re
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

exp_root = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root)

def find_best_checkpoint(trained_dir: str = None):
    files = glob.glob(f"{trained_dir}/*.pth.tar", recursive=True)
    pattern = re.compile(r"_acc_([0-9]+(?:\.[0-9]+)?)\.pth\.tar$")
    max_acc = 0
    best_file = ""
    for file in files:
        match = pattern.search(file)
        if match:
            acc = float(match.group(1))
            if acc > max_acc:
                max_acc = acc
                best_file = file
    if best_file:
        return best_file
    else:
        return None


def combine_class(c):
    if c > 1:
        return c - 1
    else:
        return c


def sampling_prob(training_set):
    class_list = []
    lab_dis = []
    for i in range(len(training_set)):
        class_list.append(combine_class(int(training_set.name_list[i].split('.')[0][-1])))
    counter_class = dict(Counter(class_list).items())
    for i in range(len(set(class_list))):
        if i in class_list:
            lab_dis.append(counter_class[i])
    reci_counts = [1 / x for x in lab_dis]
    CE_weight = torch.tensor([num / sum(reci_counts) for num in reci_counts])
    return CE_weight


def Plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Blues, save_path=None):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')

    plt.show()

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        arg_name = f"--{k}"
        # Check if this argument is already added to the parser
        if any(arg_name == action.option_strings[0] for action in parser._actions):
            print(f"Argument {arg_name} already exists. Skipping...")
            continue

        # Determine the type of the argument
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool

        # Add the argument to the parser
        parser.add_argument(arg_name, default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def check_lus_data(root):
    stem_data_dir = os.path.join(root, 'data')
    if os.path.exists(stem_data_dir) and not os.path.exists(os.path.join('LUS_dataset', '/data')):
        return os.path.join(root, 'data'), False
    else:  # Running on the server
        return os.path.join('/data', 'LUS_dataset'), True



if __name__ == '__main__':
    aaa = find_best_checkpoint()
    print(aaa)
