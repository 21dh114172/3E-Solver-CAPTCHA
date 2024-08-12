import numpy as np
import string
import glob
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from augment import AugmentData
import random
import sys
import torch
import json

def load_datasets(args):
    label_dict = get_label_dict(args)

    train_filenames = glob.glob("./dataset/" + args.dataset + "/train/*.*")
    train_filenames = [train_filename for train_filename in train_filenames if
                       train_filename.split("/")[-1] in label_dict]
    test_filenames = glob.glob("./dataset/" + args.dataset + "/test/*.*")

    dataloader_train, id2token = get_dataloader(train_filenames, label_dict, args, train=True, label=True)
    dataloader_test, _ = get_dataloader(test_filenames, label_dict, args, train=False, label=True)

    MAXLEN = max([len(value) for value in label_dict.values()])
    return dataloader_train, dataloader_test, id2token, MAXLEN + 2


def load_datasets_mean_teacher(args):
    label_dict = get_label_dict(args)

    labeled_train_filenames = glob.glob("./dataset/" + args.dataset + "/train/*.*")
    labeled_train_filenames = [train_filename for train_filename in labeled_train_filenames if
                               train_filename.split("/")[-1] in label_dict]
    nolabeled_train_filenames = glob.glob("./dataset/" + args.dataset + "/buchong/*.*")
    nolabeled_train_filenames = random.sample(nolabeled_train_filenames, args.unlabeled_number) \
                                + labeled_train_filenames
    test_filenames = glob.glob("./dataset/" + args.dataset + "/test/*.*")

    labeled_train_filenames = sorted(labeled_train_filenames)
    nolabeled_train_filenames = sorted(nolabeled_train_filenames)
    test_filenames = sorted(test_filenames)

    print(len(labeled_train_filenames))
    print(len(nolabeled_train_filenames))
    print(len(test_filenames))

    dataloader_train_nolabeled, _ = get_dataloader(nolabeled_train_filenames, label_dict, args, train=True, label=False)
    dataloader_test, _ = get_dataloader(test_filenames, label_dict, args, train=False, label=True)

    dataloader_train_labeled, id2token = get_dataloader(labeled_train_filenames, label_dict, args, train=True,
                                                        label=True, loader_len=len(dataloader_train_nolabeled))

    MAXLEN = max([len(value) for value in label_dict.values()])
    MINLEN = min([len(value) for value in label_dict.values()])
    return dataloader_train_labeled, dataloader_train_nolabeled, dataloader_test, id2token, MAXLEN + 2, MINLEN + 2


def get_label_dict(args):
    delimiter = args.delimiter_label
    label_path = './dataset/' + args.dataset + '/label/' + args.label
    f = open(label_path, 'r')
    lines = f.read().strip().split("\n")
    label_dict = {line.split(delimiter)[0]: line.split(delimiter)[1] for line in lines}
    f.close()
    return label_dict


def get_vocab(label_dict):
    all_labels = "".join([value for value in label_dict.values()])
    vocab = set(all_labels)
    ordered_vocab = sorted(vocab, key=lambda x: (x.isdigit(), x.islower(), x))
    
    return "".join(ordered_vocab)


def get_dataloader(filenames, label_dict, args, train, label, loader_len=None):
    if train and label:
        assert loader_len is not None
    else:
        assert loader_len is None

    MAXLEN = max([len(value) for value in label_dict.values()])
    TARGET_HEIGHT = args.TARGET_HEIGHT
    TARGET_WIDTH = args.TARGET_WIDTH

    vocab = get_vocab(label_dict)
    if not args.use_new_label_dict:
        checkpoint = torch.load(args.load_model)
        loaded_vocab = checkpoint.get("vocab", "")
        if loaded_vocab != "":
            vocab = loaded_vocab
            print("Loaded vocab from previous model \n")
        else:
            print("Create new vocab from current dataset \n")
    vocab += ' '

    id2token = {k + 1: v for k, v in enumerate(vocab)}
    id2token[0] = '^'
    id2token[len(vocab) + 1] = '$'
    token2id = {v: k for k, v in id2token.items()}

    img_buffer = np.zeros((len(filenames), TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
    text_buffer = []
    isPrint = False
    for i, filename in enumerate(filenames):
        captcha_image = Image.open(filename).resize((TARGET_WIDTH, TARGET_HEIGHT), Image.ANTIALIAS)
        
        if captcha_image.mode != 'RGB':
            captcha_image = captcha_image.convert("RGB")
        captcha_array = np.array(captcha_image)
        
        img_buffer[i] = captcha_array
        if train:
            if label:
                #text = label_dict[filename.split("/")[-1]]
                text = label_dict.get(filename.split("/")[-1], filename.split("/")[-1].split(".")[0])
                if not isPrint:
                    isPrint = True
                    print("Train label:")
                    print(filename.split("/")[-1])
                    print(filename.split("/")[-1])
        else:
            #text = filename.split("/")[-1].split(".")[0] # get label from file name
            try:
                #text = label_dict[filename.split("/")[-1]]
                text = label_dict.get(filename.split("/")[-1], filename.split("/")[-1].split(".")[0])
            except:
                print("filename: ")
                print(filename)
                print("label: ")
                print(label)
                print("file_name_split: ")
                print(filename.split("/")[-1])
                print(label_dict)
                sys.exit(1)
            

        if label:
            text = ("^" + text + "$")
            try:
                text_buffer.append([token2id[i] for i in text.ljust(MAXLEN + 2)])
            except:
                print("Token 2 id: ")
                print(token2id)
                print("Text: ")
                print(text)
                print("Vocab: ")
                print(vocab)
                print("Label dict: ")
                print(label_dict)
                sys.exit(1)
        else:
            text_buffer.append([-1] * (MAXLEN + 2))

    image = img_buffer.astype(np.uint8)
    text = np.array(text_buffer)

    if label:
        batch_size = args.batch_size
    else:
        batch_size = args.secondary_batch_size

    if not label:
        dataset = UnlabelData(image, TARGET_HEIGHT, TARGET_WIDTH)
        dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True, num_workers=4)
    elif train:
        dataset = LabelData(image, text, loader_len * batch_size, TARGET_HEIGHT, TARGET_WIDTH)
        dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True, num_workers=4)
    else:
        dataset = TestData(image, text)
        dataloader = DataLoader(dataset, batch_size, shuffle=False, drop_last=False, num_workers=4)
    return dataloader, id2token


class LabelData(Dataset):
    def __init__(self, image, text, dataset_len, image_height, image_width):
        self.image = image
        self.text = text
        self.dataset_len = dataset_len
        self.augument = AugmentData(image_height, image_width)

    def __len__(self):
        return int(self.dataset_len)

    def __getitem__(self, index):
        index = index % len(self.image)
        img = Image.fromarray(self.image[index])
        img = AugmentData.normalize(self.augument.weak(img))
        lb = self.text[index]

        sample = (img, lb)
        return sample


class UnlabelData(Dataset):
    def __init__(self, image, image_height, image_width):
        self.image = image
        self.augument = AugmentData(image_height, image_width)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        img = Image.fromarray(self.image[index])
        img_1 = AugmentData.normalize(self.augument.weak(img))
        img_2 = AugmentData.normalize(self.augument.strong(img))

        sample = (img_1, img_2)
        return sample


class TestData(Dataset):
    def __init__(self, image, text):
        self.image = image
        self.text = text

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        img = Image.fromarray(self.image[index])
        img = AugmentData.normalize(img)
        lb = self.text[index]

        sample = (img, lb)
        return sample
