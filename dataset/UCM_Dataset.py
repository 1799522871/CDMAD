import argparse
import logging
import math
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from dataset.augmentations import RandAugment
from dataset.augmentations import CutoutDefault
from scipy.stats import pareto

# from .randaugment import RandAugmentMC

# train_dir = 'D:/codestudy/MyPythonProject/remote-sesing/Transfer-Learning-Library/examples/semi_supervised_learning/image_classification/data/UCM_split/train/'
# test_dir = 'D:/codestudy/MyPythonProject/remote-sesing/Transfer-Learning-Library/examples/semi_supervised_learning/image_classification/data/UCM_split/test/'
# train_dir = 'C:\\Users\\hyq\\Desktop\\kk\\Transfer-Learning-Library\\examples\\semi_supervised_learning\\image_classification\\data\\UCM_split\\train'
# val_dir = 'C:\\Users\\hyq\\Desktop\\kk\\Transfer-Learning-Library\\examples\\semi_supervised_learning\\image_classification\\data\\UCM_split\\val'
# test_dir = 'C:\\Users\\hyq\\Desktop\\kk\\Transfer-Learning-Library\\examples\\semi_supervised_learning\\image_classification\\data\\UCM_split\\test'


train_dir = 'D:\\codestudy\\MyPythonProject\\remote-sesing\\imbalanced classification\\CDMAD\\data\\PatternNet_split_TL\\train'
test_dir = 'D:\\codestudy\\MyPythonProject\\remote-sesing\\imbalanced classification\\CDMAD\\data\\PatternNet_split_TL\\test'

normal_mean = (0.3588, 0.35994354, 0.31862667)
normal_std = (0.14776626, 0.14179754, 0.13599195)


# Augmentations.
transform_train = transforms.Compose([
        transforms.RandomCrop(256, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(normal_mean, normal_std)
    ])

transform_strong = transforms.Compose([
    transforms.RandomCrop(256, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(normal_mean, normal_std)
])
transform_strong.transforms.insert(0, RandAugment(3, 4))
transform_strong.transforms.append(CutoutDefault(16))

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(normal_mean, normal_std)
])

class TransformTwice:
    def __init__(self, transform, transform2):
        self.transform = transform
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform2(inp)
        out3 = self.transform2(inp)
        return out1, out2, out3

class TransformDoub:
    def __init__(self, transform, transform2):
        self.transform = transform
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform2(inp)
        return out1, out2

def get_UCM(root, test_dir, l_samples, u_samples, transform_train=transform_train, transform_strong=transform_strong, val_transform=transform_val, rand_number=0):

    base_dataset = datasets.ImageFolder(root=root, transform=None)

    train_labeled_idxs, train_unlabeled_idxs = train_split(base_dataset.targets, l_samples, u_samples, rand_number)
    # train_labeled_idxs, train_unlabeled_idxs = x_u_split(
    #     args, base_dataset.targets) # base_dataset.targets 获取标签

    train_labeled_dataset = UCMSSL(
        root=root, indexs=train_labeled_idxs,
        transform=transform_train)

    train_unlabeled_dataset = UCMSSL(
        root=root, indexs=train_unlabeled_idxs,
        transform=TransformTwice(transform_train, transform_strong))

    # val_dataset = UCMSSL_test(root=val_dir,transform=val_transform)

    test_dataset = UCMSSL_test(root=test_dir,transform=val_transform)

    return train_labeled_dataset, train_unlabeled_dataset,  test_dataset

def train_split(labels, n_labeled_per_class, n_unlabeled_per_class,rand_number):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    print(rand_number)
    np.random.seed(rand_number)
    for i in range(10):
        idxs = np.where(labels == i)[0]
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
        train_unlabeled_idxs.extend(idxs[:n_labeled_per_class[i] + n_unlabeled_per_class[i]])

    return train_labeled_idxs, train_unlabeled_idxs

# def x_u_split(args, labels):
#     label_per_class = args.num_labeled // args.num_classes
#     labels = np.array(labels)
#     labeled_idx = []
#     # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
#     unlabeled_idx = np.array(range(len(labels)))
#     for i in range(args.num_classes):
#         idx = np.where(labels == i)[0]
#         idx = np.random.choice(idx, label_per_class, False)
#         labeled_idx.extend(idx)
#     labeled_idx = np.array(labeled_idx)
#     assert len(labeled_idx) == args.num_labeled
#
#     # if args.expand_labels or args.num_labeled < args.batch_size:
#     #     num_expand_x = math.ceil(
#     #         args.batch_size * args.eval_step / args.num_labeled)
#     #     labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
#     np.random.shuffle(labeled_idx)
#     return labeled_idx, unlabeled_idx

# class TransformFixMatch_RS(object):
#     def __init__(self, mean, std):
#         self.weak = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomCrop(size=256,
#                                   padding=int(256 * 0.125),
#                                   padding_mode='reflect')])
#         self.strong = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomCrop(size=256,
#                                   padding=int(256 * 0.125),
#                                   padding_mode='reflect'),
#             RandAugmentMC(n=2, m=10)])
#         self.normalize = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=mean, std=std)])
#
#     def __call__(self, x):
#         weak = self.weak(x)
#         strong = self.strong(x)
#         return self.normalize(weak), self.normalize(strong)


class UCMSSL(Dataset):
    def __init__(self, root, indexs,transform=None,
                    target_transform=None):
        basedata = datasets.ImageFolder(root,transform=None)
        list = []
        if indexs is not None:
            # for i in indexs:
            #     list.append(np.array(Image.open(basedata.imgs[i][0])))
            for i in indexs:
                image = Image.open(basedata.imgs[i][0])
                resized_image = image.resize((256, 256))
                array = np.array(resized_image)
                list.append(array)

            self.targets = np.array(basedata.targets)[indexs]
            self.classes = basedata.classes  # Add this line
            self.class_to_idx = basedata.class_to_idx
            # self.imgs = np.array(basedata.imgs)[indexs]

            # 看的Dataset.cifar10
            # self.data = np.vstack(list).reshape(-1, 3, 256, 256)
            self.data = np.stack(list, axis = 0)
            # self.data = self.data.transpose((0, 2, 3, 1))
            print('train:{}'.format(self.data.shape))

            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(img)

        return img, target, index
    def __len__(self):
        return len(self.data)

class UCMSSL_test(Dataset):
    def __init__(self, root,
                 transform=None, target_transform=None):
        basedata = datasets.ImageFolder(root,transform=None)
        list = []
        for i in range(len(basedata.imgs)):
            image = Image.open(basedata.imgs[i][0])
            resized_image = image.resize((256, 256))
            array = np.array(resized_image)
            list.append(array)


        self.targets = basedata.targets
        self.classes = basedata.classes  # Add this line
        self.class_to_idx = basedata.class_to_idx
        self.imgs = basedata.imgs

        # 看的Dataset.cifar100
        # self.data = np.vstack(list).reshape(-1, 3, 256, 256)
        self.data =  np.stack(list,axis=0)
        # self.data = self.data.transpose((0, 2, 3, 1))
        print('test:{}'.format(self.data.shape))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    def __len__(self):
        return len(self.data)

def make_imb_data(max_num, class_num, gamma,imb):
    if imb == 'long':
        mu = np.power(2/gamma, 1/(class_num - 1))
        class_num_list = []
        for i in range(class_num):
            if i == (class_num - 1):
                class_num_list.append(int(max_num / gamma))
            else:
                class_num_list.append(int(max_num * np.power(mu, i)))
        print(class_num_list)
    if imb=='step':
        class_num_list = []
        for i in range(class_num):
            if i < int((class_num) / 2):
                class_num_list.append(int(max_num))
            else:
                class_num_list.append(int(max_num / gamma))
        print(class_num_list)

    return list(class_num_list)


if __name__ == '__main__':
    num_classes = 38
    N_SAMPLES_PER_CLASS = make_imb_data(200, num_classes, 100, 'long')
    U_SAMPLES_PER_CLASS = make_imb_data(400, num_classes, 100, 'long')

    l_samples = N_SAMPLES_PER_CLASS
    u_samples = N_SAMPLES_PER_CLASS
    rand_number = 0



    train_ds_l, train_ds_ul, test_ds = get_UCM(train_dir, test_dir, l_samples, u_samples, transform_train, transform_strong, transform_val, rand_number)
    print('=====================')

