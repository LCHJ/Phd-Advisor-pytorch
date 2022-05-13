# -*- coding:utf-8 -*-
# @Time : 2020/12/30 11:46
# @Author: LCHJ
# @File : dataloader_adjacent.py

import os

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

from config import config


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


class DetectDataset(Dataset):
    
    def __init__(self, image_size, train_path, images_num):
        super(DetectDataset, self).__init__()
        
        self.train_path = train_path
        self.image_height = image_size[0]
        self.image_width = image_size[1]
        self.channel = image_size[2]
        self.input_shape = config.input_shape
        self.images_num = images_num
        
        self.train_dictionary = {}
        self._train_alphabets = []
        
        self._current_index = 0
        self._current_img_index = 0
        
        self.load_dataset()
    
    def __len__(self):
        
        return self.images_num
    
    def load_dataset(self):
        # 遍历dataset文件夹下面的m_mnist文件夹
        train_path = os.path.join(self.train_path)
        for character in os.listdir(train_path):
            # 遍历子种类。
            character_path = os.path.join(train_path, character)
            self.train_dictionary[character] = os.listdir(character_path)
        self._train_alphabets = list(self.train_dictionary.keys())
    
    def letterbox_image(self, image, size):
        image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        
        return new_image
    
    def get_random_data(self, image):
        # 图片预处理，归一化
        image_1 = self.letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        new_img = np.asarray(image_1).astype(np.float64) / 255
        if self.input_shape[-1] == 1:
            new_img = new_img[:, :, 0].reshape(new_img.shape[0], new_img.shape[1], -1)
        
        new_img = np.transpose(new_img, [2, 0, 1])
        
        return new_img
    
    def _convert_path_list_to_images_and_labels(self, path_list):
        number_of_pairs = int(len(path_list))
        pairs_of_images = np.zeros((number_of_pairs, self.channel, self.image_height, self.image_width))
        
        # img1
        image = Image.open(path_list[0])
        image = self.get_random_data(image)
        if self.channel == 1:
            pairs_of_images[0, 0, :, :] = image
        else:
            pairs_of_images[0, :, :, :] = image
        return pairs_of_images
    
    def __getitem__(self, index):
        # if self._current_index == 0:
        #     shuffle(self._train_alphabets)
        available_characters = self._train_alphabets
        number_of_characters = len(available_characters)
        # 除去小类别的名称
        current_character = available_characters[self._current_index]
        self._current_index = (int(self._current_index) + 1) % int(number_of_characters)
        
        batch_images_path = []
        labels = []
        # 获取当前这个小类别的路径
        image_path = os.path.join(self.train_path, current_character)
        available_images = os.listdir(image_path)
        number_of_image = len(available_images)
        index = int(index) % int(number_of_image)
        # 取当前类下的img
        image = os.path.join(image_path, available_images[index])
        batch_images_path.append(image)
        labels.append(int(current_character))
        
        images = self._convert_path_list_to_images_and_labels(batch_images_path)
        return images, labels


# DataLoader中collate_fn使用
def dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.concatenate(np.array(images), axis=0)
    bboxes = np.concatenate(np.array(bboxes), axis=0)
    return images, bboxes
