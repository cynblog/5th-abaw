import config
import random
import torch
import os
import sys
import glob
import math
import time

import numpy as np
import torch.utils.data as data

from PIL import Image, ImageDraw
from tqdm import tqdm
from collections import Counter
from torchvision import transforms


class dataset(data.Dataset):
    def __init__(self, label_root, data_root, mode='train'):
        super(dataset, self).__init__()
        self.mode = mode
        self.label_root = label_root
        self.data_root = data_root
        self.images = []
        self.labels = []
        self.p_labels = []
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        if mode == 'train':
            label_list = glob.glob(os.path.join(label_root, 'Train_Set') + "/*.txt")
        elif mode == 'val':
            label_list = glob.glob(os.path.join(label_root, 'Validation_Set') + "/*.txt")
        elif mode == 'test':
            with open(os.path.join(label_root, 'Test_Set', 'test.txt')) as f:
                label_list = f.readlines()
        else:
            print('数据集类型不存在')
            exit()
        # label_list = ['/data02/mxy/CVPR_ABAW/dataset/VA_Estimation_Challenge/Validation_Set/79-30-960x720.txt']
        # v = np.zeros(5)
        # a = np.zeros(5)
        if mode != 'test':
            for file_name in label_list:
                s_time = time.time()
                with open(file_name) as f:
                    lines = f.readlines()
                # print(time.time()-s_time)
                # image_list = glob.glob(os.path.join(data_root, file_name[:-4].split('/')[-1]) + '/*.jpg')
                image_list = os.listdir(os.path.join(data_root, file_name[:-4].split('/')[-1]))
                max_image = max(image_list)
                # print(time.time()-s_time)
                # exit()
                if int(max_image[:-4].split('/')[-1]) >= len(lines):
                    continue
                for image_name in image_list:
                    # try:
                    # label = [float(x) for x in lines[int(image_name[:-4].split('/')[-1])].split(',')]
                    if image_name[-3:] != 'jpg':
                        continue
                    label = [float(x) for x in lines[int(image_name[:-4])].split(',')]
                    p_label = [self.change_label(x) for x in label]
                    # p_label = label
                    # except:
                    #     print(int(image_list[-1][:-4].split('/')[-1]))
                    #     print(len(image_list) >= len(lines))
                    #     print(image_name)
                    #     exit()
                    # if -5 in label or p_label[0] != 3:
                    if -5 in label:
                        continue
                    self.images.append(os.path.join(data_root, file_name[:-4].split('/')[-1], image_name))
                    self.labels.append(label)
                    self.p_labels.append(p_label)
                    # v[p_label[0]] += 1
                    # a[p_label[1]] += 1
                # print(self.images[0])
                # print(time.time()-s_time)
                # exit()
            # print(v)
            # print(a)
        else:
            for file_name in label_list:
                # print(file_name)
                # exit()
                image_list = sorted(os.listdir(os.path.join(data_root, file_name[:-1])))
                for image_name in image_list:
                    if image_name[-3:] != 'jpg':
                        continue
                    self.images.append(os.path.join(data_root, file_name[:-1], image_name))
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img, label, p_label = self.pull_item(index)
        # print(target.shape)
        if self.mode == 'train':
            return img, label[0], label[1], p_label[0], p_label[1]
        elif self.mode == 'val':
            return img, self.images[index], label[0], label[1], p_label[0], p_label[1]
        else:
            return img, self.images[index][46:]

    
    def pull_item(self, index):  
        s_time = time.time() 
        image_path = self.images[index]
        image = Image.open(image_path)	
        image = self.transforms(image)
        if self.mode != 'test':
            label = self.labels[index]
            p_label = self.p_labels[index]
            return image, label, p_label
        else:
            return image, 0, 0

    @staticmethod
    def change_label(label):
        if label == 0:
            return 2
        elif label > 0:
            return round(label + 0.5) + 2
        else:
            return round(label - 0.5) + 2
    
        


if __name__ == '__main__':
    from config import cfg
    dataset = dataset(cfg.label_root, cfg.data_root, mode='test')
    for i in range(10):
        print(dataset.__getitem__(i))
    # print(dataset.change_label(-1))
    # print(dataset.change_label(-0.49))
    # print(dataset.change_label(0))
    # print(dataset.change_label(0.49))
    # print(dataset.change_label(1))