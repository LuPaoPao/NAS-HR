# -*- coding: UTF-8 -*-
import numpy as np
import os
from torch.utils.data import Dataset
import cv2
import csv
import scipy.io as scio
import torchvision.transforms.functional as transF
import torchvision.transforms as transforms
from PIL import Image

def transform(image):
    image = transF.resize(image, size=(300, 600))
    image = transF.to_tensor(image)
    image = transF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image

class Data_STMap(Dataset):
    def __init__(self, root_dir, frames_num, transform = None):
        self.root_dir = root_dir
        self.frames_num = int(frames_num)
        self.datalist = os.listdir(root_dir)
        self.num = len(self.datalist)
        self.transform = transform
        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        idx = idx
        img_name = 'STMap'
        STMap_name = 'STMap_YUV_Align_CSI_POS.png'
        nowPath = os.path.join(self.root_dir, self.datalist[idx])
        temp = scio.loadmat(nowPath)
        nowPath = str(temp['Path'][0])
        Step_Index = int(temp['Step_Index'])
        STMap_Path = os.path.join(nowPath, img_name)

        gt_name = 'Label_CSI/HR.mat'
        gt_path = os.path.join(nowPath, gt_name)
        gt = scio.loadmat(gt_path)['HR']
        gt = np.array(gt.astype('float32')).reshape(-1)
        gt = np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
        gt = gt.astype('float32')

        # 读取图片序列
        feature_map = cv2.imread(os.path.join(STMap_Path, STMap_name))
        feature_map = feature_map[:, Step_Index:Step_Index + self.frames_num, :]
        for c in range(feature_map.shape[2]):
            for r in range(feature_map.shape[0]):
                feature_map[r, :, c] = 255 * ((feature_map[r, :, c] - np.min(feature_map[r, :, c])) / (0.00001 +
                            np.max(feature_map[r, :, c]) - np.min(feature_map[r, :, c])))
        feature_map = Image.fromarray(feature_map)
        if self.transform:
            feature_map = self.transform(feature_map)
        # 归一化
        return (feature_map, gt)

    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True

def CrossValidation(root_dir, fold_num=5,fold_index=0):
    datalist = os.listdir(root_dir)
    # datalist.sort(key=lambda x: int(x))
    num = len(datalist)
    test_num = round(((num/fold_num) - 2))
    train_num = num - test_num
    test_index = datalist[fold_index*test_num:fold_index*test_num + test_num-1]
    train_index = datalist[0:fold_index*test_num] + datalist[fold_index*test_num + test_num:]
    return test_index, train_index

def getIndex(root_path, filesList, save_path, Pic_path, Step, frames_num):
    Index_path = []
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for sub_file in filesList:
        now = os.path.join(root_path, sub_file)
        img_path = os.path.join(now, os.path.join('STMap', Pic_path))
        temp = cv2.imread(img_path)
        Num = temp.shape[1]
        Res = Num - frames_num - 1  # 可能是Diff数据
        Step_num = int(Res/Step)
        for i in range(Step_num):
            Step_Index = i*Step
            temp_path = sub_file + '_' + str(1000 + i) + '_.mat'
            scio.savemat(os.path.join(save_path, temp_path), {'Path': now, 'Step_Index': Step_Index})
            Index_path.append(temp_path)
    return Index_path

