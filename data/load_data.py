import re

import torch
from torch.utils.data import *
from imutils import paths
import numpy as np
import random
import cv2
import os

CHARS = [
    # NUMBERS
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    # 자가용
    '가', '나', '다', '라', '마',
    '거', '너', '더', '러', '머', '버', '서', '어', '저',
    '고', '노', '도', '로', '모', '보', '소', '오', '조',
    '구', '누', '두', '루', '무', '부', '수', '우', '주',
    '기', '니', '디', '리', '미', '비', '시', '이', '지',
    # 영업용
    '바', '사', '아', '자',
    # 영업용(택배)
    '배',
    # 렌터카
    '하', '허', '호', '히',
    # 육군, 공군, 해군
    '육', '공', '해',
    # 국방부 및 직할부대 등
    '국', '합',
]

CHARS_DICT = {char: i for i, char in enumerate(CHARS)}


class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, PreprocFun=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        random.shuffle(self.img_paths)
        self.img_size = imgSize

        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        Image = cv2.imread(filename)
        height, width, _ = Image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)
        Image = self.PreprocFun(Image)

        basename = os.path.basename(filename)
        imgname, suffix = os.path.splitext(basename)
        imgname = imgname.split("-")[0].split("_")[0]
        label = list()
        for c in imgname:
            label.append(CHARS_DICT[c])

        if len(label) == 8 or len(label) == 7:
            if self.check(label) == False:
                print(imgname)
                assert 0, "Error label ^~^!!!"

        return Image, label, len(label)

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))

        return img

    def check(self, label):
        kor_plate_pattern = re.compile('[0-9]{2,3}[가-힣][0-9]{4}')
        plate_name = kor_plate_pattern.findall(''.join([CHARS[c] for c in label]))

        return True if plate_name else False


class ImgDataLoader(Dataset):
    def __init__(self, images, imgSize=(94, 24), PreprocFun=None):
        self.images = images
        self.img_size = imgSize

        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        Image = self.images[index]
        height, width, _ = Image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)
        Image = self.PreprocFun(Image)
        dummy = '00가0000'
        label = list()
        for c in dummy:
            label.append(CHARS_DICT[c])
        return Image, label, len(label)

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))

        return img

    def check(self, label):
        kor_plate_pattern = re.compile('[0-9]{2,3}[가-힣][0-9]{4}')
        plate_name = kor_plate_pattern.findall(''.join([CHARS[c] for c in label]))

        return True if plate_name else False


def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.float32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)


if __name__ == "__main__":

    dataset = LPRDataLoader(['validation'], (94, 24))
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2, collate_fn=collate_fn)
    print('data length is {}'.format(len(dataset)))
    for imgs, labels, lengths in dataloader:
        print('image batch shape is', imgs.shape)
        print('label batch shape is', labels.shape)
        print('label length is', len(lengths))
        break
