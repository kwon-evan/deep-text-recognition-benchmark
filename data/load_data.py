import re

import torch
from torch.utils.data import *
from imutils import paths
import numpy as np
import random
from cv2 import cv2
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
    '바', '사', '아', '자', '카', '파', '타', '차', 
    # 영업용(택배)
    '배',
    # 영업용(건설)
    '영',
    # 렌터카
    '하', '허', '호', '히',
    # 육군, 공군, 해군
    '육', '공', '해',
    # 국방부 및 직할부대 등
    '국', '합',
    # 도시
    '강원', '경기', '경남', '경북', '광주', '대구', '대전', '부산',
    '서울', '인천', '전남', '전북', '제주', '충남', '충북',
    # 도시 sub
    '강남', '강서', '계양', '고양', '관악', '광명', '구로', '금천', '김포', '남동', '동대문', '동작', '미추홀',
    '부천', '부평', '서대문', '서초', '안산', '안양', '양천', '연수', '영등포', '용산', '인천', '중',
]

CHARS_DICT = {char: i for i, char in enumerate(CHARS)}


def resize_pad(img, size):
    base_pic = np.zeros((size[1],size[0],3),np.uint8)
    pic1 = img
    h, w = pic1.shape[:2]
    ash = size[1] / h
    asw = size[0] / w

    if asw < ash:
        sizeas = (int(w * asw), int(h * asw))
    else:
        sizeas = (int(w * ash), int(h * ash))

    pic1 = cv2.resize(pic1, dsize=sizeas)
    base_pic[int(size[1] / 2 - sizeas[1] / 2):int(size[1] / 2 + sizeas[1] / 2),
            int(size[0] / 2 - sizeas[0] / 2):int(size[0] / 2 + sizeas[0] / 2),:] = pic1

    return base_pic

def encode(imgname: str):
    label = []

    i = 0
    while i < len(imgname):
        j = len(imgname)
        while i < j and not imgname[i:j] in CHARS:
            j -= 1

        if imgname[i:j] in CHARS:
            label.append(CHARS_DICT[imgname[i:j]])
            i=j
        else:
            assert 0, f'no such char in {imgname}'

    return label

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
            Image = cv2.resize(Image, self.img_size, interpolation=cv2.INTER_CUBIC)
        Image = self.PreprocFun(Image)

        basename = os.path.basename(filename)
        imgname, suffix = os.path.splitext(basename)
        imgname = imgname.split("-")[0].split("_")[0]
        label = encode(imgname)

        if label:
            if not self.check(label):
                assert 0, f"{imgname} <- Error label ^~^!!!"

        return Image, label, len(label)

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))

        return img

    def check(self, label):
        kor_plate_pattern = re.compile('[가-힣]{0,5}[0-9]{0,3}[가-힣][0-9]{4}')
        plate_name = kor_plate_pattern.findall(''.join([CHARS[c] for c in label]))

        return True if plate_name else False


class ImgDataLoader(Dataset):
    def __init__(self, images, imgSize=(100, 50), PreprocFun=None):
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
            Image = cv2.resize(Image, self.img_size, interpolation=cv2.INTER_CUBIC)
        Image = self.PreprocFun(Image)
        dummy = '00가0000'
        label = list()
        for c in dummy:
            label.append(CHARS_DICT[c])
        return Image, label, len(label)

    def transform(self, img):
        base_pic=np.zeros((self.img_size[1],self.img_size[0],3),np.uint8)
        pic1=img
        h,w=pic1.shape[:2]
        ash=self.img_size[1]/h
        asw=self.img_size[0]/w
        if asw<ash:
            sizeas=(int(w*asw),int(h*asw))
        else:
            sizeas=(int(w*ash),int(h*ash))
        pic1 = cv2.resize(pic1,dsize=sizeas)
        base_pic[int(self.img_size[1]/2-sizeas[1]/2):int(self.img_size[1]/2+sizeas[1]/2),
        int(self.img_size[0]/2-sizeas[0]/2):int(self.img_size[0]/2+sizeas[0]/2),:]=pic1
        
        img = base_pic
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

    dataset = LPRDataLoader(['valid'], (100, 50))
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2, collate_fn=collate_fn)
    print('data length is {}'.format(len(dataset)))
    for imgs, labels, lengths in dataloader:
        print('image batch shape is', imgs.shape)
        print('label batch shape is', labels.shape)
        print('label length is', len(lengths))
        break
