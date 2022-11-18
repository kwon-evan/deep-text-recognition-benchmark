#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:14:16 2019

@author: xingyu
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor


class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.Mish(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.Mish(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.Mish(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class LPRNet(nn.Module):
    def __init__(self, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),                # 0
            nn.BatchNorm2d(num_features=64),
            nn.Mish(),                                                                         # 2
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            small_basic_block(ch_in=64, ch_out=128),                                           # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.Mish(),                                                                         # 6
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            small_basic_block(ch_in=64, ch_out=256),                                           # 8
            nn.BatchNorm2d(num_features=256),
            nn.Mish(),                                                                         # 10
            small_basic_block(ch_in=256, ch_out=256),                                          # *** 11 ***
            nn.BatchNorm2d(num_features=256),                                                  # 12
            nn.Mish(),                                                                         # 13
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 2, 2)),                             # 14
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(2, 4), stride=1),         # 16
            nn.BatchNorm2d(num_features=256),
            nn.Mish(),                                                                         # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(12, 2), stride=1), # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.Mish(),                                                                         # *** 22 ***
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=256 + class_num + 128 + 64, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:  # [2, 4, 8, 11, 22]
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(5, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)

        return logits


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

if __name__ == "__main__":
    from torchsummary import summary
    import torch
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=1).to(device)
    tensor = torch.rand((1, 3, 50, 100)).to(device)

    summary(lprnet, (3, 50, 100), device=device)

    t0 = time.time()
    lprnet(tensor)
    t1 = time.time()

    print("inference time: ", (t1 - t0) * 1000, "ms")
