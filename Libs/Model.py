import torch
import torch.nn as nn
import numpy as np
from time import time
import os
import datetime
import matplotlib.pyplot as plt

class FeatureExtraction(nn.Module):
  def __init__(self):
    super().__init__()

    cnn = nn.Conv2d(1, 32, 3, 1, padding='same')
    bnormal = nn.BatchNorm2d(32)
    act = nn.ReLU()
    maxpool = nn.AvgPool2d(2,2)
    drop = nn.Dropout(0.4)
    self.Block1 = nn.Sequential(cnn, bnormal, act, maxpool, drop)

    cnn = nn.Conv2d(32, 64, 3, 1, padding='same')
    bnormal = nn.BatchNorm2d(64)
    act = nn.ReLU()
    maxpool = nn.AvgPool2d(2,2)
    drop = nn.Dropout(0.4)
    self.Block2 = nn.Sequential(cnn, bnormal, act, maxpool, drop)

    cnn = nn.Conv2d(64, 128, 3, 1, padding='same')
    bnormal = nn.BatchNorm2d(128)
    act = nn.ReLU()
    maxpool = nn.AvgPool2d(2,2)
    drop = nn.Dropout(0.4)
    self.Block3 = nn.Sequential(cnn, bnormal, act, maxpool, drop)

    cnn = nn.Conv2d(128, 256, 3, 1, padding='same')
    bnormal = nn.BatchNorm2d(256)
    act = nn.ReLU()
    maxpool = nn.AvgPool2d(2,2)
    drop = nn.Dropout(0.4)
    self.Block4 = nn.Sequential(cnn, bnormal, act, maxpool, drop)

    # cnn = nn.Conv2d(128, 256, 3, 1, padding='same')
    # bnormal = nn.BatchNorm2d(256)
    # act = nn.ReLU()
    # maxpool = nn.AvgPool2d(2,2)
    # drop = nn.Dropout(0.3)
    # self.Block5 = nn.Sequential(cnn, bnormal, act, maxpool, drop)

    # self.MHead4 = nn.MultiheadAttention(4*32, 4, dropout=0, batch_first=True)
    # self.MHead5 = nn.MultiheadAttention(2*16, 4, dropout=0, batch_first=True)

  def forward(self, x):
    x = self.Block1(x)
    x = self.Block2(x)
    x = self.Block3(x)
    x = self.Block4(x)
    return x

class BottleNeck(nn.Module):
  def __init__(self):
    super().__init__()

    self.MHead1 = nn.MultiheadAttention(116, 4, dropout=0.2, batch_first=True)
    self.BNorm1 = nn.BatchNorm1d(256)
    self.MHead2 = nn.MultiheadAttention(116, 4, dropout=0.2, batch_first=True)
    self.BNorm2 = nn.BatchNorm1d(256)
    self.MHead3 = nn.MultiheadAttention(116, 2, dropout=0.2, batch_first=True)
    self.BNorm3 = nn.BatchNorm1d(256)
    self.MHead4 = nn.MultiheadAttention(116, 2, dropout=0.2, batch_first=True)
    self.BNorm4 = nn.BatchNorm1d(256)

    self.max2 = nn.MaxPool1d(2,2)

  def forward(self, x):
    xh, _ = self.MHead1(x, x, x)
    x = x + xh
    x = self.BNorm1(x)

    xh, _ = self.MHead2(x, x, x)
    x = x + xh
    x = self.BNorm2(x)

    x = self.max2(x)
    x = self.max2(x)
    return x

class Head(nn.Module):
  def __init__(self):
    super().__init__()

    act = nn.ReLU()

    F = nn.Linear(256*29, 128)
    BNorm = nn.BatchNorm1d(128)
    self.FC1 = nn.Sequential(F, BNorm, act)

    F = nn.Linear(128, 32)
    BNorm = nn.BatchNorm1d(32)
    self.FC2 = nn.Sequential(F, BNorm, act)

    F = nn.Linear(32, 8)
    BNorm = nn.BatchNorm1d(8)
    self.FC3 = nn.Sequential(F, BNorm)

    self.Sig = nn.Sigmoid()

  def forward(self, x):
    x = self.FC1(x)
    x = self.FC2(x)
    x = self.FC3(x)
    x = self.Sig(x)
    return x

class FullModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.featureextarction_model = FeatureExtraction()
    self.bottleneck_model = BottleNeck()
    self.head_model = Head()
    self.save_bottleneck = None

  def forward(self, x):
    x = self.featureextarction_model(x)
    x = torch.flatten(x, start_dim=-2)
    x = self.bottleneck_model(x)
    self.save_bottleneck = x
    x = torch.flatten(x, start_dim=-2)
    x = self.head_model(x)
    return x
