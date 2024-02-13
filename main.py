import torch
import torch.nn as nn
import numpy as np
from time import time
import os
import datetime
import matplotlib.pyplot as plt
import sys
import pandas as pd
import pickle
import copy

def run(lr,
        epochs,
        batch_size,
        Valid_split_size,
        X,
        Y,
        K=5,
        weights_dir='/content/MyDrive/MyDrive/SpeechProcessing/DNN/Transformer/MyModels/MultiHeadAttention/RAVDESS/K_Fold_Analyze/Learning_Details/', 
        file_name='K_fold_MAttention_4_seed_0_Mel64_revision', 
        auto_save_weight_acc=89, 
        save_weight_flag=True, 
        Seed=42):
  # LR = 9e-4
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print("Device: {:s}".format(device))
  
  model = FullModel().to(device)
  train = Trainer(model=model, lr=lr, weight_decay=1e-4)
  
  history = train.TrainKFold(X=X, 
                             Y=Y,
                             epochs=epochs, 
                             batch_size=batch_size,
                             Valid_split_size=Valid_split_size,
                             K=K,
                             save_weight_flag=True, 
                             auto_save_weight_acc=auto_save_weight_acc, 
                             save_path=weights_dir,
                             file_name=file_name,
                             Seed=Seed)
  return history


if __name__='__main__':
  run()
