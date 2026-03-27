import csv
import math
import os
import random
import copy
import numpy as np
import torch
import torch.nn.functional
import torchaudio
from PIL import Image
from scipy import signal
from torch.utils.data import Dataset
from torchvision import transforms
import pickle
    


class AVDataset_CD(Dataset):
  def __init__(self, mode='train'):
    classes = []
    self.data = []
    data2class = {}

    self.mode=mode
    self.visual_path = '/data/CREMA-D/Image-01-FPS/'
    self.audio_path = '/data/CREMA-D/Audio-1004/'
    self.stat_path = '/data/CREMA-D/stat.csv'
    self.train_txt = '/data/CREMA-D/train.csv'
    self.test_txt = '/data/CREMA-D/test.csv'
    if mode == 'train':
        csv_file = self.train_txt
    else:
        csv_file = self.test_txt

    
    with open(self.stat_path, encoding='UTF-8-sig') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                classes.append(row[0])
    
    with open(csv_file) as f:
      csv_reader = csv.reader(f)
      for item in csv_reader:
        if item[1] in classes and os.path.exists(self.audio_path + item[0] + '.pkl') and os.path.exists(
                        self.visual_path + '/' + item[0]):
            self.data.append(item[0])
            data2class[item[0]] = item[1]

    print('data load over')
    print(len(self.data))
    
    self.classes = sorted(classes)

    self.data2class = data2class
    self._init_atransform()
    print('# of files = %d ' % len(self.data))
    print('# of classes = %d' % len(self.classes))

    #Audio
    self.class_num = len(self.classes)

  def _init_atransform(self):
    self.aid_transform = transforms.Compose([transforms.ToTensor()])

  def __len__(self):
    return len(self.data)

  
  def __getitem__(self, idx):
    datum = self.data[idx]

    # Audio
    # fbank = torch.tensor(pickle.load(open(self.audio_path + datum + '.pkl', 'rb'))).float().unsqueeze(0)
    fbank = torch.tensor(pickle.load(open(self.audio_path + datum + '.pkl', 'rb'))).float().unsqueeze(0)
    # Visual
    if self.mode == 'train':
        transf = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transf = transforms.Compose([
            # transforms.Resize(size=(224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    folder_path = self.visual_path + datum
    file_num = len(os.listdir(folder_path))
    pick_num = 2
    seg = int(file_num/pick_num)
    image_arr = []
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

    # for i in range(pick_num):
    #   if self.mode == 'train':
    #     index = random.randint(i*seg + 1, i*seg + seg)
    #   else:
    #     index = i*seg + int(seg/2)
    #   path = folder_path + '/frame_000' + str(index).zfill(2) + '.jpg'
    #   # print(path)
    #   image_arr.append(transf(Image.open(path).convert('RGB')).unsqueeze(0))
    try:
      if self.mode == 'train':
          selected_files = random.sample(all_files, pick_num)
      else:
          mid = len(all_files) // 2
          selected_files = all_files[mid-1 : mid+1]
    except:
       print(folder_path)

    for file_name in selected_files:
        path = os.path.join(folder_path, file_name)
        img = Image.open(path).convert('RGB')
        image_arr.append(transf(img).unsqueeze(0))
    images = torch.cat(image_arr)

    return images, fbank, self.classes.index(self.data2class[datum])
