import torch
import torchvision as tv
from torchvision import transforms
from torch.utils.data import Dataset
import os
import cv2
import pandas as pd
import numpy as np
from collections import defaultdict
import random
from PIL import Image

class RavvdessDataset(Dataset):
    def __init__(self, csv_path, audio_dir, image_dir, mode='train'):
        """
        csv_path: Exact path to the csv file (train/val/test) to load.
        """
        self.df = pd.read_csv(csv_path, header=None) 
        self.audio_dir = audio_dir
        self.image_dir = image_dir
        self.mode = mode


        self.audio_files = self.df.iloc[:, 0].values
        self.image_files = self.df.iloc[:, 1].values
        

        unique_classes = sorted(self.df.iloc[:, 2].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}
        
        self.labels = [self.class_to_idx[l] for l in self.df.iloc[:, 2].values]

        self.indices_per_class = defaultdict(list)
        if self.mode == 'train':
            for idx, lbl in enumerate(self.labels):
                self.indices_per_class[lbl].append(idx)

        self.aud_transform = tv.transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[-10.59], std=[85.66])
        ])

        if self.mode == 'train':
            self.img_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        target_label = self.labels[index]

        if self.mode == 'train':
            candidate_indices = self.indices_per_class[target_label]
            audio_idx = random.choice(candidate_indices)
        else:
            audio_idx = index

        aud_path = os.path.join(self.audio_dir, self.audio_files[audio_idx])
        audio_np = np.load(aud_path) 
        
        if audio_np.ndim == 3 and audio_np.shape[2] == 3:
             audio_np = audio_np[:, :, ::-1] 
        

        audio = self.aud_transform(audio_np)


        img_idx = index
        img_path = os.path.join(self.image_dir, self.image_files[img_idx])
        
        image = Image.open(img_path).convert('RGB')
        
        image = self.img_transform(image)

        return {
            'audio': audio, 
            'image': image, 
            'label': target_label
        }