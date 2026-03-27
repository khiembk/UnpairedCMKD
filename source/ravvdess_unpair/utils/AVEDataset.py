import copy
import csv
import os
import pickle
import librosa
import numpy as np
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from collections import defaultdict 
import random 
import pdb

class AVEDataset(Dataset):

    def __init__(self, args, mode='train'):
        self.fps = 1
        self.num_frame = args.num_frame
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode
        classes = []

        self.data_root = '/data/AVE_Dataset'
        self.visual_feature_path = '/data/AVE_Dataset'
        self.audio_feature_path = '/data/AVE_Dataset/Audio-1004-SE'

        self.train_txt = os.path.join(self.data_root + '/trainSet.txt')
        self.test_txt = os.path.join(self.data_root + '/testSet.txt')
        self.val_txt = os.path.join(self.data_root + '/valSet.txt')

        if mode == 'train':
            txt_file = self.train_txt
        elif mode == 'test':
            txt_file = self.test_txt
        else:
            txt_file = self.val_txt

        with open(self.test_txt, 'r') as f1:
            files = f1.readlines()
            for item in files:
                item = item.split('&')
                if item[0] not in classes:
                    classes.append(item[0])
        class_dict = {}
        for i, c in enumerate(classes):
            class_dict[c] = i

        with open(txt_file, 'r') as f2:
            files = f2.readlines()
            for item in files:
                item = item.split('&')
                audio_path = os.path.join(self.audio_feature_path, item[1] + '.pkl')
                visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS-SE'.format(self.fps), item[1])

                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    if audio_path not in self.audio:
                        self.image.append(visual_path)
                        self.audio.append(audio_path)
                        self.label.append(class_dict[item[0]])
                else:
                    continue
        
        self.indices_per_class = defaultdict(list)
        for idx, lbl in enumerate(self.label):
            self.indices_per_class[lbl].append(idx)


    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image_idx = idx
        target_label = self.label[image_idx] 
        
        if self.mode == 'train':
            candidate_indices = self.indices_per_class[target_label]
            
            audio_idx = random.choice(candidate_indices)
        else:
            audio_idx = idx

        spectrogram = pickle.load(open(self.audio[audio_idx], 'rb'))
        spectrogram = np.resize(spectrogram, (257, 1024))

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        image_samples = os.listdir(self.image[image_idx]) 
        
        images = torch.zeros((self.num_frame, 3, 224, 224))
        for i in range(self.num_frame):
            file_name = image_samples[i % len(image_samples)] 
            img = Image.open(os.path.join(self.image[image_idx], file_name)).convert('RGB')
            img = transform(img)
            images[i] = img

        images = torch.permute(images, (1,0,2,3))

        sample = {'audio': spectrogram, 'image': images, 'label': target_label}
        return sample
