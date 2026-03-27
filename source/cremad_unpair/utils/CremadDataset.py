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

class CremadDataset(Dataset):

    def __init__(self, args, mode='train'):
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode

        self.data_root = '/cremad_unpair/CREMA-D/'
        class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}

        self.visual_feature_path = '/cremad_unpair/CREMA-D/'
        self.audio_feature_path = '/cremad_unpair/CREMA-D/AudioWAV/'

        self.train_csv = os.path.join(self.data_root, 'train.csv')
        self.test_csv = os.path.join(self.data_root, 'test.csv')

        if mode == 'train':
            csv_file = self.train_csv
        else:
            csv_file = self.test_csv

        with open(csv_file, encoding='UTF-8-sig') as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                audio_path = os.path.join(self.audio_feature_path, item[0] + '.wav')
                visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(1), item[0])

                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(class_dict[item[1]])
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
        

        # audio
        samples, rate = librosa.load(self.audio[audio_idx], sr=22050)
        resamples = np.tile(samples, 3)[:22050*3]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.

        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        # mean = np.mean(spectrogram)
        # std = np.std(spectrogram)
        # spectrogram = np.divide(spectrogram - mean, std + 1e-9)

        if self.mode == 'train':
            transform = transforms.Compose([
                # transforms.RandomResizedCrop(224),
                # transforms.Resize(size=(224, 224)),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                # transforms.Resize(size=(224, 224)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.image[image_idx])
        # select_index = np.random.choice(len(image_samples), size=1, replace=False)
        # select_index.sort()
        images = torch.zeros((1, 3, 224, 224))
        for i in range(1):
            img = Image.open(os.path.join(self.image[image_idx], image_samples[i])).convert('RGB')
            img = transform(img)
            images[i] = img

        images = torch.permute(images, (1,0,2,3))


        # return spectrogram, images, label

        sample = {'audio': spectrogram, 'image': images, 'label': target_label}
        return sample