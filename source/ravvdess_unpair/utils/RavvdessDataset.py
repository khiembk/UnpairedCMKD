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
        csv_path: Đường dẫn chính xác đến file csv (train/val/test) muốn load.
        """
        # 1. Dùng pandas đọc 1 lần duy nhất làm source of truth
        self.df = pd.read_csv(csv_path, header=None) # Giả sử không có header, nếu có thì bỏ header=None
        self.audio_dir = audio_dir
        self.image_dir = image_dir
        self.mode = mode

        # 2. Xây dựng danh sách dữ liệu từ dataframe
        # Giả sử cấu trúc CSV: [audio_name, image_name, label]
        self.audio_files = self.df.iloc[:, 0].values
        self.image_files = self.df.iloc[:, 1].values
        
        # Mapping label sang index (nếu label là string)
        # Tự động lấy tất cả các class có trong file hiện tại
        unique_classes = sorted(self.df.iloc[:, 2].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}
        
        # Convert cột label text sang số
        self.labels = [self.class_to_idx[l] for l in self.df.iloc[:, 2].values]

        # 3. Logic cho Unpaired Training (Chỉ chạy khi train)
        self.indices_per_class = defaultdict(list)
        if self.mode == 'train':
            for idx, lbl in enumerate(self.labels):
                self.indices_per_class[lbl].append(idx)

        # 4. Định nghĩa Transform ngay tại __init__ (Tối ưu tốc độ)
        self.aud_transform = tv.transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[-10.59], std=[85.66])
        ])

        if self.mode == 'train':
            self.img_transform = transforms.Compose([
                # Không cần ToPILImage vì ta sẽ dùng PIL để load ảnh ngay từ đầu
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
        # Lấy label của mẫu hiện tại
        target_label = self.labels[index]

        # --- Xử lý Audio (Unpaired Logic) ---
        if self.mode == 'train':
            # Chọn ngẫu nhiên một audio khác CÙNG class
            candidate_indices = self.indices_per_class[target_label]
            audio_idx = random.choice(candidate_indices)
        else:
            # Test/Val thì lấy đúng cặp
            audio_idx = index

        # Load Audio
        aud_path = os.path.join(self.audio_dir, self.audio_files[audio_idx])
        audio_np = np.load(aud_path) # Shape: (H, W, C) hoặc (C, H, W)?
        
        # Fix channel swap cho audio (Chỉ giữ nếu file npy thực sự là ảnh 3 kênh BGR)
        if audio_np.ndim == 3 and audio_np.shape[2] == 3:
             audio_np = audio_np[:, :, ::-1] # BGR to RGB (gọn hơn cách dùng list index)
        
        # Transform Audio
        # Lưu ý: ToTensor của torch sẽ tự chuyển (H, W, C) -> (C, H, W) và scale về [0, 1] nếu input là uint8
        # Nếu audio_np là float thì nó chỉ permute dimension.
        audio = self.aud_transform(audio_np)


        # --- Xử lý Image ---
        img_idx = index
        img_path = os.path.join(self.image_dir, self.image_files[img_idx])
        
        # Dùng PIL đọc ảnh trực tiếp (nhanh hơn và tương thích tốt với torchvision)
        # PIL đọc mặc định là RGB, không cần swap BGR->RGB như cv2
        image = Image.open(img_path).convert('RGB')
        
        # Transform Image
        image = self.img_transform(image)

        return {
            'audio': audio, 
            'image': image, 
            'label': target_label
        }