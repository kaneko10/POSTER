import torch
import torch.utils.data as data
# from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import re

class CustomDataset(data.Dataset):
    # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral（RAF-DB）
    # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral（AffectNet）
    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # self.classes = sorted(os.listdir(root_dir))
        # self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        # self.classes = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']  # RAF-DBの場合
        self.classes = classes
        # self.classes = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']  # AffectNetの場合、違うかも
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.img_paths = self._get_img_paths()
        
    def _get_img_paths(self):
        img_paths = []
        img_names = []
        print(self.classes)
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            print("path: ", cls_dir)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                if not img_name.startswith('.'):  # 隠しファイルを無視
                    img_names.append(img_name)
            # 数字の部分を抜き出してソート
            sorted_image_names = sorted(img_names, key=lambda x: int(re.search(r'\d+', x).group()))
            for img_name in sorted_image_names:
                img_path = os.path.join(cls_dir, img_name)
                # print("image: ", img_name)
                if os.path.isfile(img_path) and not img_name.startswith('.'):  # ファイルのみを対象とし、隠しファイルを無視
                    img_paths.append((img_path, self.class_to_idx[cls]))
        return img_paths
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image.copy())
        return image, label, img_path  # ファイル名も返す
