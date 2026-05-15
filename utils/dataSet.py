import torchvision.transforms as transforms
import cv2, random
from torch.utils.data import Dataset
from PIL import Image
from .get_imgR import rotate_crop
    

import random
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class CoCo_Dataset(Dataset):
    def __init__(self, imgPath, fileName, tShape):
        
        # ColorJitter randomly changes brightness, contrast, saturation, and hue.
        # This prevents the model from relying on color features, forcing it to learn shapes.
        color_augmentation = transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.8,
            hue=0.5
        )
        
        # Optional: Randomly convert images to grayscale to further penalize color reliance
        grayscale_augmentation = transforms.RandomGrayscale(p=0.5)

        self.transform_img = transforms.Compose([
            color_augmentation,
            grayscale_augmentation,
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  
                std=[0.229, 0.224, 0.225]
            )            
        ])
        
        self.transform_roi = transforms.Compose([
            color_augmentation,
            grayscale_augmentation,
            transforms.Resize(tShape),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  
                std=[0.229, 0.224, 0.225]
            )            
        ])
        
        self.imgPath   = imgPath
        self.samples   = []
        self.tShape    = tShape
        
        with open(fileName, 'r') as lines:
            for line in lines:
                name, x, y, w, h = line.strip().split(',')
                self.samples.append([
                    name,
                    (int(x), int(y), int(w), int(h))
                ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        while True:
            name, bbox  = self.samples[idx]
            image       = cv2.imread(self.imgPath + name)
            image       = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            x, y, w, h =  [round(v) for v in bbox]
            template   =  image[y:y+h, x:x+w]
             
            # Assume rotate_crop is defined elsewhere in your module
            imgR, corners, angle = rotate_crop(image, bbox)
            if imgR is None:
                idx = random.randint(0, len(self.samples)-1)
                continue

            scale_y    =  h / self.tShape[0]
            scale_x    =  w / self.tShape[1]
            center     = corners.mean(axis=0)

            # Transforms expect a PIL Image, which is already being handled here
            imgR       = self.transform_img(Image.fromarray(imgR))
            template   = self.transform_roi(Image.fromarray(template))
            
            return name, imgR, template, center[1], center[0], scale_y, scale_x, angle
