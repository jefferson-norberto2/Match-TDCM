import warnings
warnings.filterwarnings("ignore")

import cv2, torch, math
from model import Model
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from utils import get_center
from utils.refine import *


device = torch.device('cpu')

model = Model('dict/model.pth').to(device).eval()

transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225]),
        ])
        
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


query_image_path = 'screen.png'
template_image_path = 'plane.png'

with torch.no_grad():
    image = cv2.imread(query_image_path)
    template = cv2.imread(template_image_path)
    template = cv2.resize(template, (36, 36))
    
    image = transform_img(Image.fromarray(image)).to(device).unsqueeze(0)
    template = transform_img(Image.fromarray(template)).to(device).unsqueeze(0)
        
    pred_score, pred_sign, pred_cos, pred_scale_x, pred_scale_y = model(image, template)
    
    # Get the predicted center, angle, and scale from the model's output
    pred = pred_score[0, 0, :, :]
    max_idx = torch.nonzero(pred == pred.max())[0]
    pre_Y, pre_X = max_idx.cpu().numpy()
    pred_y, pred_x = get_center((pre_Y, pre_X), torch.nonzero(pred > 0.5).cpu().numpy())
    
    # Take the sign of the angle into account
    sign = 1 if pred_sign[0, 0, pre_Y, pre_X] > 0.5 else -1
    pred_r = torch.arccos(pred_cos[0, 0, pre_Y, pre_X].clamp(-1, 1)) * 180 / math.pi
    pred_r = sign * pred_r.item()
    
    # Get the predicted scale from the model's output
    pred_sx = pred_scale_x[0, 0, pre_Y, pre_X].item()
    pred_sy = pred_scale_y[0, 0, pre_Y, pre_X].item()

    print(f"Predicted Center: ({pred_x:.2f}, {pred_y:.2f})")
    print(f"Predicted Scale: (sx: {pred_sx:.2f}, sy: {pred_sy:.2f})")

    search_img = (denormalize(image.squeeze(0))).permute(1, 2, 0).cpu().numpy()
    template_img = (denormalize(template.squeeze(0))).permute(1, 2, 0).cpu().numpy()
    
    # This is where the refinement of the predicted parameters happens using the bisection method and other refinement techniques
    pred_sx, _ = refine_scale_x(
        search_img, template_img, pred_x, pred_y, pred_sx, pred_sy, pred_r
    )
    pred_sy, _ = refine_scale_y(
        search_img, template_img, pred_x, pred_y, pred_sx, pred_sy, pred_r
    )

    print(f"Refined Angle: {pred_r:.2f} degrees")
    print(f"Refined Scale: (sx: {pred_sx:.2f}, sy: {pred_sy:.2f})")

    # Draw the box on the image

    # 1. Calculate the width and height of the predicted box based on the refined scale and the original template size
    h, w, _ = template_img.shape

    # 2. Calculate the top-left corner of the predicted box based on the predicted center and the refined scale
    pred_w = w * pred_sx
    pred_h = h * pred_sy

    pred_x1 = int(pred_x - pred_w / 2)
    pred_y1 = int(pred_y - pred_h / 2)

    # 3. Calculate the bottom-right corner of the predicted box
    pred_x2 = int(pred_x + pred_w / 2)
    pred_y2 = int(pred_y + pred_h / 2)
    
    result_img = cv2.rectangle(search_img.copy(), (pred_x1, pred_y1), (pred_x2, pred_y2), color=(0, 255, 0), thickness=2)
    
    plt.imshow(result_img)
    plt.title(f"Predicted Box with Center: ({pred_x:.2f}, {pred_y:.2f}), Scale: (sx: {pred_sx:.2f}, sy: {pred_sy:.2f}), Angle: {pred_r:.2f} degrees")
    plt.axis('off')
    plt.show()

    