from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import time
import os
import copy

from utils.general_debug import scale_coords
from PIL import Image
cudnn.benchmark = True
import sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Data augmentation and normalization for training
# Just normalization for validation
mean = [0.4069, 0.4107, 0.4176]
std = [0.1626, 0.1614, 0.1572]
manual_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # 1. Reshape all images to 224x224 (though some models may require different sizes)
    transforms.ToTensor(), # 2. Turn image values to between 0 & 1 
    transforms.Normalize(mean, # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                         std=std) # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(activity_model_path = 'best_weights_activity_recog.pt'):
    model_ft = models.efficientnet_b7()
    # Get the length of class_names (one output unit for each class)
    output_shape = 2
    # Recreate the classifier layer and seed it to the target device
    model_ft.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), 
        torch.nn.Linear(in_features=2560, 
                        out_features=output_shape, # same number of output units as our number of classes
                        bias=True)).to(device)

    model_ft = model_ft.to(device)
    model_ft.load_state_dict(torch.load(activity_model_path, map_location=device))
    return model_ft


def apply_classifier(x, model_ft, img, im0):
    print('IN apply_classifier activity recog!!!')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    labels = ['lying', 'standing']
    print(f'\nim0 shape {im0[0].shape}')

    for i, d in enumerate(x):  # per image
        if(1):
            if d is not None and len(d):
                d = d.clone()

                # Rescale boxes from img_size to im0 size
                scale_coords(img.shape[2:], d[:, :4], im0[i].shape)
                
                print(f'd after scaling: {d}')

                # Classes
                pred_cls1 = d[:, 5].long()
                ims = []
                probs = []
                preds = []
                #ims = x = torch.empty(size=(len(d), 768))
 
                for j, a in enumerate(d):  # per item
                    if(1):
                        cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]

                        cutout = cutout[:, :, ::-1] # BGR to RGB
                        cutout = Image.fromarray(cutout)
                        
                        cutout = manual_transforms(cutout)
                        
                        cutout = torch.unsqueeze(cutout, dim=0)
                        cutout = cutout.to(d.device)
                                               
                        model_ft.eval()
                        with torch.no_grad():
                            out = model_ft(cutout)
                        #out = F.softmax(out)
                        prob, pred_label_cls = torch.max(out, 1)
                        
                        probs.append(prob.item())
                        preds.append(labels[pred_label_cls.item()])
 
                '''
                for obj_i in range(len(x[i])):
                    x[i][obj_i].append(pred_label_cls[obj_i])
                ''' 
    print('\nPredicted activities: ', preds)
    return preds

def activity_recognition_run(x, img, im0):
    activity_model = load_model()
    return apply_classifier(x, activity_model, img, im0)


