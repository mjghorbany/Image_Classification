# import torchvision.models as models
import torch.nn as nn
import numpy as np
import torch.utils.data
import os
# from skimage import io, transform
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
from torch import optim
import pickle
import time
from skimage import io
import torch
from torch.autograd import Variable
import random
import os





class dataset(torch.utils.data.Dataset):
    def __init__(self,typ = 'train'):
        class1 = ['Boreal','Deciduous','DV','Rainforest','TropGrass','Tundra','USGrass']
        class2 = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        dict1 = dict()
        dict2 = dict()
        for i,cl in enumerate(class1):
            dict1[cl]=i
        for i,cl in enumerate(class2):
            dict2[cl]=i
        self.typ = typ
        
        if self.typ == 'train':
            
            dir = './data/Train/'
            ims_path = os.listdir(dir)
            all_images = dict()
            for path in ims_path:
                if path[0]=='.':
                    continue
                p = path.split('_')
                c1 = dict1[p[0]]
                c2 = dict2[p[1][:3]]
                all_images[dir+path]=[c1,c2]
            
        else:
            dir = './data/Test/'
            ims_path = os.listdir(dir)
            all_images = dict()
            for path in ims_path:
                if path[0]=='.':
                    continue
                p = path.split('_')
                c1 = dict1[p[0]]
                c2 = dict2[p[1][:3]]
                all_images[dir+path]=[c1,c2]
        
        self.all_images = all_images
        

    def __len__(self):

        return len(self.all_images)

    def __getitem__(self, idx):
        
        img_name = list(self.all_images)[idx]
       
        image = Image.open(img_name).convert('RGB')
        s1,s2 = image.size        
        if s1<s2:
            tup = (240,int((240*s2)/s1))
        else:
            tup = (int((240*s1)/s2),240)
        lis=[transforms.Resize(tup,interpolation=2),transforms.CenterCrop((224,224)),
                     transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]
        transform_resize = transforms.Compose(lis)

        image = transform_resize(image)
        
        return (image,torch.LongTensor([self.all_images[img_name][0]]),torch.LongTensor([self.all_images[img_name][1]]))
        