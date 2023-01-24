import os
import cv2
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from glob import glob


def image_minmax(img):
    img_minmax = ((img - np.min(img)) / (np.max(img) - np.min(img))).copy()
    img_minmax = (img_minmax).astype(np.float32)
        
    return img_minmax

class VFDataset(Dataset):
    def __init__(self, is_Train, args):
        self.args = args
        self.is_Train = is_Train
        self.img_list,self.clincal_list = self._load_image_list()

    def __getitem__(self, index):
        img_path = self.img_list[index]

        image = self.preprocessing(img_path)
        clinical_idx = self.clincal_list[index]

        return image,torch.tensor(clinical_idx), os.path.basename(img_path).split(".")[0]

    def __len__(self):
        return len(self.img_list)

    def preprocessing(self,image):
        return image

    def _load_image_list(self):
        # read images
        impath = [f for f in glob(self.args.dataset+ "*.dcm")]
        # read data
        df = pd.read_csv(self.args.bcFile,"\t")

        clist = []
        imlist = []
        for img in impath:
            name = img.split("/")[-1]
            patient = df.loc[df.xid==name]
            imlist.append(img)
            clist.append([patient["label"]])
        
        return imlist,clist


def load_dataloader(args):
    dataset = VFDataset(is_Train=False, args=args)
    batch_ = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)        

    return batch_