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
        self.img_list = self._load_image_list()

    def __getitem__(self, index):
        img_path = self.img_list[index]

        image = self.preprocessing(img_path)

        return image, os.path.basename(img_path).split(".")[0]

    def __len__(self):
        return len(self.img_list)

    def preprocessing(self,image):
        return image

    def _load_image_list(self):
        # read images
        impath = [f for f in glob(self.args.dataset+ "*.dcm")]

        imlist = []
        for img in impath:
            name = img.split("/")[-1]
            patient = df.loc[df.xid==name]
            imlist.append(img)
        
        return imlist


def load_dataloader(args):
    dataset = VFDataset(is_Train=False, args=args)
    batch_ = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)        

    return batch_


def VERTE_X_preprocessing(args,image):
        # Equalize the histogram of the image to enhance contrast
    image = cv2.equalizeHist(image)

    # Normalize the image using min-max scaling
    image = image_minmax(image)

    # Crop unnecessary sides from the image, need to define by users
    image = crop_side(image)

    # Resize
    # Get the current height and width of the image
    h, w = image.shape

    # Create a blank background image with a fixed size (1024x512)
    bg_img = np.zeros((1024, 512))

    # Resize the image proportionally based on its aspect ratio
    if w > h:  # Landscape orientation
        x = 512  # Set width to 512 pixels
        y = int(h / w * x)  # Calculate the proportional height
    else:  # Portrait orientation
        y = 1024  # Set height to 1024 pixels
        x = int(w / h * y)  # Calculate the proportional width

        # Ensure width doesn't exceed 512 pixels, adjusting height proportionally
        if x > 512:
            x = 512
            y = int(h / w * x)

    # Resize the image to the calculated dimensions
    img_resize = cv2.resize(image, (x, y))

    # Calculate padding offsets to center the resized image on the background
    xs = int((512 - x) / 2)
    ys = int((1024 - y) / 2)

    # Place the resized image in the center of the background
    bg_img[ys:ys + y, xs:xs + x] = img_resize
    image = bg_img  # Update the image with the padded version

    return image

