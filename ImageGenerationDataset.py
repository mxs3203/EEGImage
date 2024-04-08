import glob
import random

import torch
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np

class ImageGenerationDataset(Dataset):
    def __init__(self, data, targets, mnist_folder,device, number_of_imgs_per_class=100):
        self.data = data
        self.targets = targets
        self.mnist_folder = mnist_folder
        self.device = device

        self.images = {}
        self.readMNISTImages(number_of_imgs_per_class)


    def readMNISTImages(self, number_of_imgs_per_class):
        for number in range(0, 10):
            specific_number_images = []
            cnt = 0
            for image_file in glob.glob("{}/{}/*.jpg".format(self.mnist_folder,number)):
                image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
                image = image/255.0
                specific_number_images.append(image)
                cnt = cnt + 1
                if cnt == number_of_imgs_per_class:
                    break
            self.images[number] = specific_number_images
    def __len__(self):
        return len(self.targets)


    def __getitem__(self, idx):
        number = self.targets[idx]
        eeg_data = self.data[idx]
        images_of_that_num = self.images[number]
        image = np.array(random.sample(images_of_that_num, 1))
        eeg_data_tensor = torch.tensor(eeg_data, dtype=torch.float32).to(self.device)
        number_tensor = torch.tensor(number, dtype=torch.long).to(self.device)
        image_tensor = torch.tensor(image, dtype=torch.float32).to(self.device)
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0)
        return eeg_data_tensor, number_tensor, image_tensor