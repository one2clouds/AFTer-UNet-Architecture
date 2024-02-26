from torch.utils.data import Dataset, DataLoader, random_split
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
import torch
import glob
import os
import torchio as tio 



class MyDataset(Dataset):
    def __init__(self, image_location, mask_location, num_classes):
        self.image_list = sorted(glob.glob(image_location))
        self.mask_list = sorted(glob.glob(mask_location))
        self.scaler = MinMaxScaler()
        self.transform = tio.CropOrPad((128,128,128))
        self.num_classes = num_classes
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        image = nib.load(self.image_list[idx]).get_fdata()
        image = self.scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
        image = torch.tensor(image, dtype=torch.float64)
    
        mask = nib.load(self.mask_list[idx]).get_fdata()
        mask = self.scaler.fit_transform(mask.reshape(-1, mask.shape[-1])).reshape(mask.shape)
        mask = torch.tensor(mask, dtype=torch.int64)

        mask = torch.nn.functional.one_hot(mask, self.num_classes)
        # print(mask.shape) # torch.Size([512, 512, 179, 4])
        mask = mask.permute(3,0,1,2)

        # print(mask.shape)torch.Size([512, 512, 179, 4])

        # print(image.unsqueeze(dim=0).shape)
        # print(mask.shape)

        image_and_mask = tio.Subject(
                _3d_image = tio.ScalarImage(tensor=image.unsqueeze(dim=0)), # unsqueezing for 1 channel
                _3d_mask = tio.LabelMap(tensor = mask))
        
        image_and_mask = self.transform(image_and_mask)
        return image_and_mask
        

def get_from_loader(image_location, mask_location, num_classes):
    my_dataset = MyDataset(image_location, mask_location, num_classes)

    train_ratio = 0.8
    test_ratio = 1.0 - train_ratio

    num_samples = len(my_dataset)
    num_train_samples = int(train_ratio * num_samples)
    num_test_samples = num_samples - num_train_samples

    train_dataset, test_dataset = random_split(my_dataset, [num_train_samples , num_test_samples])
    
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_dataloader, test_dataloader



