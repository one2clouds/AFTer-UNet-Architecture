from torch.utils.data import Dataset, DataLoader, random_split
from monai.transforms import (ResizeWithPadOrCropd,Activations,
                              Activationsd,AsDiscrete,AsDiscreted,
                              Compose,Invertd,LoadImaged,
                              NormalizeIntensityd,Orientationd,
                              RandFlipd,RandScaleIntensityd,
                              RandSpatialCropd,Spacingd,EnsureTyped,
                              EnsureChannelFirstd,RandShiftIntensityd,
                              Rand3DElasticd
                              )
import numpy as np 
import torch
import glob
import os
import monai
from monai.utils import set_determinism 
# from torchvision.transforms import ElasticTransform

train_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"],pixdim=(1.0, 1.0, 2.5),mode=("bilinear", "nearest"),),
        ResizeWithPadOrCropd(keys=["image","label"],spatial_size=(64,64,64)),

        # Adding ELASTIC 3D AS MENTIONED ON THE PAPER 
        
        Rand3DElasticd(keys=['image','label'],sigma_range=(5,7),magnitude_range=(50,15), prob=1,padding_mode='zeros',mode=['bilinear','nearest']),
        # RandSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 128], random_size=False),
        # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        # RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
])
val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"],pixdim=(1.0, 1.0, 1.0),mode=("bilinear", "nearest"),),

            ResizeWithPadOrCropd(keys=["image","label"],spatial_size=(128,128,128)),
            
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ])

class SegTHORDataset(Dataset):
    def __init__(self, file_names, transform,num_classes, ):
        self.file_names = file_names
        self.transform = transform
        self.num_classes = num_classes
        self.as_discrete = AsDiscrete(to_onehot=self.num_classes)

    def __getitem__(self, index):
        file_names = self.file_names[index]
        dataset = self.transform(file_names) 
        dataset["label"] = self.as_discrete(dataset["label"])
        return dataset
    
    def __len__(self):
        return len(self.file_names)

def get_from_loader_segthor(image_location, mask_location, batch_size, num_classes):
    set_determinism(seed=12345)
    images = sorted(glob.glob(image_location))
    labels = sorted(glob.glob(mask_location))

    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(images, labels)]
    train_files, val_files = data_dicts[:-9], data_dicts[-9:]

    train_ds = SegTHORDataset(train_files, train_transform,num_classes)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    # TODO CHANGE TRAIN_TRANSFORM TO VAL_TRANSFORM 
    val_ds = SegTHORDataset(val_files, train_transform, num_classes)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, train_ds, val_ds

def get_from_loader_brats(t2_location,t1ce_location,flair_location,mask_location, batch_size):
    my_dataset = SegTHORDataset(t2_location,t1ce_location,flair_location,mask_location)

    train_ratio = 0.9
    test_ratio = 1.0 - train_ratio
    # Calculate the number of samples for training and testing
    num_samples = len(my_dataset)
    num_train_samples = int(train_ratio * num_samples)
    num_test_samples = num_samples - num_train_samples

    train_dataset, test_dataset = random_split(my_dataset, [num_train_samples , num_test_samples])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader


if __name__ =="__main__":
    image_location = '/mnt/Enterprise2/shirshak/SegTHOR/train/P*/P*.nii.gz'
    mask_location = '/mnt/Enterprise2/shirshak/SegTHOR/train/P*/G*.nii.gz'

    train_dataloader, test_dataloader, train_ds, val_ds = get_from_loader_segthor(image_location, mask_location, batch_size=1)

    print(train_ds[0]["image"].shape)
    print(train_ds[0]["label"].shape)
