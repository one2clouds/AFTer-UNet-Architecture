from Data.prepare_dataloader import get_from_loader_segthor
import torch
import torch.optim as optim
from axial_fusion_transformer import axial_fusion_transformer
import os
from tqdm import tqdm 
from operator import add
import yaml 
import json
import matplotlib.pyplot as plt
from check_size_and_voxels.check_dataset import check_dataset
import wandb
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, LossMetric, MeanIoU

import warnings
import monai 
import random
import argparse

def parse_training_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_config")
    parser.add_argument("val_config")
    parser.add_argument("--dataset_to_use")
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_training_arguments()
    train_config = args.train_config
    val_config = args.val_config
    dataset__to_use = args.dataset_to_use

    with open(val_config, 'r') as config_file:
        config_params = yaml.safe_load(config_file)
        model_config = json.dumps(config_params)

    with open(train_config,'r') as config_file:
        config_params_train = yaml.safe_load(config_file)
        model_config_train = json.dumps(config_params)

    warnings.filterwarnings("ignore")

    if dataset__to_use=='segthor_data':
        image_location = '/mnt/Enterprise2/shirshak/SegTHOR/train/P*/P*.nii.gz'
        mask_location = '/mnt/Enterprise2/shirshak/SegTHOR/train/P*/G*.nii.gz'
        batch_size=1
        num_classes = 5
        num_channels_before_training = 1

        train_dataloader, test_dataloader = get_from_loader_segthor(image_location, mask_location, num_classes, batch_size)                                   
    else:
        print("Dataset to use is brats data....but it is currently not supported")

    Na = config_params["testing_params"]["Na"]
    Nf = config_params["testing_params"]["Nf"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = axial_fusion_transformer(Na, Nf, num_classes, num_channels_before_training).to(device)
    
    optimizer = optim.Adam(params=model.parameters())

    checkpoint_path = args.checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
    else:
        print("Model not found!!")

    model.eval()
    
    i=0
    for image_and_mask in list(test_dataloader)[:10]:
        fig, ax = plt.subplots(1,3)
        image = image_and_mask['_3d_image']['data'].float().to(device)
        mask = image_and_mask['_3d_mask']['data'].float().to(device)
        output = model(image)

        image = image.squeeze()
        #         print(mask.shape)
        mask = mask.squeeze(dim=0).argmax(0)
        output = output.squeeze(dim=0).argmax(0)
        #         print(image.shape) # torch.Size([1, 1, 128, 128, 128])
        #         print(image.max()) # tensor(1., device='cuda:0')
        #         print(image.min()) # tensor(0., device='cuda:0')
        #         print(mask.max()) # tensor(1., device='cuda:0')
        #         print(mask.min()) # tensor(0., device='cuda:0')
        #         print(output.max()) #tensor(0.9952, device='cuda:0')
        #         print(output.min()) # tensor(6.8929e-08, device='cuda:0')
        #         print(output.shape) # torch.Size([4, 128, 128, 128])

        # n_slice = 55
        for j in range(10):
            n_slice = int(random.random() * 128)
            ax[0].imshow(image[:,:,n_slice].cpu())
            ax[0].set_title('Image of SegTHOR CT Scan')
            ax[1].imshow(mask[:,:,n_slice].cpu())
            ax[1].set_title('Original Mask')
            ax[2].imshow(output[:,:,n_slice].cpu())
            ax[2].set_title('Predicted Mask')

            fig.savefig(f'images/full_figure{i} {j} {n_slice}.png')
        i += 1