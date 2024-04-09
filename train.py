from Data.prepare_dataloader import get_from_loader
import torch
import torch.optim as optim
from axial_fusion_transformer import axial_fusion_transformer
import os
from tqdm import tqdm 
from calculate_metrics import calculate_metrics
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


with open('./config/train_config.yaml', 'r') as config_file:
    config_params = yaml.safe_load(config_file)
    model_config = json.dumps(config_params)

def training_phase(train_dataloader, test_dataloader, num_classes, wandb):

    num_epochs = wandb.config['epochs']

    Na = config_params["training_params"]["Na"]
    Nf = config_params["training_params"]["Nf"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config_params["training_params"]["model_name"] == "axial_fusion_transformer":
        model = axial_fusion_transformer(Na,Nf, num_classes).to(device)
    
    loss_function = DiceLoss(include_background=True, reduction="mean")
    # dice metric vanne use garda nan values aayo
    # dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric = DiceMetric(include_background=True, reduction="mean_channel")
    iou_metric = MeanIoU(include_background=True, reduction="mean_channel")
    optimizer = optim.Adam(model.parameters(), lr = wandb.config['lr'])


    checkpoint_path = 'model.pth'
    start_epoch = 1
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")


    for epoch in range(start_epoch,num_epochs+1):
        epoch_train_loss = 0.0
        epoch_test_loss = 0.0  
        images = []
        metrics_score = [0.0, 0.0]
        model.train()
        torch.cuda.empty_cache()
        
        for img_and_mask in tqdm(train_dataloader, desc=f'Training Epoch {epoch}/{num_epochs}', unit='epoch'):
            torch.cuda.empty_cache()
            
            image = img_and_mask['_3d_image']['data'].float().to(device)
            mask = img_and_mask['_3d_mask']['data'].float().to(device)
            optimizer.zero_grad()
            output = model(image)

            # print(output.shape) # torch.Size([5, 128, 128, 128])
            # print(mask.shape) # torch.Size([1, 5, 128, 128, 128])
            train_loss = loss_function(output, mask.squeeze(dim=0))
            train_loss.backward()

            score = calculate_metrics(output, mask.squeeze(dim=0))
            metrics_score = list(map(add,metrics_score, score))

            dice_score_train = dice_metric(output.unsqueeze(dim=0), mask)

            jaccard_score_train_monai = iou_metric(output.unsqueeze(dim=0), mask)

            # print(jaccard_score_train_monai.shape)  #torch.Size([1, 5])
            # print(dice_score_train.shape)  #torch.Size([1, 5])

            optimizer.step()
            epoch_train_loss += train_loss.item()

            # print(dice_score_train) # metatensor([[0.9095]])

            epoch_train_loss = epoch_train_loss/len(train_dataloader)
            epoch_train_jaccard = metrics_score[0]/len(train_dataloader)
            epoch_train_acc = metrics_score[1]/len(train_dataloader)
        
        model.eval()
        metrics_score = [0.0, 0.0]
        with torch.no_grad():
            for img_and_mask in tqdm(test_dataloader, desc=f'Test Epoch {epoch}/{num_epochs}', unit='epoch'):
                torch.cuda.empty_cache()
                image = img_and_mask['_3d_image']['data'].float().to(device)
                mask = img_and_mask['_3d_mask']['data'].float().to(device)
                output = model(image)
                
                test_loss = loss_function(output, mask.squeeze(dim=0))
                
                score = calculate_metrics(output, mask)
                metrics_score = list((map(add, metrics_score, score)))

                # print(dice_score_test) # metatensor([[0.9115]])
                # print(output.shape) # torch.Size([5, 128, 128, 128])
                # print(mask.shape) # torch.Size([1, 5, 128, 128, 128])

                dice_score_test = dice_metric(output.unsqueeze(dim=0), mask) # only passing batchC, H,W,D to the metric 
                jaccard_score_test_monai = iou_metric(output.unsqueeze(dim=0), mask) # only passing batch, C, H,W,D to the metric 

                # print(dice_score_test.shape) #torch.Size([1, 5])
                # print(jaccard_score_test_monai.shape) #torch.Size([1, 5])

                epoch_test_loss += test_loss.item()

                epoch_test_loss = epoch_train_loss/len(test_dataloader)
                epoch_test_jaccard = metrics_score[0]/len(test_dataloader)
                epoch_test_acc = metrics_score[1]/len(test_dataloader)

            # TAKING FIRST 4 IMAGES OF TEST AND PASSING IT TO WANDB FOR VISUALIZATION AFTER EVERY EPOCH

            for img_and_mask in list(test_dataloader)[:4]: 
                image = img_and_mask['_3d_image']['data'].float().to(device)
                mask = img_and_mask['_3d_mask']['data'].float().to(device)
                output = model(image) 
                image = image.squeeze()
                mask = mask.squeeze(dim=0).argmax(0)
                output = output.argmax(0)

                # print(image.shape) # torch.Size([128, 128, 128])
                # print(mask.shape) # torch.Size([128, 128, 128])
                # print(output.shape) # torch.Size([128, 128, 128])
            
                fig, ax = plt.subplots(1,3)
                n_slice = 55 #int(random.random() * 128)
                ax[0].imshow(image[:,:,n_slice].cpu())
                ax[0].set_title('Image of SegTHOR CT Scan')
                ax[1].imshow(mask[:,:,n_slice].cpu())
                ax[1].set_title('Original Mask')
                ax[2].imshow(output[:,:,n_slice].cpu() > 0.5)
                ax[2].set_title('Predicted Mask')

                # fig.savefig(f'images/figure {n_slice}.png')

                images.append(fig)

                
        torch.save({'epoch':start_epoch, 
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    }, checkpoint_path)

        wandb.log({"train_loss":train_loss.item(),
                   "test_loss":test_loss.item(),
                   "train_jaccard":epoch_train_jaccard,
                   "test_jaccard":epoch_test_jaccard,
                   "train_dice":dice_score_train.mean().item(),
                   "test_dice":dice_score_test.mean().item(), 
                   "train_jaccard_monai":jaccard_score_train_monai.mean().item(),
                   "test_jaccard_monai":jaccard_score_test_monai.mean().item(),
                   "image": [wandb.Image(img) for img in images]
                   })    
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
            f'Train Loss: {train_loss.item():.4f}, '
            f'Train Jaccard: {epoch_train_jaccard.item():.4f}, '
            f'Train Accuracy: {epoch_train_acc:.4f}, '
            f'Test Loss: {test_loss.item():.4f}, '
            f'Test Jaccard: {epoch_test_jaccard.item():.4f}, '
            f'Train Dice Score: {dice_score_train.mean().item():.4f}, '
            f'Test Dice Score: {dice_score_test.mean().item():.4f}, '
            f'Train Jaccard Monai: {jaccard_score_train_monai.mean().item():.4f}, '
            f'Test Jaccard Monai: {jaccard_score_test_monai.mean().item():.4f}, '
            f'Test Accuracy: {epoch_test_acc}, ')
    
    return model,num_epochs,optimizer, train_loss






if __name__ =='__main__':

    warnings.filterwarnings("ignore")

    image_location = '/mnt/Enterprise2/shirshak/SegTHOR/train/P*/P*.nii.gz'
    mask_location = '/mnt/Enterprise2/shirshak/SegTHOR/train/P*/G*.nii.gz'

    wandb.login(key=config_params["training_params"]["wandb_key"])

    wandb.init(
        project="SegTHOR Axial Implementation",
        config={
            "epochs": config_params["training_params"]["num_epochs"],
            "batch_size":config_params["training_params"]["batch_size"],
            "lr": config_params["training_params"]["lr"]
        }
    )
# check for voxel shape and load in csv format
    check_dataset(image_location, mask_location)

    # print(config_params["training_params"]["num_classes"]) #5

    
    train_dataloader, test_dataloader = get_from_loader(image_location, mask_location, config_params["training_params"]["num_classes"], wandb.config['batch_size'])                                   
    
    model, num_epochs,optimizer, loss= training_phase(train_dataloader,test_dataloader, config_params["training_params"]["num_classes"], wandb)

