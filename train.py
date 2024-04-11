from Data.prepare_dataloader import get_from_loader
import torch
import torch.optim as optim
from axial_fusion_transformer import axial_fusion_transformer
import os
from tqdm import tqdm 
from operator import add
import yaml 
import json
from check_size_and_voxels.check_dataset import check_dataset
import wandb
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, MeanIoU, LossMetric

from monai.metrics.meandice import compute_dice
from monai.metrics.meaniou import compute_iou
import warnings
from utils import show_img_mask_output


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

    dice_metric = DiceMetric(include_background=True, reduction="mean", num_classes=num_classes)
    loss_metric = LossMetric(loss_fn=loss_function, reduction="mean")
    iou_metric = MeanIoU(include_background=True, reduction="mean")


    optimizer = optim.Adam(model.parameters(), lr = wandb.config['lr'])


    checkpoint_path = 'model.pth'
    start_epoch = 1
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")

    # lr scheduler helps in stopping overfitting
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)


    for epoch in range(start_epoch,num_epochs+1):
        images_wandb_train = []
        images_wandb_test = []
        model.train()
        for count, img_and_mask in enumerate(tqdm(train_dataloader, desc=f'Training Epoch {epoch}/{num_epochs}', unit='epoch')):
            image = img_and_mask['_3d_image']['data'].float().to(device)
            mask = img_and_mask['_3d_mask']['data'].float().to(device)
            optimizer.zero_grad()
            output = model(image)

            # print(output.shape) # torch.Size([5, 128, 128, 128])
            # print(mask.shape) # torch.Size([1, 5, 128, 128, 128])

            train_loss = loss_function(output, mask.squeeze(dim=0)) # DICE LOSS TAKES I/P in the form of Batch Channel HWD 
            train_loss.backward()
            
            dice_metric(output, mask.squeeze(dim=0)) # DICE METRIC TAKES I/P in the form of Batch Channel HWD 
            loss_metric(output, mask.squeeze(dim=0))
            iou_metric(output, mask.squeeze(dim=0)) # MEAN IOU TAKES I/P in the form of Batch Channel HWD

            computed_dice_train = compute_dice(output.unsqueeze(dim=0), mask).mean()
            computed_iou_train= compute_iou(output.unsqueeze(dim=0), mask).mean()

            # print(computed_dice_train)
            # print(computed_iou_train)
            
            optimizer.step()

            # FOR LOGGIN THE IMAGES IN THE WANDB AND VISUALIZING RESULTS
            if count%10 ==0:
                img_wandb_train = show_img_mask_output(image, mask, output)
                images_wandb_train.append(img_wandb_train)

        lr_scheduler.step()

        dice_score_train = dice_metric.aggregate().item()
        dice_score_train_2 = loss_metric.aggregate().item()
        jaccard_score_train = iou_metric.aggregate().item()


        # print(f'Dice Score Train : {type(dice_score_train)}') # Dice Score Train : <class 'float'>
        # print(f'Jaccard Score Train : {type(jaccard_score_train)}') # Jaccard Score Train : <class 'float'>
        # print(f'Train Loss : {type(train_loss)}') # Train Loss : <class 'monai.data.meta_tensor.MetaTensor'>

        dice_metric.reset()
        loss_metric.reset()
        iou_metric.reset()
        
        model.eval()
        with torch.no_grad():
            for count, img_and_mask in enumerate(tqdm(test_dataloader, desc=f'Test Epoch {epoch}/{num_epochs}', unit='epoch')):
                torch.cuda.empty_cache()
                image = img_and_mask['_3d_image']['data'].float().to(device)
                mask = img_and_mask['_3d_mask']['data'].float().to(device)
                output = model(image)
                
                test_loss = loss_function(output.unsqueeze(dim=0), mask) # DICE LOSS TAKES I/P in the form of Batch Channel HWD 

                # print(dice_score_test) # metatensor([[0.9115]])
                # print(output.shape) # torch.Size([5, 128, 128, 128])
                # print(mask.shape) # torch.Size([1, 5, 128, 128, 128])

                dice_metric(output.unsqueeze(dim=0), mask) # DICE METRIC TAKES I/P in the form of Batch Channel HWD
                loss_metric(output.unsqueeze(dim=0), mask)
                iou_metric(output.unsqueeze(dim=0), mask) # MEAN IOU TAKES I/P in the form of Batch Channel HWD

                computed_dice_test= compute_dice(output.unsqueeze(dim=0), mask).mean()
                computed_iou_test= compute_iou(output.unsqueeze(dim=0), mask).mean()

                # print(computed_dice_test)
                # print(computed_iou_test)


                # FOR LOGGIN THE IMAGES IN THE WANDB AND VISUALIZING RESULTS
                img_wandb_test = show_img_mask_output(image, mask, output)
                images_wandb_test.append(img_wandb_test)
                
            dice_score_test = dice_metric.aggregate().item()
            dice_score_test_2 = loss_metric.aggregate().item()
            jaccard_score_test = iou_metric.aggregate().item()

            # print(dice_score_test.shape) #torch.Size([1, 5])
            # print(jaccard_score_test_monai.shape) #torch.Size([1, 5])

            dice_metric.reset()
            loss_metric.reset()
            iou_metric.reset()

                
        torch.save({'epoch':epoch, 
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    }, checkpoint_path)
        
        # print(type(train_loss)) # <class 'monai.data.meta_tensor.MetaTensor'>
        # print(type(test_loss)) # <class 'monai.data.meta_tensor.MetaTensor'>

        # print(type(dice_score_train)) # <class 'float'>
        # print(type(dice_score_test)) # <class 'float'>

        # print(type(computed_dice_train)) # <class 'monai.data.meta_tensor.MetaTensor'>
        # print(type(computed_dice_test)) # <class 'monai.data.meta_tensor.MetaTensor'>
        # print(type(computed_iou_train)) # <class 'monai.data.meta_tensor.MetaTensor'>
        # print(type(computed_iou_test)) # <class 'monai.data.meta_tensor.MetaTensor'>

        # print(type(dice_score_train_2)) # <class 'float'>
        # print(type(dice_score_test_2)) # <class 'float'>
        # print(type(jaccard_score_train)) # <class 'float'>
        # print(type(jaccard_score_test)) # <class 'float'>

        wandb.log({"train_loss":train_loss.item(),
                   "test_loss":test_loss.item(),

                   "train_dice":dice_score_train,
                   "test_dice":dice_score_test, 

                   "Train Dice FROM FUNC":computed_dice_train.item(),
                   "Test Dice FROM FUNC": computed_dice_test.item(), 
                   "Train Iou FROM FUNC":computed_iou_train.item(), 
                   "Test Iou FROM FUNC":computed_iou_test.item(),

                   "train_dice_LOSS_METRIC":dice_score_train_2,
                   "test_dice_LOSS_METRIC":dice_score_test_2,
                   "train_jaccard":jaccard_score_train,
                   "test_jaccard":jaccard_score_test,

                   "image_wandb_train": [wandb.Image(img) for img in images_wandb_train],
                   "image_wandb_test": [wandb.Image(img) for img in images_wandb_test]
                   })    
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
            f'Train Loss: {train_loss:.4f}, '
            f'Test Loss: {test_loss:.4f}, '
            f'Train Dice Score: {dice_score_train:.4f}, '
            f'Test Dice Score: {dice_score_test:.4f}, '
            f'Train Dice FROM LOSS : {dice_score_train_2:.4f}, '
            f'Test Dice FROM LOSS: {dice_score_test_2:.4f}, '

            f'Train Dice FROM FUNC : {computed_dice_train:.4f}, '
            f'Test Dice FROM FUNC: {computed_dice_test:.4f}, '

            f'Train Iou FROM FUNC : {computed_iou_train:.4f}, '
            f'Test Iou FROM FUNC: {computed_iou_test:.4f}, '

            f'Train Jaccard: {jaccard_score_train:.4f}, '
            f'Test Jaccard: {jaccard_score_test:.4f}, '
            )
    
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
    # check_dataset(image_location, mask_location)

    # print(config_params["training_params"]["num_classes"]) #5

    
    train_dataloader, test_dataloader = get_from_loader(image_location, mask_location, config_params["training_params"]["num_classes"], wandb.config['batch_size'])                                   
    
    model, num_epochs,optimizer, loss= training_phase(train_dataloader,test_dataloader, config_params["training_params"]["num_classes"], wandb)

