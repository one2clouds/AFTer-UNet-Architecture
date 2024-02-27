from Data.prepare_dataloader import get_from_loader
import torch
from dice import DiceLoss
import torch.optim as optim
from axial_fusion_transformer import axial_fusion_transformer
import os
from tqdm import tqdm 
from calculate_metrics import calculate_metrics
from operator import add
import yaml 
import json
import matplotlib.pyplot as plt

with open('./config/train_config.yaml', 'r') as config_file:
    config_params = yaml.safe_load(config_file)
    model_config = json.dumps(config_params)

def training_phase(train_dataloader, test_dataloader, num_classes):
    overall_train_loss_per_epoch = []
    overall_test_loss_per_epoch = []
    overall_train_jaccard_per_epoch = []
    overall_test_jaccard_per_epoch = []
    overall_train_acc_per_epoch = []
    overall_test_acc_per_epoch = []

    num_epochs = config_params["training_params"]["num_epochs"]

    Na = config_params["training_params"]["Na"]
    Nf = config_params["training_params"]["Nf"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config_params["training_params"]["model_name"] == "axial_fusion_transformer":
        model = axial_fusion_transformer(Na,Nf, num_classes).to(device)
    loss_function = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


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
        metrics_score = [0.0, 0.0]

        model.train()
        torch.cuda.empty_cache()

        for img_and_mask in tqdm(train_dataloader, desc=f'Training Epoch {epoch}/{num_epochs}', unit='epoch'):
            torch.cuda.empty_cache()
            
            image = img_and_mask['_3d_image']['data'].float().to(device)
            mask = img_and_mask['_3d_mask']['data'].float().to(device)
            optimizer.zero_grad()
            output = model(image)
    #         print(output.shape) # torch.Size([4, 128, 128, 128])
    #         print(mask.squeeze(dim=0).shape) # torch.Size([1, 4, 128, 128, 128])
            train_loss = loss_function(output, mask.squeeze(dim=0))
            train_loss.backward()

            score = calculate_metrics(output, mask)
            metrics_score = list(map(add,metrics_score, score))
            
            optimizer.step()
            epoch_train_loss += train_loss.item()
            
            epoch_train_loss = epoch_train_loss/len(train_dataloader)
            epoch_train_jaccard = metrics_score[0]/len(train_dataloader)
            epoch_train_acc = metrics_score[1]/len(train_dataloader)
            
        overall_train_loss_per_epoch.append(train_loss.item())
        overall_train_jaccard_per_epoch.append(epoch_train_jaccard)
        overall_train_acc_per_epoch.append(epoch_train_acc)
        
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
                optimizer.step()
                
                epoch_test_loss += test_loss.item()
                epoch_test_loss = epoch_train_loss/len(test_dataloader)
                epoch_test_jaccard = metrics_score[0]/len(test_dataloader)
                epoch_test_acc = metrics_score[1]/len(test_dataloader)
                
                torch.save({'epoch':start_epoch, 
                            'model_state_dict':model.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict(),
                            }, checkpoint_path)


            overall_test_loss_per_epoch.append(test_loss.item()) 
            overall_test_jaccard_per_epoch.append(epoch_test_jaccard)
            overall_test_acc_per_epoch.append(epoch_test_acc)
            
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
            f'Train Loss: {train_loss.item():.4f}, '
            f'Train Jaccard: {epoch_train_jaccard.item():.4f}, '
            f'Train Accuracy: {epoch_train_acc:.4f}, '
            f'Test Loss: {test_loss.item():.4f}, '
            f'Test Jaccard: {epoch_test_jaccard.item():.4f}, '
            f'Test Accuracy: {epoch_test_acc:.4f}, ')
    
    return model,num_epochs,optimizer, train_loss, overall_train_loss_per_epoch, overall_train_jaccard_per_epoch, overall_train_acc_per_epoch, overall_test_loss_per_epoch, overall_test_jaccard_per_epoch, overall_test_acc_per_epoch






if __name__ =='__main__':
    image_location = '/mnt/Enterprise2/shirshak/SegTHOR/train/P*/P*.nii.gz'
    mask_location = '/mnt/Enterprise2/shirshak/SegTHOR/train/P*/G*.nii.gz'
    
    train_dataloader, test_dataloader = get_from_loader(image_location, mask_location, config_params["training_params"]["num_classes"])
                                                        
    # model, num_epochs,optimizer, loss, overall_train_loss_per_epoch, overall_train_jaccard_per_epoch, overall_train_acc_per_epoch, overall_test_loss_per_epoch, overall_test_jaccard_per_epoch, overall_test_acc_per_epoch = training_phase(train_dataloader,test_dataloader, config_params["training_params"]["num_classes"])

    Na = config_params["training_params"]["Na"]
    Nf = config_params["training_params"]["Nf"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = axial_fusion_transformer(Na,Nf, config_params["training_params"]["num_classes"]).to(device)
    
    model.load_state_dict(torch.load('model.pth')['model_state_dict'])
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

        n_slice = 55
        ax[0].imshow(image[:,:,n_slice].cpu())
        ax[0].set_title('Image of SegTHOR CT Scan')
        ax[1].imshow(mask[:,:,n_slice].cpu())
        ax[1].set_title('Original Mask')
        ax[2].imshow(output[:,:,n_slice].cpu() > 0.5)
        ax[2].set_title('Predicted Mask')

        fig.savefig(f'images/full_figure{i}.png')
        i += 1

