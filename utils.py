import torch 
import matplotlib.pyplot as plt
import random




def convert3d_image2_slices(a_3d_image):
    ct_scan_slices = []
    D = a_3d_image.shape[-1]
    for d in range(D):
        # just remove the batch_size but keep channel so only squeezing outside channel
        one_slice = a_3d_image.squeeze(dim=0)[:,:,d]    # sabae channel,height,width dinxa according to D widths
        ct_scan_slices.append(one_slice)
    return torch.stack(ct_scan_slices)


def append_neighboring_slices(Na,Nf,D,ct_scan_slices):
    x = []
    for d in range(D):
        neighbor_slices = []
        for n in range(Na):
            an = d - Nf * (Na//2 - n) 
            if an < 0:
                an = an + D
            elif an >= D:
                an = an - D
            neighbor_slices.append(ct_scan_slices[an])
        t_neighbor_slices = torch.stack(neighbor_slices)
#         print(t_neighbor_slices.shape)
#             print(an)
        x.append(t_neighbor_slices)
#     print(len(x))
    final_slices = torch.stack(x)
    
#     print(final_slices.shape)
    return final_slices

def visualize_img_mask_output(img, mask, output, num_channels_before_training):
    # print(image.shape)  # 4,1,128,128,128
    # print(mask.shape) # torch.Size([batch_size,128, 128, 128])
    # print(output.shape) # torch.Size([batch_size,128, 128, 128])

    figs = []
    # print(img.shape) # torch.Size([2, 1, 128, 128, 128]) # torch.Size([batch_size,num_channel_before_training, 128, 128, 128])
    # print(mask.shape) # torch.Size([2, 5, 128, 128, 128]) # torch.Size([batch_size,num_classes,128,128,128])
    # print(output.shape) # torch.Size([2, 5, 128, 128, 128]) # torch.Size([batch_size,num_classes,128,128,128])
    for image,mask,output in zip(img,mask,output):
        # print(image.shape) # torch.Size([1, 128, 128, 128])
        # print(mask.shape) # torch.Size([5, 128, 128, 128])
        # print(output.shape) # torch.Size([5, 128, 128, 128])
        if num_channels_before_training == 1:
            fig, ax = plt.subplots(1,3)
            n_slice = int(random.random() * 128) # 55
            image = image[0] # Taking the first channel of the image
            mask = mask.argmax(dim=0)
            output = output.argmax(dim=0)
            ax[0].imshow(image[:,:,n_slice].cpu())
            ax[0].set_title('Image of SegTHOR CT Scan')
            ax[1].imshow(mask[:,:,n_slice].cpu())
            ax[1].set_title('Original Mask')
            ax[2].imshow(output[:,:,n_slice].cpu())
            ax[2].set_title('Predicted Mask')

        if num_channels_before_training == 3:
            fig, ax = plt.subplots(1,3)
            n_slice = int(random.random() * 128) # 55
            image = image[0] # Taking the first channel of the image
            mask = mask.argmax(dim=0)
            output = output.argmax(dim=0)
            ax[0].imshow(image[:,:,n_slice].cpu())
            ax[0].set_title('Image of SegTHOR CT Scan')
            ax[1].imshow(mask[:,:,n_slice].cpu())
            ax[1].set_title('Original Mask')
            ax[2].imshow(output[:,:,n_slice].cpu())
            ax[2].set_title('Predicted Mask')

        figs.append(fig)

    # fig.savefig(f'images/figure {n_slice}.png')
    return figs
