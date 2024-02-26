import torch 




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