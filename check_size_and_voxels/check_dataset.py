import pandas as pd
from tqdm import tqdm 
import nibabel as nib 
import glob

def print_max_and_min_values(df):
    values = df['image'].apply(lambda x: x[0])
    print(f'X_min value is :{values.apply(lambda x: x[0]).min()}')
    print(f'X_max value is :{values.apply(lambda x: x[0]).max()}')
    print(f'Y_min value is :{values.apply(lambda x: x[1]).min()}')
    print(f'Y_max value is :{values.apply(lambda x: x[1]).max()}')





image_location = '/mnt/Enterprise2/shirshak/SegTHOR/train/P*/P*.nii.gz'
mask_location = '/mnt/Enterprise2/shirshak/SegTHOR/train/P*/G*.nii.gz'


def check_dataset(image_location, mask_location):
    image_list = sorted(glob.glob(image_location))
    mask_list = sorted(glob.glob(mask_location))

    columns = ['image', 'segmentation']
    df = pd.DataFrame(columns=columns)

    for index in tqdm(range(len(image_list[:25]))):
        image_size=nib.load(image_list[index]).get_fdata()
        image_voxel_sizes = nib.load(image_list[index]).header.get_zooms()

        seg_size=nib.load(mask_list[index]).get_fdata()
        seg_voxel_sizes = nib.load(mask_list[index]).header.get_zooms()

        df.loc[index+1, 'image'] = image_size[index].shape, image_voxel_sizes
        df.loc[index+1, 'segmentation'] = seg_size[index].shape, seg_voxel_sizes

    # print_max_and_min_values(df)
    df.to_csv('check_size_and_voxels/sizes_and_voxels.csv')