import os
import cv2

import torch

class ShopeeDataset(torch.utils.data.Dataset):
    '''
    Define Shopee Dataset
    '''
    def __init__(self, df, transform=None):
        self.df = df
        self.root_dir = 'train_images'
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        # define image path
        image_path = os.path.join(self.root_dir, row.image)
        # read input image and transform
        transformed_image = cv2.imread(image_path)
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
        label = row.label_group

        augmented_image = self.transform(image=transformed_image)
        transformed_image = augmented_image['image']

        return {
            'image': transformed_image,
            'label': torch.tensor(label).long()
        }