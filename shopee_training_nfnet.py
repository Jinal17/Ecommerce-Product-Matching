# -*- coding: utf-8 -*-
"""Shopee_custom_training_simple.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1p1XCfodXT-GnLu2tTID3Eg0BLzRMgJxr
"""

from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir("/content/drive/MyDrive/Shopee")

# RAPIDs setup on google colab to run cudf, cuml libs
!nvidia-smi

!git clone https://github.com/rapidsai/rapidsai-csp-utils.git
!python rapidsai-csp-utils/colab/env-check.py

# This will update the Colab environment and restart the kernel.  Don't run the next cell until you see the session crash.
!bash rapidsai-csp-utils/colab/update_gcc.sh
import os
os._exit(00)

# This will install CondaColab.  This will restart your kernel one last time.  Run this cell by itself and only run the next cell once you see the session crash.
import condacolab
condacolab.install()

# you can now run the rest of the cells as normal
import condacolab
condacolab.check()

from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir("/content/drive/MyDrive/Shopee")

# Installing RAPIDS is now 'python rapidsai-csp-utils/colab/install_rapids.py <release> <packages>'
# The <release> options are 'stable' and 'nightly'.  Leaving it blank or adding any other words will default to stable.
# The <packages> option are default blank or 'core'.  By default, we install RAPIDSAI and BlazingSQL.  The 'core' option will install only RAPIDSAI and not include BlazingSQL, 
!python rapidsai-csp-utils/colab/install_rapids.py stable
import os
os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda/nvvm/lib64/libnvvm.so'
os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda/nvvm/libdevice/'
os.environ['CONDA_PREFIX'] = '/usr/local'

!pip3 install timm
!pip3 install albumentations --no-binary imgaug,albumentations
!pip3 uninstall opencv-python
!pip3 install opencv-python
!pip3 install kaggle

# !mkdir ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json

# !kaggle competitions download shopee-product-matching

# !unzip shopee-product-matching.zip

import pandas as pd
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

import torch

from tqdm.notebook import tqdm 
from sklearn.preprocessing import LabelEncoder

class CommonConfig:

    EPOCHS = 15

    NUM_WORKERS = 4
    DEVICE = 'cuda'

def transform_image():
    '''
    Image transformations by applying albumentations lib
    '''
    return albumentations.Compose(
        [   
            albumentations.Resize(512, 512, always_apply=True),
            albumentations.HorizontalFlip(),
            albumentations.VerticalFlip(),
            albumentations.Rotate(limit=120),
            albumentations.RandomBrightness(limit=(0.09, 0.6)),
            albumentations.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

def train(model, data_loader, optimizer, i):
    '''
    Train Shopee model and compute loss
    '''
    model.train()
    final_loss = 0.0
    tk = tqdm(data_loader, desc = "Epoch" + " [TRAIN] " + str(i+1))

    for ptr,image in enumerate(tk):
        for key,value in image.items():
            image[key] = value.to(CommonConfig.DEVICE)
        # initialize optimizer
        optimizer.zero_grad()
        # compute loss
        _, loss = model(**image)
        loss.backward()
        optimizer.step()
        # accumulate train loss
        final_loss += loss.item()

        # print loss after every iteration
        tk.set_postfix({'loss' : '%.6f' %float(final_loss/(ptr+1)), 'LR' : optimizer.param_groups[0]['lr']})

    return final_loss / len(data_loader)


# Image training
from model import ShopeeDataset, ShopeeModel
# collect training loss for plotting
training_losses = []
def training():
    '''
    Training the model
    '''
    # read training metadata
    df = pd.read_csv('train.csv')

    # defining label encoder for label group
    labelencoder=LabelEncoder()
    df['label_group'] = labelencoder.fit_transform(df['label_group'])

    # create training set on transformed images
    trainset = ShopeeDataset(df, transform = transform_image())

    data_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size = 8
    )

    # define custom model
    model = ShopeeModel()
    model.to(CommonConfig.DEVICE)

    # using Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000092)

    # iterate through every epoch
    for ptr in range(CommonConfig.EPOCHS):

        # train the model and compute loss
        avg_loss_train = train(model, data_loader, optimizer, ptr)
        print("avg_loss_train ", avg_loss_train)
        # collect training loss after every epoch
        training_losses.append(avg_loss_train)
        # update and save model after every epoch
        torch.save(model.state_dict(),'nfnet_model.pt')

training()


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(training_losses, label="training_loss")
indices = [i for i in range(CommonConfig.EPOCHS)]
plt.plot(indices, training_losses, label="train_loss")
plt.title("Training Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()





