#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import sys
sys.path.append('../input/timm-pytorch-image-models/pytorch-image-models-master')


# In[3]:


import numpy as np 
import pandas as pd 

import random 
import os 

from tqdm import tqdm 

import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2

import torch 
from torch.utils.data import Dataset 
import math
import gc
import cudf
import cuml
import cupy
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from albumentations import Compose, Resize, Normalize, HorizontalFlip, VerticalFlip,                           Rotate, CenterCrop
from transformers import AutoTokenizer, AutoModel
# In[4]:


class Shopee_Config:
    
    seed = 2020    
    device = 'cuda'

# In[5]:


def read_dataset():
    '''
    Read test dataset
    '''
    df = pd.read_csv('../input/shopee-product-matching/test.csv')
    # use cudf lib to load test data
    df_cu = cudf.DataFrame(df)
    # initialize image paths
    image_paths = '../input/shopee-product-matching/test_images/' + df['image']
    return df, df_cu, image_paths


# In[6]:

random.seed(Shopee_Config.seed)
os.environ['PYTHONHASHSEED'] = str(Shopee_Config.seed)
np.random.seed(Shopee_Config.seed)
torch.manual_seed(Shopee_Config.seed)
torch.cuda.manual_seed(Shopee_Config.seed)
torch.backends.cudnn.deterministic = True

# In[7]:


def combine_predictions(row):
    '''
    combine predictions made from image, text and pHash
    '''
    x = np.concatenate([row['image_predictions'], row['text_predictions'], row['phash_predictions']])
    return ' '.join( np.unique(x))

# In[8]:


def get_image_predictions(df, image_embeddings):
    '''
    compute image predictions
    '''

    # using KNN to get 50 nearest neighbors based on image embeddings using cosine similarity
    model = NearestNeighbors(n_neighbors=50, metric='cosine')
    model.fit(image_embeddings)
    # get nearest neighbors and their indices
    neighbors, indices = model.kneighbors(image_embeddings)
    
    image_predictions = []
    for key in tqdm(range(image_embeddings.shape[0])):
        # find index of the nearest neighbors
        index = np.where(neighbors[key,] < 0.36)[0]
        # find matched ids
        matched_ids = indices[key,index]
        # collect matched image Ids
        posting_ids = df['posting_id'].iloc[matched_ids].values
        # append to image predictions
        image_predictions.append(posting_ids)

    return image_predictions


# In[9]:


def transform_test_images():

    return A.Compose(
        [
            A.Resize(512,512,always_apply=True),
            A.Normalize(),
        ToTensorV2()
        ]
    )

from model import ShopeeDataset, EnsembleModel

# In[18]:


def get_image_embeddings(image_paths):
    embeds = []
    
    model = EnsembleModel()
    
    image_dataset = ShopeeDataset(image_paths=image_paths,transforms=transform_test_images())
    image_loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=12,
        pin_memory=True,
        drop_last=False,
        num_workers=4
    )
    
    
    with torch.no_grad():
        for image,label in tqdm(image_loader): 
            image = image.cuda()
            label = label.cuda()
            feature = model(image,label)
            image_embeddings = feature.detach().cpu().numpy()
            embeds.append(image_embeddings)
    
    
    del model
    image_embeddings = np.concatenate(embeds)
    print(f'Our image embeddings shape is {image_embeddings.shape}')
    return image_embeddings


### ArcFace
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if Shopee_Config.use_amp:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight)).float()  # if Shopee_Config.use_amp
        else:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device=Shopee_Config.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return output, self.criterion(output, label)



### BERT

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class ShopeeBertModel(nn.Module):
    def __init__(self, n_classes=Shopee_Config.classes, model_name=Shopee_Config.bert_model_name,
                fc_dim=Shopee_Config.fc_dim, margin=Shopee_Config.margin, scale=Shopee_Config.scale, use_fc=True):

        super(ShopeeBertModel, self).__init__()

        # Get the tokenizer and backbone from Huggingface
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name).to(Shopee_Config.device)

        # Establish number of dimensions for embeddings
        in_features = 768
        self.use_fc = use_fc

        if use_fc:
            # Linear leayer and batch normalization
            self.classifier = nn.Linear(in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)

            # Initialize weights for the layers above
            self._init_params()
            in_features = fc_dim

        # Define the ArcFace loss
        self.final = ArcMarginProduct(in_features, n_classes, scale=scale, 
                                    margin=margin, easy_margin=False, ls_eps=0.0)

    # Initialize weights
    def _init_params(self):
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, texts, labels=torch.tensor([0])):

        # Get the tokenized features from our language model
        features = self.extract_features(texts)
        if self.training:

            # Run our embeddings through the ArcFace loss
            logits = self.final(features, labels.to(Shopee_Config.device))
            return logits
        else:
            return features

    # Utilize the language model to obtain embeddings
    def extract_features(self, texts):
        encoding = self.tokenizer(texts, padding=True, truncation=True,
                                  max_length=Shopee_Config.max_length, return_tensors='pt').to(Shopee_Config.device)

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        embedding = self.backbone(input_ids, attention_mask=attention_mask)
        x = mean_pooling(embedding, attention_mask)

        if self.use_fc and self.training:
            x = self.classifier(x)
            x = self.bn(x)

        return x


# In[19]:
data = pd.read_csv("../input/shopee-product-matching/test.csv")
text_model = ShopeeBertModel()
text_model.to(Shopee_Config.device);

def get_bert_embeddings(df, column, model, chunk=32):
    model.eval()

    # Initialize embedding vector
    bert_embeddings = torch.zeros((df.shape[0], 768)).to(Shopee_Config.device)

    # Splice into chunks as to relax the amount of data placed into memory
    for i in tqdm(list(range(0, df.shape[0], chunk)) + [df.shape[0] - chunk], desc="get_bert_embeddings", ncols=80):
        titles = []

        # Read the titles in the chunk
        for title in df[column][i: i + chunk].values:

            # Attempt to read it in Unicode
            try:
                title = title.encode('utf-8').decode("unicode_escape")
                title = title.encode('ascii', 'ignore').decode("unicode_escape")
            except:

                # Ignore titles that couldn't be read
                pass
            
            titles.append(title.lower())

        # Run the title through the model
        with torch.no_grad():
            if Shopee_Config.use_amp:
                with torch.cuda.amp.autocast():
                    model_output = model(titles)
            else:
                model_output = model(titles)

        # Set the embedding
        bert_embeddings[i: i + chunk] = model_output

    # Perform garbage collection
    del model, titles, model_output
    gc.collect()
    torch.cuda.empty_cache()

    return bert_embeddings


def get_neighbors(df, embeddings, knn=50, threshold=0.0):
    # Create a nearest neighbors model and fit the embeddings
    model = NearestNeighbors(n_neighbors=knn, metric='cosine')
    model.fit(embeddings)

    # Get the neighbors for each embeddings
    distances, indices = model.kneighbors(embeddings)

    preds = []

    #Go through our embeddings
    for k in range(embeddings.shape[0]):
        idx = np.where(distances[k,] < threshold)[0]
        ids = indices[k, idx]

        # Add to our predictions similarities greater than the threshold
        posting_ids = df['posting_id'].iloc[ids].values
        preds.append(posting_ids)

    # Perform garbage collection
    del model, distances, indices
    gc.collect()
    return preds

def get_text_predictions(data):
    
    # Load the model that we have previously trained so we dont waste time training again
    text_model.load_state_dict(torch.load('../input/xlmmultilingual/paraphrase-xlm-r-multilingual-v1.pt', 
                                            map_location=Shopee_Config.device))

    embeddings = get_bert_embeddings(data, 'title', text_model)
    predictions = get_neighbors(data, embeddings.detach().cpu().numpy(),
                                      knn=70, threshold=0.39)

    # In[36]:

    # Add predictions to dataframe
    return predictions


# In[20]:


df,df_cu,image_paths = read_dataset()
df.head()


# In[21]:


image_embeddings = get_image_embeddings(image_paths.values)


# In[22]:

# get image predictions based on image embeddings
image_predictions = get_image_predictions(df, image_embeddings)
# get text predictions based on text embeddings
text_predictions = get_text_predictions(data)


# In[23]:
# grouping by image pHash
duplicate_dict = df.groupby('image_phash').posting_id.agg('unique').to_dict()
df['phash_predictions'] = df["image_phash"].map(duplicate_dict)


# In[24]:


df['image_predictions'] = image_predictions
df['text_predictions'] = text_predictions
# combine image, text, pHash predictions
df['matches'] = df.apply(combine_predictions, axis = 1)
# create submission file
df[['posting_id', 'matches']].to_csv('submission.csv', index=False)

# In[ ]:




