# Text training


### Dataset
from tqdm import tqdm
import torch


class TitleDataset(torch.utils.data.Dataset):
    def __init__(self, df, text_column, label_column):
        texts = df[text_column]
        self.labels = df[label_column].values

        self.titles = []
        for title in texts:
            title = title.encode('utf-8').decode("unicode_escape")
            title = title.encode('ascii', 'ignore').decode("unicode_escape")
            title = title.lower()
            self.titles.append(title)

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        text = self.titles[idx]
        label = torch.tensor(self.labels[idx])
        return text, label


# In[21]:


### SAM Optimizer 2020/1/16
# https://github.com/davda54/sam/blob/main/sam.py

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


# In[22]:


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import gc
from transformers import AutoTokenizer, AutoModel

# In[24]:


### Train one epoch

def train_fn(model, data_loader, optimizer, scheduler, use_sam, accum_iter, epoch, device, use_amp):
    model.train()
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    fin_loss = 0.0
    tk = tqdm(data_loader, desc="Training epoch: " + str(epoch + 1), ncols=100)

    for t, (texts, labels) in enumerate(tk):
        texts = list(texts)

        if use_sam:
            if use_amp:
                with torch.cuda.amp.autocast():
                    _, loss = model(texts, labels)
                loss.mean().backward()
                optimizer.first_step(zero_grad=True)
                fin_loss += loss.item()
                with torch.cuda.amp.autocast():
                    _, loss_second = model(texts, labels)
                loss_second.mean().backward()
                optimizer.second_step(zero_grad=True)
                optimizer.zero_grad()
            else:
                _, loss = model(texts, labels)
                loss.mean().backward()
                optimizer.first_step(zero_grad=True)
                fin_loss += loss.item()
                _, loss_second = model(texts, labels)
                loss_second.mean().backward()
                optimizer.second_step(zero_grad=True)
                optimizer.zero_grad()

        else:  # if use_sam == False
            if use_amp:
                with torch.cuda.amp.autocast():
                    _, loss = model(texts, labels)
                scaler.scale(loss).backward()
                fin_loss += loss.item()
                # mini-batch accumulation
                if (t + 1) % accum_iter == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                _, loss = model(texts, labels)
                loss.backward()
                fin_loss += loss.item()
                # mini-batch accumulation
                if (t + 1) % accum_iter == 0:
                    optimizer.step()
                    optimizer.zero_grad()

        tk.set_postfix({'loss': '%.6f' % float(fin_loss / (t + 1)), 'LR': optimizer.param_groups[0]['lr']})

    scheduler.step()
    return model, fin_loss / len(data_loader)


# In[25]:


### Validation

def getMetric(col):
    def f1score(row):
        n = len(np.intersect1d(row.target, row[col]))
        return 2 * n / (len(row.target) + len(row[col]))

    return f1score


def get_bert_embeddings(df, column, model, chunk=32):
    model.eval()

    bert_embeddings = torch.zeros((df.shape[0], 768)).to(CFG.device)
    for i in tqdm(list(range(0, df.shape[0], chunk)) + [df.shape[0] - chunk], desc="get_bert_embeddings", ncols=80):
        titles = []
        for title in df[column][i: i + chunk].values:
            try:
                title = title.encode('utf-8').decode("unicode_escape")
                title = title.encode('ascii', 'ignore').decode("unicode_escape")
            except:
                pass
            # title = text_punctuation(title)
            title = title.lower()
            titles.append(title)

        with torch.no_grad():
            if CFG.use_amp:
                with torch.cuda.amp.autocast():
                    model_output = model(titles)
            else:
                model_output = model(titles)

        bert_embeddings[i: i + chunk] = model_output

    del model, titles, model_output
    gc.collect()
    torch.cuda.empty_cache()

    return bert_embeddings

import sys

class CFG:
    compute_cv = True  # set False to train model for submission

    ### BERT
    if 'kaggle_web_client' in sys.modules:  # for kaggle notebook
        bert_model_name = '../input/bertmodel/paraphrase-xlm-r-multilingual-v1'
    else:
        bert_model_name = '../input/bertmodel/distilbert-base-indonesian'
    max_length = 128

    ### ArcFace
    scale = 30
    margin = 0.8
    fc_dim = 768
    seed = 412
    classes = 11014

    ### Training
    n_splits = 4  # GroupKFold(n_splits)
    batch_size = 16
    accum_iter = 1  # 1 if use_sam = True
    epochs = 10
    min_save_epoch = epochs // 3
    use_sam = True  # SAM (Sharpness-Aware Minimization for Efficiently Improving Generalization)
    use_amp = True  # Automatic Mixed Precision
    num_workers = 2  # On Windows, set 0 or export train_fn and TitleDataset as .py files for faster training.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    ### NearestNeighbors
    bert_knn = 50
    bert_knn_threshold = 0.4  # Cosine distance threshold

    scheduler_params = {
        "lr_start": 7.5e-6,
        "lr_max": 1e-4,
        "lr_min": 1e-8,  # 2.74e-5, # 1.5e-5,
    }
    multiplier = scheduler_params['lr_max'] / scheduler_params['lr_start']
    eta_min = scheduler_params['lr_min']  # last minimum learning rate
    freeze_epo = 0
    warmup_epo = 2
    cosine_epo = epochs - freeze_epo - warmup_epo
    save_model_path = f"./{bert_model_name.rsplit('/', 1)[-1]}_epoch{epochs}-bs{batch_size}x{accum_iter}.pt"
    print("save_model_path ", save_model_path)


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
        if CFG.use_amp:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight)).float()  # if CFG.use_amp
        else:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device=CFG.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return output, self.criterion(output, label)


# In[27]:


### BERT

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class ShopeeBertModel(nn.Module):
    def __init__(
            self,
            n_classes=CFG.classes,
            model_name=CFG.bert_model_name,
            fc_dim=CFG.fc_dim,
            margin=CFG.margin,
            scale=CFG.scale,
            use_fc=True
    ):

        super(ShopeeBertModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name).to(CFG.device)

        in_features = 768
        self.use_fc = use_fc

        if use_fc:
            self.dropout = nn.Dropout(p=0.0)
            self.classifier = nn.Linear(in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            in_features = fc_dim

        self.final = ArcMarginProduct(
            in_features,
            n_classes,
            scale=scale,
            margin=margin,
            easy_margin=False,
            ls_eps=0.0
        )

    def _init_params(self):
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, texts, labels=torch.tensor([0])):
        features = self.extract_features(texts)
        if self.training:
            logits = self.final(features, labels.to(CFG.device))
            return logits
        else:
            return features

    def extract_features(self, texts):
        encoding = self.tokenizer(texts, padding=True, truncation=True,
                                  max_length=CFG.max_length, return_tensors='pt').to(CFG.device)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        embedding = self.backbone(input_ids, attention_mask=attention_mask)
        x = mean_pooling(embedding, attention_mask)

        if self.use_fc and self.training:
            x = self.dropout(x)
            x = self.classifier(x)
            x = self.bn(x)

        return x


### Create Model

model = ShopeeBertModel()
model.to(CFG.device)
torch.save(model.state_dict(),CFG.save_model_path)