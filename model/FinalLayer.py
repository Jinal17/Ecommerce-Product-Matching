import torch
import torch.nn.functional as F
from torch import nn
import math

class FinalLayer(nn.Module):
    '''
    Final layer to compute distance based on margin and scale.
    Cross Entropy Loss is used for loss function
    '''
    def __init__(self,
                 in_features,
                 out_features,
                 scale=30.0,
                 margin=0.50,
                 easy_margin=False):
        super(FinalLayer, self).__init__()
        # define nput features
        self.in_features = in_features
        # define output features
        self.out_features = out_features
        # define scale
        self.scale = scale
        # define margin 
        self.margin = margin
        # define weight
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))

        # define easy margin for further computation
        self.easy_margin = easy_margin
        # define cosine margin for further computation
        self.cos_margin = math.cos(margin)
        # define sine margin for further computation
        self.sin_margin = math.sin(margin)
        # define theta based on margin for further computation
        self.theta = math.cos(math.pi - margin)
        # define custom margin
        self.custom_margin = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        '''
        define forwarding layer
        '''
        # compute cosine based on final linear layer
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # compute sine based on computed cosine value
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # compute phi based on cosine and margins
        phi = cosine * self.cos_margin - sine * self.sin_margin
        # if needed, recompute phi
        phi = torch.where(cosine > self.theta, phi, cosine - self.custom_margin)
        # one hot encoded values
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # resize
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # compute output
        output = ((one_hot * phi) + ((1.0 - one_hot) * cosine))*self.scale
        # return computed output and loss values
        return output, nn.CrossEntropyLoss()(output,label)