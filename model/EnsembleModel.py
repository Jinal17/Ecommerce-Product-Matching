from torch import nn

from model import ShopeeModel
import torch


def get_model(model_name=None, model_path=None, n_classes=None):
    model = ShopeeModel(model_name=model_name)
    model.eval()
    model.load_state_dict(torch.load(model_path))
    model = model.to('cuda')

    return model

class EnsembleModel(nn.Module):
    '''
    Ensemble model is a combination of nfnet, efficientNet and resnet
    '''

    def __init__(self):
        super(EnsembleModel, self).__init__()

        self.m1 = get_model('eca_nfnet_l0', '../input/d/jinal17/models/nfnet_model.pt')
        self.m2 = get_model('efficientnet_b0', '../input/d/jinal17/models/efficientnet_model.pt')
        self.m3 = get_model('resnet50', '../input/d/jinal17/models/resnet50_model.pt')

    def forward(self, image, label):
        feature1 = self.m1(image, label)
        feature2 = self.m2(image, label)
        feature3 = self.m3(image, label)

        return (feature1 + feature2 + feature3) / 3




