import timm
from torch import nn
from model import FinalLayer

class ShopeeModel(nn.Module):

    '''
    Define custom model for training image dataset
    '''
    def __init__(
        self,
        n_classes=11014,
        model_name='eca_nfnet_l0',
        fc_dim=512):

        super(ShopeeModel,self).__init__()

        self.main = timm.create_model(model_name, pretrained=False)
        # initialize input features
        input_features = self\
            .main\
            .head\
            .fc\
            .in_features
        self.main.head.fc = nn.Identity()
        # define global pool
        self.main.head.global_pool = nn.Identity()
        # define pooling layer
        self.pooling =  nn.AdaptiveAvgPool2d(1)
        # adding dropout
        self.dropout = nn.Dropout(p=0.0)
        # define fully connected layer
        self.fc = nn.Linear(input_features, fc_dim)
        # adding batch normalization
        self.bn = nn.BatchNorm1d(fc_dim)
        # init params
        self._init_params()
        # use input features
        input_features = fc_dim
        # define final layer
        self.final = FinalLayer(
            input_features,
            n_classes
        )

    def _init_params(self):
        '''
        Initialize params
        '''
        # initialize bias for fully connected layer
        nn.init.constant_(self.fc.bias, 0)
        # initialize weight
        nn.init.constant_(self.bn.weight, 1)
        # initialize bias for batch norm
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, image, label):
        '''
        define forwarding layer
        '''
        # extract image features
        feature = self.extract_image_features(image)
        # compute logits
        logits = self.final(feature,label)
        return logits

    def extract_image_features(self, x):
        '''
        Extract image features
        '''
        # define batch size
        batch_size = x.shape[0]
        x = self.main(x)
        # add pooling
        x = self.pooling(x)
        # resize image
        x = x.view(batch_size, -1)

        # add dropout layers
        x = self.dropout(x)
        # add fully connected layer
        x = self.fc(x)
        # add batch norm
        x = self.bn(x)
        return x