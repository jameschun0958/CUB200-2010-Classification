import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable

from torchinfo import summary

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

# Define the ResNet50-based Model
class resnet50(nn.Module):

    def __init__(self, num_classes, droprate=0.5, stride=2):
        super(resnet50, self).__init__()

        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft

        # extract global feature
        self.feature_layer = nn.Linear(2048, 512)
        self.feature_layer.apply(weights_init_kaiming)

        # BNNeck (inference layer)
        self.bnneck = nn.BatchNorm1d(512)
        self.bnneck.bias.requires_grad_(False)
        self.bnneck.apply(weights_init_kaiming)

        # global classifier
        self.last_layer = nn.Sequential(
            nn.Linear(512, num_classes)
        )
        self.last_layer.apply(weights_init_classifier)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        feat_maps = self.model.layer3(x)
        x = self.model.layer4(feat_maps)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        global_feat = self.feature_layer(x) # for triple loss
        feat = self.bnneck(global_feat)
        out = self.last_layer(feat) # for ID loss

        return out, global_feat

'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
    net = resnet50(num_classes=200, )

    summary(net.cuda(), (1, 3, 256, 256))
    net.to('cpu')
