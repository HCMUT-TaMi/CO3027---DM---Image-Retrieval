import torch.nn as nn
import torch.functional as F
from functools import reduce

from model.backbone.utils import *

# DEFAULT CONFIG
RESNET50 = [
    (256, 3),
    (512, 4),  
    (1024, 6), 
    (2048, 3)  
]

class BottleNeck(nn.Module): 
    def __init__(self, img_size, input_c, output_c, **kwargs): 
        """
        Input:
            img_size (tuple(int, int)): image size HxW
            input_c (int): input channels shape 
            output_c (int): output channels shape

        Output: 
            Bottle Neck accept BxHxWxc_in -> Bx(H/2)x(W/2)xc_out 

        """

        super().__init__()
        # assert output_c > input_c, "The output channels must be larger than input channels"
        self.H, self.W = img_size
        self.stride1, self.stride3 = kwargs.get("stride1", 1), kwargs.get("stride3", 2) # recommended by community 
        self.reluClamp = kwargs.get("relu", 0.01)
        self.bneps = kwargs.get("bn", 5e-3)
        self.group = kwargs.get("group", 2)
        hidden_dim = kwargs.get("hidden_dim", input_c // 2)

        self.conv1 = nn.Conv2d(kernel_size=1, in_channels=input_c, out_channels=hidden_dim, stride = self.stride1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=hidden_dim, eps = self.bneps)
        self.relu1 = reluS(self.reluClamp)

        self.conv2 = nn.Conv2d(kernel_size=1, in_channels=hidden_dim, out_channels=hidden_dim, stride = self.stride3, padding=1, groups = self.group,bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=hidden_dim, eps = self.bneps)
        self.relu2 = reluS(self.reluClamp)

        self.conv3 = nn.Conv2d(kernel_size=1, in_channels=hidden_dim, out_channels=output_c, stride = self.stride1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=output_c, eps = self.bneps)
        self.relu3 = reluS(self.reluClamp)

        #   Skip connection
        if input_c != output_c or self.stride3 != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_c, output_c, kernel_size=1, stride=self.stride3, bias=False),
                nn.BatchNorm2d(output_c, eps = self.bneps)
            )

        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if identity.shape != out.shape:
            identity = F.interpolate(identity, size=out.shape[2:], mode='bilinear', align_corners=False)
        out += identity
        out = self.relu3(out)
        return out

class ResNetBlock(nn.Module): 
    def __init__(self, img_size, in_c, out_c, d, **kwargs) -> None:
        super(ResNetBlock, self).__init__()
        self.H, self.W = img_size
        self.stride1, self.stride3 = kwargs.get("stride1", 1), kwargs.get("stride3", 2) # recommended by community 
        self.reluClamp = kwargs.get("relu", 0.01)
        self.bneps = kwargs.get("bn", 5e-3)
        self.group = kwargs.get("group", 2)
        hidden_dim = kwargs.get("hidden_dim", in_c // 2)
        self.blocks = []
        for _ in range(d):
            self.blocks.append(BottleNeck(img_size=img_size, input_c=in_c, output_c=out_c, stride1=self.stride1, stride3=self.stride3, bn = self.bneps, group = self.group, hidden_dim = hidden_dim))
            in_c = out_c

    def forward(self, X):
        return reduce(
            lambda x,y: y(x), 
            self.blocks,
            X
        ) 

class ResNet(nn.Module):
    """ResNet backbone with RGEM + GEMP, single-scale forward only."""

    def __init__(self, depth, reduction_dim, relup=0.01,
                 img_size=(224,224),
                 num_groups=2, width_per_group=32, **kwargs):
        
        super().__init__()

        self.depth = depth
        self.reduction_dim = reduction_dim
        self.relup = relup
        self.img_size = img_size
        
        depth_config = {
            50:  (3, 4, 6, 3),
            101: (3, 4, 23, 3),
            152: (3, 8, 36, 3),
        }
        (d1, d2, d3, d4) = depth_config[self.depth]

        g = num_groups
        gw = width_per_group
        base_w = gw * g

        # ------------ Stem ------------
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64, eps=5e-3),
            reluS(relup),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # ------------ ResNet Stages ------------
        H, W = img_size
        self.s1 = ResNetBlock((H//4,  W//4),   64,   256, d1, **kwargs)
        self.s2 = ResNetBlock((H//8,  W//8),   256,  512, d2, **kwargs)
        self.s3 = ResNetBlock((H//16, W//16),  512, 1024, d3, **kwargs)
        self.s4 = ResNetBlock((H//32, W//32), 1024, 2048, d4, **kwargs)

        # ------------ RGEM + GEM + FC ------------
        self.rgem = rgem()
        self.gemp = gemp()
        self.sgem = sgem()
        self.fc = nn.Linear(2048, reduction_dim)

    def forward(self, x):
        # Backbone
        x = self.stem(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)

        x = self.rgem(x)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)
        x = self.gemp(x)
        x = self.sgem([x])

        x = F.normalize(x, p=2, dim=-1)
        x = x.squeeze(-1)
        x = self.fc(x)

        return x