import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
from torchsummary import summary

# __all__ = ['resnet50_feature']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers,cls_num, **kwargs):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_embed = nn.Linear(512 * block.expansion, cls_num)
        self.conv1_1 = nn.Conv2d(512, 1024, kernel_size=1, stride=1)
        # if with_fc:
        #     self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x2 = self.maxpool(x)
        x2 = self.conv1_1(x2)
        x = self.layer3(x)
        x_local = torch.add(x2, x)
        x1 = self.layer4(x_local)
        x1 = self.avgpool(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc_embed(x1)
        x = torch.cat((x, x_local), dim=1)

        return x, x1
class Conv_att(nn.Module):
    def __init__(self,inplanes, num_classes):
        super(Conv_att,self).__init__()
        self.conv_att = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=512, kernel_size=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
        )
    def forword(self,x):
        x = self.conv_att(x)
        x = F.softmax(x, dim=1)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEblock(nn.Module):
    def __init__(self,inplanes):
        super(SEblock,self).__init__()
        self.se=nn.Sequential(
            SELayer(inplanes),
            nn.Conv2d(inplanes,out_channels=256,kernel_size=1),
            SELayer(channel=256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            SELayer(channel=256),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1),
            SELayer(channel=512),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1),
        )

    def forword(self, x):
        x=self.se(x)
        return x

class SEnet(nn.Module):
    def __init__(self,inplanes=1024,num_classes=10,a=0.45):
        super(SEnet,self).__init__()
        self.res=ResNet(Bottleneck, [3, 4, 6])
        self.attention=Conv_att(inplanes,num_classes)
        self.se=SEblock(inplanes=num_classes)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(inplanes, num_classes, bias=False)
        )
        self.alpha=weights(a)

    def forword(self, x):
        x=self.res(x)
        y=self.attention(x)
        y=self.se(y)
        y=self.fc(y)
        x=self.fc(x)
        out=self.alpha*x+(1-self.alpha)*y
        out=F.softmax(out,dim=0)
        return out

def weights(a=0.45):
    alpha = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
    alpha.data.fill_(a)
    return alpha


# if __name__=="__main__":
#     resnet50=SEnet(num_classes=10)
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model=resnet50.to(device)
#     summary(model,(3,224,224))

def resnet50(cls_num):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3],cls_num)
    state = model.state_dict()
    loaded_state_dict = torch.load('resnet50-19c8e357.pth')
    for k in loaded_state_dict:
        if k in state:
            state[k] = loaded_state_dict[k]
    model.load_state_dict(state)
    return model

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：        num: int，the number of loss
    x: multi-task loss    Examples：
    loss1=1        loss2=2
    awl = AutomaticWeightedLoss(2)
    loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 * torch.exp(self.params[i]) * loss + self.params[i]  #
        return loss_sum
