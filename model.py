import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


class Tripletnet(nn.Module):
    def __init__(self, embeddingnet, criterion,cls_num):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet
        self.criterion = criterion
        self.cls_num = cls_num

    def forward(self, x, y, z, c):
        """ x: Anchor image,
            y: Distant (negative) image,
            z: Close (positive) image,
            c: Integer indicating according to which attribute images are compared"""
        embedded_x, x1 = self.embeddingnet(x, c)
        embedded_y, y1 = self.embeddingnet(y, c)
        embedded_z, z1 = self.embeddingnet(z, c)
        sim_a = torch.sum(embedded_x * embedded_y, dim=1)
        sim_b = torch.sum(embedded_x * embedded_z, dim=1)
        cls = torch.eye(self.cls_num)[c,:]
        cls = cls.cuda()
        loss = (self.criterion(x1, cls) +self.criterion(y1, cls) +self.criterion(z1, cls))/3.


        return sim_a, sim_b, loss


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.LeakyReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        Mout = self.fc2(self.relu1(self.fc1(x)))
        out = self.sigmoid(Mout)
        return out



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class AG_MAN(nn.Module):
    def __init__(self, backbonenet, embedding_size, n_attributes):
        super(AG_MLAN, self).__init__()
        self.backbonenet = backbonenet
        self.n_attributes = n_attributes
        self.embedding_size = embedding_size

        self.attr_embedding = torch.nn.Embedding(n_attributes, 512)

        self.attr_transform1 = nn.Linear(512, 512)
        self.conv1 = nn.Conv2d(2048, 512, kernel_size=1, stride=1)
        self.img_bn1 = nn.BatchNorm2d(512)
        self.sa = SpatialAttention()
        self.attr_transform2 = nn.Linear(512, 512)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 512)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.w1 = torch.nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.w = torch.nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.masks = torch.nn.Embedding(n_attributes, 512)
        # initialize masks
        mask_array = np.zeros([n_attributes,512])
        mask_array.fill(0.1)
        mask_len = int(512 / n_attributes)
        for i in range(n_attributes):
            mask_array[i, i * mask_len:(i + 1) * mask_len] = 1
        # no gradients for the masks
        self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=True)

    def forward(self, x, c, norm=True):
        x, x1 = self.backbonenet(x)
        x = self.w.expand_as(x) * x
        x = self.conv1(x)
        attmap = self.ASA(x, c)

        x = x * self.w1.expand_as(attmap) * attmap
        x = self.sa(x) * x
        x = x.view(x.size(0), x.size(1), x.size(2) * x.size(3))
        x = x.sum(dim=2)  # (batch_size, 512)

        mask = self.ACA(x, c)

        x = x * self.w2.expand_as(mask) * mask
        self.mask = self.masks(c)
        self.mask = torch.nn.functional.relu(self.mask)
        x = x * self.mask
        if norm:
            x = l2norm(x)

        return x, x1

    def ASA(self, x, c):
        # attribute-aware spatial attention
        img = self.img_bn1(x)
        img = self.tanh(img)

        attr = self.attr_embedding(c)
        attr = self.attr_transform1(attr)
        attr = self.sigmoid(attr) * attr
        attr = attr.view(attr.size(0), attr.size(1), 1, 1)
        attr = attr.expand(attr.size(0), attr.size(1), 14, 14)

        attmap = attr * img
        attmap = torch.sum(attmap, dim=1, keepdim=True)
        attmap = torch.div(attmap, 512 ** 0.5)
        attmap = attmap.view(attmap.size(0), attmap.size(1), -1)
        attmap = self.softmax(attmap)
        attmap = attmap.view(attmap.size(0), attmap.size(1), 14, 14)

        return attmap

    def ACA(self, x, c):
        # attribute-aware channel attention
        attr = self.attr_embedding(c)
        attr = self.attr_transform2(attr)
        attr = self.sigmoid(attr) * attr
        img_attr = torch.cat((x, attr), dim=1)
        mask = self.fc1(img_attr)
        mask = self.relu(mask)
        mask = self.fc2(mask)
        mask = self.sigmoid(mask)

        return mask

    def get_heatmaps(self, x, c):
        x, x1 = self.backbonenet(x)
        x = self.w.expand_as(x) * x
        x = self.conv1(x)
        attmap = self.ASA(x, c)
        attmap = self.w1.expand_as(attmap) * attmap
        attmap = attmap.squeeze()

        return attmap

    
class ConditionalSimNet(nn.Module):
    def __init__(self, embeddingnet, embedding_size, n_attributes, learnedmask=True, prein=False):
        super(ConditionalSimNet, self).__init__()
        self.learnedmask = learnedmask
        self.embeddingnet = embeddingnet
        self.embed_fc = nn.Linear(1024, embedding_size)
        self.avgpool = nn.AvgPool2d(14)
        # create the mask
        if learnedmask:
            if prein:
                # define masks 
                self.masks = torch.nn.Embedding(n_attributes, embedding_size)
                # initialize masks
                mask_array = np.zeros([n_attributes, embedding_size])
                mask_array.fill(0.1)
                mask_len = int(embedding_size / n_attributes)
                for i in range(n_attributes):
                    mask_array[i, i*mask_len:(i+1)*mask_len] = 1
                # no gradients for the masks
                self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=True)
            else:
                # define masks with gradients
                self.masks = torch.nn.Embedding(n_attributes, embedding_size)
                # initialize weights
                self.masks.weight.data.normal_(0.9, 0.7) # 0.1, 0.005
        else:
            # define masks 
            self.masks = torch.nn.Embedding(n_attributes, embedding_size)
            # initialize masks
            mask_array = np.zeros([n_attributes, embedding_size])
            mask_len = int(embedding_size / n_attributes)
            for i in range(n_attributes):
                mask_array[i, i*mask_len:(i+1)*mask_len] = 1
            # no gradients for the masks
            self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=False)

    def forward(self, x, c, norm=True):
        embedded_x = self.embeddingnet(x)
        embedded_x = self.avgpool(embedded_x)
        embedded_x = embedded_x.view(embedded_x.size(0), -1)
        embedded_x = self.embed_fc(embedded_x)
        self.mask = self.masks(c)
        if self.learnedmask:
            self.mask = torch.nn.functional.relu(self.mask)
        masked_embedding = embedded_x * self.mask

        if norm:
            masked_embedding = l2norm(masked_embedding)
            
        return masked_embedding
    

model_dict = {
    'Tripletnet': Tripletnet,
    'AG_MAN': AG_MAN,
    'ConditionalSimNet': ConditionalSimNet
}
def get_model(name):
    return model_dict[name]


