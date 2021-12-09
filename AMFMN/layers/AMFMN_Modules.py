# -----------------------------------------------------------
# "Exploring a Fine-Grained Multiscale Method for Cross-Modal Remote Sensing Image Retrieval"
# Yuan, Zhiqiang and Zhang, Wenkai and Fu, Kun and Li, Xuan and Deng, Chubo and Wang, Hongqi and Sun, Xian
# IEEE Transactions on Geoscience and Remote Sensing 2021
# Writen by YuanZhiqiang, 2021.  Our code is depended on MTFN
# ------------------------------------------------------------
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from torchvision.models.resnet import resnet18
import torch.nn.functional as F
from layers import seq2vec
import math
import copy

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class ExtractFeature(nn.Module):
    def __init__(self, opt = {}, finetune=True):
        super(ExtractFeature, self).__init__()

        self.embed_dim = opt['embed']['embed_dim']

        self.resnet = resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = finetune

        self.pool_2x2 = nn.MaxPool2d(4)

        self.up_sample_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_sample_4 = nn.Upsample(scale_factor=4, mode='nearest')

        self.linear = nn.Linear(in_features=512, out_features=self.embed_dim)

    def forward(self, img):
        x = self.resnet.conv1(img)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        f1 = self.resnet.layer1(x)
        f2 = self.resnet.layer2(f1)
        f3 = self.resnet.layer3(f2)
        f4 = self.resnet.layer4(f3)

        # Lower Feature
        f2_up = self.up_sample_2(f2)
        lower_feature = torch.cat([f1, f2_up], dim=1)

        # Higher Feature
        f4_up = self.up_sample_2(f4)
        higher_feature = torch.cat([f3, f4_up], dim=1)
        # higher_feature = self.up_sample_4(higher_feature)

        # batch * 512
        feature = f4.view(f4.shape[0], 512, -1)
        solo_feature = self.linear(torch.mean(feature,dim=-1))

        # torch.Size([10, 192, 64, 64])
        # torch.Size([10, 768, 64, 64])
        # torch.Size([10, 512])
        return lower_feature, higher_feature, solo_feature

class VSA_Module(nn.Module):
    def __init__(self, opt = {}):
        super(VSA_Module, self).__init__()

        # extract value
        channel_size = opt['multiscale']['multiscale_input_channel']
        out_channels = opt['multiscale']['multiscale_output_channel']
        embed_dim = opt['embed']['embed_dim']

        # sub sample
        self.LF_conv = nn.Conv2d(in_channels=192, out_channels=channel_size, kernel_size=3, stride=4)
        self.HF_conv = nn.Conv2d(in_channels=768, out_channels=channel_size, kernel_size=1, stride=1)

        # visual attention
        self.conv1x1_1 = nn.Conv2d(in_channels=channel_size*2, out_channels=out_channels, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(in_channels=channel_size*2, out_channels=out_channels, kernel_size=1)

        # solo attention
        self.solo_attention = nn.Linear(in_features=256, out_features=embed_dim)

    def forward(self, lower_feature, higher_feature, solo_feature):

        # b x channel_size x 16 x 16
        lower_feature = self.LF_conv(lower_feature)
        higher_feature = self.HF_conv(higher_feature)

        # concat
        concat_feature = torch.cat([lower_feature, higher_feature], dim=1)

        # residual
        concat_feature = higher_feature.mean(dim=1,keepdim=True).expand_as(concat_feature) + concat_feature

        # attention
        main_feature = self.conv1x1_1(concat_feature)
        attn_feature = torch.sigmoid(self.conv1x1_2(concat_feature).view(concat_feature.shape[0],1,-1)).view(concat_feature.shape[0], 1, main_feature.shape[2], main_feature.shape[3])
        atted_feature = (main_feature*attn_feature).squeeze(dim=1).view(attn_feature.shape[0], -1)

       # solo attention
        solo_att = torch.sigmoid(self.solo_attention(atted_feature))
        solo_feature = solo_feature*solo_att

        return solo_feature

class Skipthoughts_Embedding_Module(nn.Module):
    def __init__(self, vocab, opt, out_dropout=-1):
        super(Skipthoughts_Embedding_Module, self).__init__()
        self.opt = opt
        self.vocab_words = vocab

        self.seq2vec = seq2vec.factory(self.vocab_words, self.opt['seq2vec'], self.opt['seq2vec']['dropout'])

        self.to_out = nn.Linear(in_features=2400, out_features=self.opt['embed']['embed_dim'])
        self.dropout = out_dropout

    def forward(self, input_text ):
        x_t_vec = self.seq2vec(input_text)
        out = F.relu(self.to_out(x_t_vec))
        if self.dropout >= 0:
            out = F.dropout(out, self.dropout)

        return out 

def cosine_similarity(x1, x2, dim=-1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

# cross attention
class CrossAttention(nn.Module):

    def __init__(self, opt={}):
        super(CrossAttention, self).__init__()

        self.att_type = opt['cross_attention']['att_type']
        dim = opt['embed']['embed_dim']

        if self.att_type == "soft_att":
            self.cross_attention = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Sigmoid()
            )
        elif self.att_type == "fusion_att":
            self.cross_attention_fc1 = nn.Sequential(
                nn.Linear(2*dim, dim),
                nn.Sigmoid()
            )
            self.cross_attention_fc2 = nn.Sequential(
                nn.Linear(2*dim, dim),
            )
            self.cross_attention = lambda x:self.cross_attention_fc1(x)*self.cross_attention_fc2(x)

        elif self.att_type == "similarity_att":
            self.fc_visual = nn.Sequential(
                nn.Linear(dim, dim),
            )
            self.fc_text = nn.Sequential(
                nn.Linear(dim, dim),
            )
        else:
            raise Exception

    def forward(self, visual, text):
        batch_v = visual.shape[0]
        batch_t = text.shape[0]

        if self.att_type == "soft_att":
            visual_gate = self.cross_attention(visual)

            # mm
            visual_gate = visual_gate.unsqueeze(dim=1).expand(-1, batch_t, -1)
            text = text.unsqueeze(dim=0).expand(batch_v, -1, -1)

            return visual_gate*text

        elif self.att_type == "fusion_att":
            visual = visual.unsqueeze(dim=1).expand(-1, batch_t, -1)
            text = text.unsqueeze(dim=0).expand(batch_v, -1, -1)

            fusion_vec = torch.cat([visual,text], dim=-1)

            return self.cross_attention(fusion_vec)
        elif self.att_type == "similarity_att":
            visual = self.fc_visual(visual)
            text = self.fc_text(text)

            visual = visual.unsqueeze(dim=1).expand(-1, batch_t, -1)
            text = text.unsqueeze(dim=0).expand(batch_v, -1, -1)

            sims = visual*text
            return F.sigmoid(sims) * text

class VGMF_Fusion(nn.Module):
    def __init__(self, opt = {}):
        super(VGMF_Fusion, self).__init__()
        self.gate = nn.Linear(1024, opt['embed']['embed_dim'])

    def forward(self, sv, kv):
        # l2 norm
        sv = l2norm(sv, dim=-1)
        kv = l2norm(kv, dim=-1)

        # concat fc
        sw_s = F.sigmoid(self.gate(torch.cat([sv, kv], dim=-1)))
        ones = torch.ones(sw_s.shape).cuda()
        sw_k = ones - sw_s

        out = sw_s*sv + sw_k*kv
        return out