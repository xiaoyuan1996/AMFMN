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
from .AMFMN_Modules import *
import copy

class BaseModel(nn.Module):
    def __init__(self, opt={}, vocab_words=[]):
        super(BaseModel, self).__init__()

        # img feature
        self.extract_feature = ExtractFeature(opt = opt)

        # vsa feature
        self.mvsa =VSA_Module(opt = opt)

        # text feature
        self.text_feature = Skipthoughts_Embedding_Module(
            vocab= vocab_words,
            opt = opt
        )

        self.cross_attention_s = CrossAttention(opt = opt)

        self.vgmf_gate = VGMF_Fusion(opt = opt)

        self.Eiters = 0

    def forward(self, img, text, text_lens):

        # extract features
        lower_feature, higher_feature, solo_feature = self.extract_feature(img)

        # mvsa featrues
        mvsa_feature = self.mvsa(lower_feature, higher_feature, solo_feature)

        # text features
        text_feature = self.text_feature(text)

        # VGMF
        Ft = self.cross_attention_s(mvsa_feature, text_feature)

        # sim dual path
        mvsa_feature = mvsa_feature.unsqueeze(dim=1).expand(-1, Ft.shape[1], -1)
        dual_sim = cosine_similarity(mvsa_feature, Ft)

        return dual_sim


def factory(opt, vocab_words, cuda=True, data_parallel=True):
    opt = copy.copy(opt)

    model = BaseModel(opt, vocab_words)

    if data_parallel:
        model = nn.DataParallel(model).cuda()
        if not cuda:
            raise ValueError

    if cuda:
        model.cuda()

    return model
