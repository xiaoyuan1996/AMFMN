# -----------------------------------------------------------
# "Exploring a Fine-Grained Multiscale Method for Cross-Modal Remote Sensing Image Retrieval"
# Yuan, Zhiqiang and Zhang, Wenkai and Fu, Kun and Li, Xuan and Deng, Chubo and Wang, Hongqi and Sun, Xian
# IEEE Transactions on Geoscience and Remote Sensing 2021
# Writen by YuanZhiqiang, 2021.  Our code is depended on MTFN
# ------------------------------------------------------------

# A revision version from Skip-thoughs
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import skipthoughts
from skipthoughts import BayesianUniSkip

def factory(vocab_words, opt , dropout=0.25):
    if opt['arch'] == 'skipthoughts':
        st_class = getattr(skipthoughts, opt['type'])
        seq2vec = st_class(opt['dir_st'],
                           vocab_words,
                           dropout=dropout,
                           fixed_emb=opt['fixed_emb'])

    else:
        raise NotImplementedError
    return seq2vec
