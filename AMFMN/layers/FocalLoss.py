# -----------------------------------------------------------
# "Exploring a Fine-Grained Multiscale Method for Cross-Modal Remote Sensing Image Retrieval"
# Yuan, Zhiqiang and Zhang, Wenkai and Fu, Kun and Li, Xuan and Deng, Chubo and Wang, Hongqi and Sun, Xian
# IEEE Transactions on Geoscience and Remote Sensing 2021
# Writen by YuanZhiqiang, 2021.  Our code is depended on MTFN
# ------------------------------------------------------------
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

def label_smoothing(inputs, epsilon=0.1):
    K = inputs.shape[-1]
    return ((1-epsilon) * inputs) + (epsilon / K)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True, class_num=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.class_name = class_num

    def forward(self, inputs, targets):
#        target = []
#        for t in targets.cpu().numpy():
#            tmp = [0]*self.class_name
#            tmp[t] = 1
#            target.append(tmp)
#        targets = torch.Tensor(target).cuda()
        targets = label_smoothing(targets)

        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

if __name__ == "__main__":
    from torch.autograd import Variable
    import numpy as np
    input_demo = np.array([0, 1, 0, 0])
    input_demo = Variable(torch.from_numpy(input_demo))
    print(input_demo)

    output_demo = label_smoothing(input_demo, epsilon=0.1)
    print(output_demo)
