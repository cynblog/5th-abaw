import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metric import concordance_correlation_coefficient
from torch.autograd import Variable


class CCCLoss(nn.Module):
    def __init__(self, digitize_num, range=[-1, 1], eps=1e-8):
        super(CCCLoss, self).__init__() 
        self.digitize_num =  digitize_num
        self.range = range
        self.eps=eps
        if self.digitize_num !=0:
            bins = np.linspace(*self.range, num= self.digitize_num)
            self.bins = Variable(torch.as_tensor(bins, dtype = torch.float32).cuda()).view((1, -1))

    def forward(self, x, y): 
        y = y.view(-1)
        if self.digitize_num !=1:
            x = F.softmax(x, dim=-1)
            x = (self.bins * x).sum(-1)
        x = x.view(-1)
        vx = x - torch.mean(x) 
        vy = y - torch.mean(y) 
        rho =  torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))+self.eps) * torch.sqrt(torch.sum(torch.pow(vy, 2))+self.eps)+self.eps)
        # print('rho:{}'.format(rho))
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x)
        y_s = torch.std(y)
        # print('x_s:{}'.format(x_s))
        # print('up:{}'.format(2*rho*x_s*y_s))
        # print('down:{}'.format((torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))))
        ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
        return 1-ccc

class MatchLoss(nn.Module):
    def __init__(self, threshold):
        super(MatchLoss, self).__init__()
        self.threshold = threshold

    def forward(self, v, a):
        # return torch.mean(torch.abs(torch.abs(v) - torch.abs(a)).mul(torch.clamp(torch.abs(v) - self.threshold, min=0)))
        return torch.mean(torch.abs(torch.abs(v) - torch.abs(a)).mul(torch.abs(v)))

if __name__ == '__main__':
    ccc = CCCLoss(1)
    a = torch.Tensor([1, 2, 3, 4, 5])
    b = torch.Tensor([1, 3, 2, 1, 1])
    print(torch.std(a))
    print(np.std(np.array(a)))
    # loss = ccc(a, b)

    # ccc = concordance_correlation_coefficient(np.array(a), np.array(b))
    # print(1 - loss)
    # print(ccc)

class CELoss(nn.Module):
    def __init__(self, digitize_num, range=[-1, 1], weights=None):
        super(CELoss, self).__init__() 
        self.digitize_num = digitize_num
        if weights is not None:
            self.weights = torch.Tensor(weights).cuda()
        else:
            self.weights = None
        assert self.digitize_num !=1
        self.edges = np.linspace(*range, num= self.digitize_num+1)
    
    def forward(self, x, y): 
        y = y.view(-1)
        y_numpy = y.data.cpu().numpy()
        y_dig = np.digitize(y_numpy, self.edges) - 1
        y_dig[y_dig==self.digitize_num] = self.digitize_num -1
        y = Variable(torch.cuda.LongTensor(y_dig))
        return F.cross_entropy(x, y, weight=self.weights)

class CCC_CE_Loss(nn.Module):
    def __init__(self, digitize_num, range=[-1, 1], alpha=0.5, beta=0.5):
        super(CCC_CE_Loss, self).__init__()
        self.ccc_loss = CCCLoss(digitize_num, range=range)
        self.ce_loss = CELoss(digitize_num, range=range)
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, y):
        cccl = self.ccc_loss(x, y)
        cel = self.ce_loss(x, y)
        # print(cccl, cel)
        loss = self.alpha*cccl + self.beta*cel
        return loss