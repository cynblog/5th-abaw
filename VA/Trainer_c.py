import torch
import math
import random
import glob
import os
import timm
import time

import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np

from config import cfg
from metric import concordance_correlation_coefficient
from loss import CCCLoss, MatchLoss
from torch.autograd import Variable
from tqdm import tqdm
from utils import get_model_list, get_stat_pred
from collections import Counter
from network import Classifier, Classifier_V, Classifier_A, Classifier_VA, Classifier_P
from sklearn.metrics import accuracy_score, f1_score


class Trainer(nn.Module):
    def __init__(self, model_name, mode=None):
        super(Trainer, self).__init__()
        self.model_name = model_name
        self.mode = mode
        self.build_net()
        # self.optimizer_model = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)
        # self.optimizer_v = optim.SGD(self.vhead.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        # self.optimizer_a = optim.SGD(self.ahead.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.ce = nn.CrossEntropyLoss()
        self.criterion = CCCLoss(1)
        self.matchloss = MatchLoss(0.2)
        self.list_init()
    
    def build_net(self):
        if self.model_name == 'resnet50':
            print('构建resnet50模型')
            self.model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
            # self.vhead = Classifier(1000).cuda()
            # self.ahead = Classifier(1000).cuda()
            self.vahead = Classifier_VA(1000, 1000)
            model_params = list(self.model.parameters())
            self.optimizer_model = torch.optim.Adam([p for p in model_params if p.requires_grad],
                                        lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)
        elif self.model_name == 'enet':
            print('构建enet模型')
            self.model_path = r"/amax/cvpr23_competition/challenge_4/code_v1/save_model_dir/dir_2/enet_b2_8_best.pt"
            self.model = torch.load(self.model_path)
            self.model.classifier = torch.nn.Identity()
            for k, v in self.model.named_parameters():
                v.requires_grad = True
            # self.vhead = Classifier(1408).cuda()
            # self.ahead = Classifier(1408).cuda()
            self.vahead = Classifier_VA(1408, 1408)
            model_params = list(self.model.parameters())
            self.optimizer_model = torch.optim.Adam([p for p in model_params if p.requires_grad],
                                        lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)
        elif self.model_name == 'resnet_enet':
            print('构建resnet与enet模型')
            self.resnet = timm.create_model('resnet50')
            self.resnet.load_state_dict(torch.load('/data02/mxy/CVPR_ABAW/precode/output/pretrain/resnet50-19c8e357.pth'))
            self.enet = torch.load(r"/amax/cvpr23_competition/challenge_4/code_v1/save_model_dir/dir_2/enet_b2_8_best.pt")
            self.enet.classifier = torch.nn.Identity()
            for k, v in self.enet.named_parameters():
                v.requires_grad = True
            self.head = Classifier(2408, mode='p').cuda()
            model_params = list(self.resnet.parameters()) + list(self.enet.parameters())
            self.optimizer_model = torch.optim.Adam([p for p in model_params if p.requires_grad],
                                        lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)
        params = list(self.head.parameters())
        self.optimizer_h = torch.optim.Adam([p for p in params if p.requires_grad],
                                        lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)

    def list_init(self):
        self.pred = list()
        self.label = list()

    def train(self, images, labels):
        self.head.train()
        # self.ahead.train()
        if self.model_name == 'resnet_enet':
            self.resnet.train()
            self.enet.train()
            features = torch.cat([self.resnet(images), self.enet(images)], dim=1)
        else:
            self.model.train()
            features = self.model(images)
        # v = torch.tanh(self.vhead(features))
        # a = torch.tanh(self.ahead(features))
        pred = self.head(features)
        # a = self.ahead(features)

        loss = self.criterion(pred, labels)

        self.optimizer_model.zero_grad()
        self.optimizer_h.zero_grad()
        # self.optimizer_a.zero_grad()
        loss.backward()
        self.optimizer_model.step()
        self.optimizer_h.step()
        # self.optimizer_a.step()

        pred = pred.squeeze()

        self.pred.extend(pred.detach().cpu().tolist())
        self.label.extend(labels.detach().cpu().tolist())

        return loss

    @torch.no_grad()
    def val(self, images, labels):
        self.head.eval()
        # self.ahead.eval()
        if self.model_name == 'resnet_enet':
            self.resnet.eval()
            self.enet.eval()
            features = torch.cat([self.resnet(images), self.enet(images)], dim=1)
        else:
            self.model.eval()
            features = self.model(images)
        pred = self.head(features)
        # a = self.ahead(features)

        loss = self.criterion(pred, labels)

        pred = pred.squeeze()

        self.pred.extend(pred.detach().cpu().tolist())
        self.label.extend(labels.detach().cpu().tolist())

        return loss

    def CCC(self):
        ccc = concordance_correlation_coefficient(self.label, self.pred)
        self.list_init()
        return ccc
    
    def metric_p(self):
        pred_pv, label_pv, pred_pa, label_pa = torch.cat(self.pred_pv).cpu(), torch.cat(self.label_pv).cpu(), torch.cat(self.pred_pa).cpu(), torch.cat(self.label_pa).cpu()
        pred_pv = pred_pv.argmax(axis=1).cpu()
        pred_pa = pred_pa.argmax(axis=1).cpu()
        acc_v = accuracy_score(pred_pv, label_pv)
        # f1_score = F1Score(num_classes=6)
        f1_v = f1_score(pred_pv, label_pv, average='macro')
        acc_a = accuracy_score(pred_pa, label_pa)
        # f1_score = F1Score(num_classes=6)
        f1_a = f1_score(pred_pa, label_pa, average='macro')
        return acc_v, f1_v, acc_a, f1_a
    
    def save(self, save_path, iterations, ccc):
        save_name = os.path.join(save_path, self.model_name + '_%03d_%08d.pt' % (int(ccc*1000), iterations + 1))
        opt_name = os.path.join(save_path, 'optimizer.pt')
        if self.model_name == 'resnet_enet':
            torch.save({'resnet': self.resnet.state_dict(), 'enet': self.enet.state_dict(), 'h': self.head.state_dict()}, save_name)
        else:
            torch.save({'model': self.model.state_dict(), 'h': self.head.state_dict()}, save_name)
        torch.save({'model': self.optimizer_model.state_dict(), 'h': self.optimizer_h.state_dict()}, opt_name)

    def resume(self, resume_path):
        last_model_name, iterations = get_model_list(resume_path, self.model_name)
        print('正在加载参数：', last_model_name)
        state_dict = torch.load(last_model_name)
        if self.model_name == 'resnet_enet':
            self.resnet.load_state_dict(state_dict['resnet'])
            self.enet.load_state_dict(state_dict['enet'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.head.load_state_dict(state_dict['h'])
        # self.ahead.load_state_dict(state_dict['a'])

        state_dict = torch.load(os.path.join(resume_path, 'optimizer.pt'))
        self.optimizer_model.load_state_dict(state_dict['model'])
        self.optimizer_h.load_state_dict(state_dict['h'])
        # self.optimizer_a.load_state_dict(state_dict['a'])
        
        return iterations

    def get_pred_bar(self):
        get_stat_pred(self.pred, self.label, '正性v')

if __name__ == '__main__':
    trainer = Trainer()
    a = [1,2]
    b = [1]
    print(a.extend(b))
    print(a)