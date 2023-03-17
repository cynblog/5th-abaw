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

torch.autograd.set_detect_anomaly(True)
class Trainer(nn.Module):
    def __init__(self, model_name, phase=None, mode=None):
        super(Trainer, self).__init__()
        self.model_name = model_name
        self.mode = mode
        if phase == 'test':
            self.build_net_test()
        else:
            self.build_net()
        # self.optimizer_model = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)
        # self.optimizer_v = optim.SGD(self.vhead.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        # self.optimizer_a = optim.SGD(self.ahead.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.ce = nn.CrossEntropyLoss()
        self.criterion = CCCLoss(1)
        self.matchloss = MatchLoss(0.2)
        self.mse = nn.MSELoss()
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
        elif self.model_name == 'regnet':
            print('构建regnet模型')
            model_path = '/data02/mxy/CVPR_ABAW/precode/output/origin/regnety_040_ra3-670e1166.pth'
            self.model = timm.create_model('regnety_040', pretrained=False)
            self.model.load_state_dict(torch.load(model_path))
            for k, v in self.model.named_parameters():
                v.requires_grad = True
            self.vahead = Classifier_VA(1000, 1000)
            model_params = list(self.model.parameters())
            self.optimizer_model = torch.optim.Adam([p for p in model_params if p.requires_grad],
                                        lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)
        elif self.model_name == 'resnet_enet_regnet':
            print('构建resnet,enet,regnet模型')
            model_path = '/data02/mxy/CVPR_ABAW/precode/output/origin/regnety_040_ra3-670e1166.pth'
            self.regnet = timm.create_model('regnety_040', pretrained=False)
            self.regnet.load_state_dict(torch.load(model_path))
            for k, v in self.regnet.named_parameters():
                v.requires_grad = True
            self.resnet = timm.create_model('resnet50')
            self.resnet.load_state_dict(torch.load('/data02/mxy/CVPR_ABAW/precode/output/pretrain/resnet50-19c8e357.pth'))
            self.enet = torch.load(r"/amax/cvpr23_competition/pretrain_model/efficientnet_affectnet/enet_b2_8_best.pt")
            self.enet.classifier = torch.nn.Identity()
            for k, v in self.enet.named_parameters():
                v.requires_grad = True
            for k, v in self.resnet.named_parameters():
                v.requires_grad = True
            self.vahead = Classifier_VA(3408, 3408)
            model_params = list(self.resnet.parameters()) + list(self.enet.parameters())
            self.optimizer_model = torch.optim.Adam([p for p in model_params if p.requires_grad],
                                        lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)
        elif self.model_name == 'enet':
            print('构建enet模型')
            self.model_path = r"/amax/cvpr23_competition/pretrain_model/efficientnet_affectnet/enet_b2_8_best.pt"
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
            self.enet = torch.load(r"/amax/cvpr23_competition/pretrain_model/efficientnet_affectnet/enet_b2_8_best.pt")
            self.enet.classifier = torch.nn.Identity()
            for k, v in self.enet.named_parameters():
                v.requires_grad = True
            for k, v in self.resnet.named_parameters():
                v.requires_grad = True
            # self.vhead = Classifier(2408).cuda()
            # self.ahead = Classifier(2408).cuda()
            self.vahead = Classifier_VA(2408, 2408)
            model_params = list(self.resnet.parameters()) + list(self.enet.parameters())
            self.optimizer_model = torch.optim.Adam([p for p in model_params if p.requires_grad],
                                        lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)
            # self.optimizer_model = None
        if self.mode == 'help':
            # self.vhead = Classifier_V(2408).cuda()
            # self.ahead = Classifier_A(2408, 512).cuda()
            print('构建互助流')
            self.vahead = Classifier_VA(2408, 2408, 512, mode='help')
        elif self.mode == 'p':
            print('构建极性分类器')
            self.pvhead = Classifier_P(1408)
            self.pahead = Classifier_P(1408)
            params = list(self.pvhead.parameters()) + list(self.pahead.parameters())
            self.optimizer_p = torch.optim.Adam([p for p in params if p.requires_grad],
                                    lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)
        
        params = list(self.vahead.parameters())
        self.optimizer_va = torch.optim.Adam([p for p in params if p.requires_grad],
                                    lr=0.01, betas=(0.5, 0.999), weight_decay=0.0001)
        # else:
        # v_params = list(self.vhead.parameters())
        # a_params = list(self.ahead.parameters())
        # self.optimizer_v = torch.optim.Adam([p for p in v_params if p.requires_grad],
        #                                 lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)
        # self.optimizer_a = torch.optim.Adam([p for p in a_params if p.requires_grad],
        #                                 lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)

    def build_net_test(self):
        if self.model_name == 'resnet_enet':
            print('构建resnet与enet模型')
            self.resnet = timm.create_model('resnet50')
            self.enet = torch.load(r"/amax/cvpr23_competition/pretrain_model/efficientnet_affectnet/enet_b2_8_best.pt")
            self.enet.classifier = torch.nn.Identity()
            self.vhead = Classifier(2408).cuda()
            self.ahead = Classifier(2408).cuda()

    def list_init(self):
        self.pred_v = list()
        self.label_v = list()
        self.pred_a = list()
        self.label_a = list()
        self.pred_pv = list()
        self.pred_pa = list()
        self.label_pv = list()
        self.label_pa = list()
        self.images = list()
        self.dic = dict()

    @torch.no_grad()
    def get_feature(self, images):
        if self.model_name == 'resnet_enet':
            self.resnet.train()
            self.enet.train()
            features = torch.cat([self.resnet(images), self.enet(images)], dim=1)
        elif self.model_name == 'resnet_enet_regnet':
            self.resnet.train()
            self.enet.train()
            self.regnet.train()
            features = torch.cat([self.resnet(images), self.enet(images), self.regnet(images)], dim=1)
        else:
            self.model.train()
            features = self.model(images)
        return features

    def train(self, images, v_labels, a_labels):
        self.vahead.train()
        # self.ahead.train()
        # features = self.get_feature(images)
        if self.model_name == 'resnet_enet':
            self.resnet.train()
            self.enet.train()
            features = torch.cat([self.resnet(images), self.enet(images)], dim=1)
        elif self.model_name == 'resnet_enet_regnet':
            self.resnet.train()
            self.enet.train()
            self.regnet.train()
            features = torch.cat([self.resnet(images), self.enet(images), self.regnet(images)], dim=1)
        else:
            self.model.train()
            features = self.model(images)
        # v = torch.tanh(self.vhead(features))
        # a = torch.tanh(self.ahead(features))
        v, a = self.vahead(features)
        # a = self.ahead(features)

        v_loss = self.criterion(v, v_labels)
        a_loss = self.criterion(a, a_labels)
        # v_mse = self.mse(v.squeeze(), v_labels)
        # a_mse = self.mse(a.squeeze(), a_labels)
        match_loss = self.matchloss(v, a) 
        final_loss = a_loss + v_loss + 0.5 * match_loss
        # print(v.dtype)
        # print(v_labels.dtype)
        # print(type(v_loss.item()))
        # print(type(match_loss.item()))
        # print(type(v_mse.item()))
        # print(type(a_mse.item()))
        # exit()
        self.optimizer_model.zero_grad()
        self.optimizer_va.zero_grad()
        # self.optimizer_a.zero_grad()
        # with torch.autograd.detect_anomaly():
        final_loss.backward()
        self.optimizer_model.step()
        self.optimizer_va.step()
        # self.optimizer_a.step()

        v = v.squeeze()
        a = a.squeeze()

        self.pred_v.extend(v.detach().cpu().tolist())
        self.label_v.extend(v_labels.detach().cpu().tolist())
        self.pred_a.extend(a.detach().cpu().tolist())
        self.label_a.extend(a_labels.detach().cpu().tolist())

        return final_loss.item(), v_loss.item(), a_loss.item(), match_loss

    def train_p(self, images, v_labels, a_labels):
        s_time = time.time()
        if self.model_name == 'resnet_enet':
            self.resnet.train()
            self.enet.train()
            features = torch.cat([self.resnet(images), self.enet(images)], dim=1)
        else:
            self.model.train()
            features = self.model(images)
        self.pvhead.train()
        self.pahead.train()
        # print('前向传播时间：{}'.format(time.time()-s_time))
        pred_pv = self.pvhead(features)
        pred_pa = self.pahead(features)
        loss_pv = self.ce(pred_pv, v_labels)
        loss_pa = self.ce(pred_pa, a_labels)
        final_loss = 2 * loss_pv + 1 * loss_pa
        s_time = time.time()
        self.optimizer_model.zero_grad()
        self.optimizer_p.zero_grad()
        # self.optimizer_a.zero_grad()
        final_loss.backward()
        self.optimizer_model.step()
        self.optimizer_p.step()
        # print('反向传播时间：{}'.format(time.time()-s_time))
        s_time = time.time()
        self.pred_pv.append(pred_pv)
        self.pred_pa.append(pred_pa)
        self.label_pv.append(v_labels)
        self.label_pa.append(a_labels)
        # print('标签存储时间：{}'.format(time.time()-s_time))
        return final_loss.item(), loss_pv.item(), loss_pa.item()

    @torch.no_grad()
    def val_p(self, images, v_labels, a_labels):
        self.vahead.eval()
        # self.ahead.eval()
        if self.model_name == 'resnet_enet':
            self.resnet.eval()
            self.enet.eval()
            features = torch.cat([self.resnet(images), self.enet(images)], dim=1)
        else:
            self.model.eval()
            features = self.model(images)
        pred_pv = self.pvhead(features)
        pred_pa = self.pahead(features)
        self.pred_pv.append(pred_pv)
        self.pred_pa.append(pred_pa)
        self.label_pv.append(v_labels)
        self.label_pa.append(a_labels)

    @torch.no_grad()
    def val(self, images, images_name, v_labels, a_labels):
        self.vahead.eval()
        # self.vhead.eval()
        # self.ahead.eval()
        if self.model_name == 'resnet_enet':
            self.resnet.eval()
            self.enet.eval()
            features = torch.cat([self.resnet(images), self.enet(images)], dim=1)
        elif self.model_name == 'resnet_enet_regnet':
            self.resnet.eval()
            self.enet.eval()
            self.regnet.eval()
            features = torch.cat([self.resnet(images), self.enet(images), self.regnet(images)], dim=1)
        else:
            self.model.eval()
            features = self.model(images)
        v, a = self.vahead(features)
        # v = self.vhead(features)
        # a = self.ahead(features)

        v_loss = self.criterion(v, v_labels)
        a_loss = self.criterion(a, a_labels)
        match_loss = self.matchloss(v, a)
        v = v.squeeze()
        a = a.squeeze()

        self.pred_v.extend(v.detach().cpu().tolist())
        self.label_v.extend(v_labels.detach().cpu().tolist())
        self.pred_a.extend(a.detach().cpu().tolist())
        self.label_a.extend(a_labels.detach().cpu().tolist())
        # self.images.extend(images_name)
        return v_loss.item(), a_loss.item(), match_loss.item()

    @torch.no_grad()
    def val_v(self, images, v_labels, a_labels):
        self.model.eval()
        self.vhead.eval()
        features = self.model(images)
        v = torch.clamp(self.vhead(features), min=-1, max=1)

        v_loss = self.criterion(v, v_labels)
        v = v.squeeze()

        self.pred_v.extend(v.detach().cpu().tolist())
        self.label_v.extend(v_labels.detach().cpu().tolist())

        return v_loss.item(), 0, 0
 
    @torch.no_grad()
    def test(self, images, images_name):
        # self.vahead.eval()
        self.vhead.eval()
        self.ahead.eval()
        if self.model_name == 'resnet_enet':
            self.resnet.eval()
            self.enet.eval()
            features = torch.cat([self.resnet(images), self.enet(images)], dim=1)
        elif self.model_name == 'resnet_enet_regnet':
            self.resnet.eval()
            self.enet.eval()
            self.regnet.eval()
            features = torch.cat([self.resnet(images), self.enet(images), self.regnet(images)], dim=1)
        else:
            self.model.eval()
            features = self.model(images)
        # v, a = self.vahead(features)
        v = self.vhead(features)    
        a = self.ahead(features)
        v = v.squeeze()
        a = a.squeeze()
        self.pred_v.extend(v.detach().cpu().tolist())
        # print(self.pred_v)
        # exit()
        self.pred_a.extend(a.detach().cpu().tolist())
        self.images.extend(images_name)

    def write_results(self, result_path):
        # print(self.pred_v[0])
        for index, image_name in enumerate(self.images):
            with open(result_path, 'a') as f:
                f.write(image_name + ',' + str(self.pred_v[index]) + ',' + str(self.pred_a[index]) + '\n')
                # if index == 100:
                #     exit()


    def CCC(self):

        v_ccc = concordance_correlation_coefficient(self.label_v, self.pred_v)
        a_ccc = concordance_correlation_coefficient(self.label_a, self.pred_a)

        self.list_init()
        return v_ccc, a_ccc

    def CCC_v(self):
        v_ccc = concordance_correlation_coefficient(self.label_v, self,pred_v)
        self.list_init()
        return v_ccc, 0
    
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
        self.list_init()
        return acc_v, f1_v, acc_a, f1_a

    def get_pred_bar(self):
        pred_pv, label_pv, pred_pa, label_pa = torch.cat(self.pred_pv).cpu(), torch.cat(self.label_pv).cpu(), torch.cat(self.pred_pa).cpu(), torch.cat(self.label_pa).cpu()
        
        pred_pv = pred_pv.argmax(axis=1).cpu()
        pred_pa = pred_pa.argmax(axis=1).cpu()
        # print(pred_pv)
        get_stat_pred(pred_pv, label_pv, 'pv')
        get_stat_pred(pred_pa, label_pa, 'pa')
    
    def save_old(self, save_path, iterations, v_ccc, a_ccc):
        save_name = os.path.join(save_path, self.model_name + '_%03d_%03d_%08d.pt' % (int(v_ccc*1000), int(a_ccc*1000), iterations + 1))
        opt_name = os.path.join(save_path, 'optimizer.pt')
        if self.model_name == 'resnet_enet':
            torch.save({'resnet': self.resnet.state_dict(), 'enet': self.enet.state_dict(), 'v': self.vhead.state_dict(), 'a': self.ahead.state_dict()}, save_name)
        else:
            torch.save({'model': self.model.state_dict(), 'v': self.vhead.state_dict(), 'a': self.ahead.state_dict()}, save_name)
        torch.save({'model': self.optimizer_model.state_dict(), 'v': self.optimizer_v.state_dict(), 'a': self.optimizer_a.state_dict()}, opt_name)
    
    def save(self, save_path, iterations, v_ccc, a_ccc):
        save_name = os.path.join(save_path, self.model_name + '_%03d_%03d_%08d.pt' % (int(v_ccc*1000), int(a_ccc*1000), iterations + 1))
        opt_name = os.path.join(save_path, 'optimizer.pt')
        if self.model_name == 'resnet_enet':
            torch.save({'resnet': self.resnet.state_dict(), 'enet': self.enet.state_dict(), 'va': self.vahead.state_dict()}, save_name)
        elif self.model_name == 'resnet_enet_regnet':
            torch.save({'resnet': self.resnet.state_dict(), 'enet': self.enet.state_dict(), 'regnet': self.enet.state_dict(), 'va': self.vahead.state_dict()}, save_name)
        else:
            torch.save({'model': self.model.state_dict(), 'va': self.vahead.state_dict()}, save_name)
        torch.save({'model': self.optimizer_model.state_dict(), 'va': self.optimizer_va.state_dict()}, opt_name)
    
    def save_p(self, save_path, iterations, acc_v, acc_a):
        save_name = os.path.join(save_path, self.model_name + '_%03d_%03d_%08d.pt' % (int(acc_v*1000), int(acc_a*1000), iterations + 1))
        opt_name = os.path.join(save_path, 'optimizer.pt')
        if self.model_name == 'resnet_enet':
            torch.save({'resnet': self.resnet.state_dict(), 'enet': self.enet.state_dict(), 'pv': self.pvhead.state_dict(), 'pa': self.pahead.state_dict()}, save_name)
        else:
            torch.save({'model': self.model.state_dict(), 'pv': self.pvhead.state_dict(), 'pa': self.pahead.state_dict()}, save_name)
        torch.save({'model': self.optimizer_model.state_dict(), 'p': self.optimizer_p.state_dict()}, opt_name)

    def resume_p(self, resume_path):
        last_model_name, iterations = get_model_list(resume_path, self.model_name)
        print('正在加载参数：', last_model_name)
        state_dict = torch.load(last_model_name)
        if self.model_name == 'resnet_enet':
            self.resnet.load_state_dict(state_dict['resnet'])
            self.enet.load_state_dict(state_dict['enet'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.pvhead.load_state_dict(state_dict['pv'])
        self.pahead.load_state_dict(state_dict['pa'])
        state_dict = torch.load(os.path.join(resume_path, 'optimizer.pt'))
        self.optimizer_model.load_state_dict(state_dict['model'])
        self.optimizer_p.load_state_dict(state_dict['p'])
        return iterations

    def resume_old(self, resume_path):
        last_model_name, iterations = get_model_list(resume_path, self.model_name)
        print('正在加载参数：', last_model_name)
        state_dict = torch.load(last_model_name)
        if self.model_name == 'resnet_enet':
            self.resnet.load_state_dict(state_dict['resnet'])
            self.enet.load_state_dict(state_dict['enet'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.vhead.load_state_dict(state_dict['v'])
        self.ahead.load_state_dict(state_dict['a'])

        state_dict = torch.load(os.path.join(resume_path, 'optimizer.pt'))
        self.optimizer_model.load_state_dict(state_dict['model'])
        self.optimizer_v.load_state_dict(state_dict['v'])
        self.optimizer_a.load_state_dict(state_dict['a'])
        
        return iterations

    def resume(self, resume_path):
        last_model_name, iterations = get_model_list(resume_path, self.model_name)
        print('正在加载参数：', last_model_name)
        state_dict = torch.load(last_model_name)
        if self.model_name == 'resnet_enet':
            self.resnet.load_state_dict(state_dict['resnet'])
            self.enet.load_state_dict(state_dict['enet'])
        elif self.model_name == 'resnet_enet_regnet':
            self.resnet.load_state_dict(state_dict['resnet'])
            self.enet.load_state_dict(state_dict['enet'])
            self.regnet.load_state_dict(state_dict['regnet'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.vahead.load_state_dict(state_dict['va'])
        # self.ahead.load_state_dict(state_dict['a'])

        state_dict = torch.load(os.path.join(resume_path, 'optimizer.pt'))
        self.optimizer_model.load_state_dict(state_dict['model'])
        self.optimizer_va.load_state_dict(state_dict['va'])
        # self.optimizer_a.load_state_dict(state_dict['a'])
        
        return iterations
    def resume_test(self, resume_path):
        last_model_name, iterations = get_model_list(resume_path, self.model_name)
        print('正在加载参数：', last_model_name)
        state_dict = torch.load(last_model_name)
        if self.model_name == 'resnet_enet':
            self.resnet.load_state_dict(state_dict['resnet'])
            self.enet.load_state_dict(state_dict['enet'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.vhead.load_state_dict(state_dict['v'])
        self.ahead.load_state_dict(state_dict['a'])

if __name__ == '__main__':
    trainer = Trainer()
    a = [1,2]
    b = [1]
    print(a.extend(b))
    print(a)