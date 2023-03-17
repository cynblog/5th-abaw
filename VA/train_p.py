import torch
import math
import random
import argparse
import os
import sys
import signal
import time

import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data

from dataset import dataset
from config import cfg
from metric import concordance_correlation_coefficient
from loss import CCCLoss
from torch.autograd import Variable
from tqdm import tqdm
from Trainer import Trainer
from tensorboardX import SummaryWriter


# 参数配置
parser = argparse.ArgumentParser(
    description='VA Establish With Pytorch')
parser.add_argument('--resume',
                    default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--save_path',
                    default="./output/enet_p/21", type=str,
                    help='path for saving checkpoint models')

args = parser.parse_args()

# 创建存储路径
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print('创建存储路径')

log_path= os.path.join(args.save_path, 'log')
if not os.path.exists(log_path):
    os.makedirs(log_path)
    print('创建日志路径')

summary_writer = SummaryWriter(log_path)

# 加载模型
trainer = Trainer('enet', 'p').cuda()
print('模型配置完成')
# trainer.save(args.save_path, 0, 0, 0)

iterations = trainer.resume_p(args.resume) if args.resume is not None else 0
print('加载模型完成，从{}轮继续训练'.format(iterations))
# 加载数据集
train_dataset = dataset(cfg.label_root, cfg.data_root)
valid_dataset = dataset(cfg.label_root, cfg.data_root, mode='val')

train_loader = data.DataLoader(train_dataset, 64,
                               num_workers=8,
                               shuffle=True,
                               pin_memory=False
                            #    sampler=train_sampler
                               )
valid_loader = data.DataLoader(valid_dataset, 64,
                               num_workers=8,
                               shuffle=False,
                               pin_memory=True
                            #    sampler=valid_sampler
                               )
print('数据集配置完成')
# 开始训练
max_all = 1.5
max_v = 0
max_a = 0
for epoch in range(50):
    print('**************************')
    print('第{}轮开始训练'.format(epoch))

    total_final_loss = 0
    total_v_loss = 0
    total_a_loss = 0
    # for batch, (images, _, _, v_labels, a_labels) in tqdm(enumerate(valid_loader)):
    #     images = Variable(images.cuda())
    #     v_labels = Variable(v_labels.cuda())
    #     a_labels = Variable(a_labels.cuda())

    #     trainer.val_p(images, v_labels, a_labels)
    # v_ccc, a_ccc = trainer.CCC()
    # acc_v, f1_v, acc_a, f1_a = trainer.metric_p()
    # print('acc_v = ', acc_v)
    # print('f1_v = ', f1_v)
    # print('acc_a = ', acc_a)
    # print('f1_a = ', f1_a)
    # exit()
    for batch, (images, _, _, v_labels, a_labels) in tqdm(enumerate(train_loader)):
        # s_time = time.time()
        images = images.cuda().detach()
        v_labels = v_labels.cuda().detach()
        a_labels = a_labels.cuda().detach()
        # print('数据放置时间:{}'.format(time.time()-s_time))
        '''训练va值'''
        s_time = time.time()
        final_loss, v_loss, a_loss = trainer.train_p(images, v_labels, a_labels)
        total_final_loss += final_loss
        total_v_loss += v_loss
        total_a_loss += a_loss
        # print('训练时间:{}'.format(time.time()-s_time))
    
    acc_v, f1_v, acc_a, f1_a = trainer.metric_p()
    print('\n-----------loss---------------')
    print('loss = {:.6f}, v_loss = {:.6f}, a_loss = {:.6f}'.format(total_final_loss/(batch+1), total_v_loss/(batch+1), total_a_loss/(batch+1)))
    print('-----------train---------------')
    print('acc_v= ', acc_v)
    print('f1_v= ', f1_v)
    print('acc_a= ', acc_a)
    print('f1_a= ', f1_a)

    # summary_writer.add_scalar('train/v_ccc', v_ccc, epoch)
    # summary_writer.add_scalar('train/a_ccc', a_ccc, epoch)
    # summary_writer.add_scalar('train/all_loss', total_final_loss/(batch+1), epoch)
    # summary_writer.add_scalar('train/v_loss', total_v_loss/(batch+1), epoch)
    # summary_writer.add_scalar('train/a_loss', total_a_loss/(batch+1), epoch)
    # summary_writer.add_scalar('train/match_loss', total_match_loss/(batch+1), epoch)

    for batch, (images, _, _, v_labels, a_labels) in tqdm(enumerate(valid_loader)):
        images = Variable(images.cuda())
        v_labels = Variable(v_labels.cuda())
        a_labels = Variable(a_labels.cuda())

        trainer.val_p(images, v_labels, a_labels)
    # v_ccc, a_ccc = trainer.CCC()
    acc_v, f1_v, acc_a, f1_a = trainer.metric_p()
    print('-----------val---------------')
    print('acc_v= ', acc_v)
    print('f1_v= ', f1_v)
    print('acc_a= ', acc_a)
    print('f1_a= ', f1_a)
    if acc_v + acc_a > max_all:
        trainer.save_p(args.save_path, epoch + iterations, acc_v, acc_a)
        max_all = acc_v + acc_a
        max_v = acc_v
        max_a = acc_a
    # summary_writer.add_scalar('val/v_ccc', v_ccc, epoch)
    # summary_writer.add_scalar('val/a_ccc', a_ccc, epoch)
    # summary_writer.add_scalar('val/v_loss', val_v_loss/(batch+1), epoch)
    # summary_writer.add_scalar('val/a_loss', val_a_loss/(batch+1), epoch)
summary_writer.close()