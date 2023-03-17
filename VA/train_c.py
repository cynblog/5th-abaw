import torch
import math
import random
import argparse
import os
import sys
import signal

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
from Trainer_c import Trainer
from tensorboardX import SummaryWriter


# 参数配置
parser = argparse.ArgumentParser(
    description='VA Establish With Pytorch')
parser.add_argument('--resume',
                    default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--save_path',
                    default="./output/resnet_enet_c/v+", type=str,
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


# 加载模型
trainer = Trainer('resnet_enet').cuda()
print('模型配置完成')
# trainer.save(args.save_path, 0, 0, 0)

iterations = trainer.resume(args.resume) if args.resume is not None else 0

# 加载数据集
train_dataset = dataset(cfg.label_root, cfg.data_root)
valid_dataset = dataset(cfg.label_root, cfg.data_root, mode='val')

train_loader = data.DataLoader(train_dataset, 64,
                               num_workers=8,
                               shuffle=True,
                               pin_memory=True
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
max_all = 0.25
max_v = 0
max_a = 0
for epoch in range(50):
    print('**************************')
    print('第{}轮开始训练'.format(epoch))

    total_final_loss = 0
    total_v_loss = 0
    total_a_loss = 0
    total_match_loss = 0
    for batch, (images, v_labels, a_labels, _, _) in tqdm(enumerate(train_loader)):
        images = Variable(images.cuda()).detach()
        v_labels = Variable(v_labels.cuda()).detach()
        a_labels = Variable(a_labels.cuda()).detach()
        '''训练va值'''
        loss = trainer.train(images, a_labels)
        total_final_loss += loss
        
    ccc = trainer.CCC()
    print('-----------loss---------------')
    print('loss = {:.6f}'.format(total_final_loss/(batch+1)))
    print('-----------train---------------')
    print('ccc = ', ccc)

    # summary_writer.add_scalar('train/v_ccc', v_ccc, epoch)
    # summary_writer.add_scalar('train/a_ccc', a_ccc, epoch)
    # summary_writer.add_scalar('train/all_loss', total_final_loss/(batch+1), epoch)
    # summary_writer.add_scalar('train/v_loss', total_v_loss/(batch+1), epoch)
    # summary_writer.add_scalar('train/a_loss', total_a_loss/(batch+1), epoch)
    # summary_writer.add_scalar('train/match_loss', total_match_loss/(batch+1), epoch)

    val_loss = 0
    for batch, (images, v_labels, a_labels, _, _) in tqdm(enumerate(valid_loader)):
        images = Variable(images.cuda())
        v_labels = Variable(v_labels.cuda())
        a_labels = Variable(a_labels.cuda())

        loss = trainer.val(images, a_labels)
        val_loss += loss
    ccc = trainer.CCC()
    print('loss = {:.6f}'.format(val_loss/(batch+1)))
    print('-----------val---------------')
    print('ccc = ', ccc)
    if ccc > max_all:
        trainer.save(args.save_path, epoch + iterations, ccc)
        max_all = ccc