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
from Trainer import Trainer
from tensorboardX import SummaryWriter


# 参数配置
parser = argparse.ArgumentParser(
    description='VA Establish With Pytorch')
parser.add_argument('--resume',
                    default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--save_path',
                    default="./output/resnet_enet_regnet/110", type=str,
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
trainer = Trainer('regnet').cuda()
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
max_all = 0.55
max_v = 0
max_a = 0
for epoch in range(50):
    print('**************************')
    print('第{}轮开始训练'.format(epoch))
    # if epoch == 0:
    #     val_v_loss = 0
    #     val_a_loss = 0
    #     val_mse_loss = 0
    #     for batch, (images, image_names, v_labels, a_labels, _, _) in tqdm(enumerate(valid_loader)):
    #         images = Variable(images.cuda())
    #         v_labels = Variable(v_labels.cuda().float())
    #         a_labels = Variable(a_labels.cuda().float())
    #         v_loss, a_loss, mse_loss = trainer.val(images, image_names, v_labels, a_labels)
    #         val_v_loss += v_loss
    #         val_a_loss += a_loss
    #         val_mse_loss += mse_loss

    #     v_ccc, a_ccc = trainer.CCC()
    #     # acc_v, f1_v, acc_a, f1_a = trainer.metric_p()
    #     print('v_loss = {:.6f}, a_loss = {:.6f}, mse_loss = {:.6f}'.format(val_v_loss/(batch+1), val_a_loss/(batch+1), val_mse_loss/(batch+1)))
    #     print('-----------val---------------')
    #     print('v_ccc= ', v_ccc)
    #     print('a_ccc= ', a_ccc)

    total_final_loss = 0
    total_v_loss = 0
    total_a_loss = 0
    total_mse_loss = 0
    for batch, (images, v_labels, a_labels, _, _) in tqdm(enumerate(train_loader)):
        images = Variable(images.cuda()).detach()
        v_labels = Variable(v_labels.cuda().float()).detach()
        a_labels = Variable(a_labels.cuda().float()).detach()
        '''训练va值'''
        final_loss, v_loss, a_loss, mse_loss = trainer.train(images, v_labels, a_labels)
        total_final_loss += final_loss
        total_v_loss += v_loss
        total_a_loss += a_loss
        total_mse_loss += mse_loss
        
    v_ccc, a_ccc = trainer.CCC()
    print('-----------loss---------------')
    print('loss = {:.6f}, v_loss = {:.6f}, a_loss = {:.6f}, mse_loss = {:.6f}'.format(total_final_loss/(batch+1), total_v_loss/(batch+1), total_a_loss/(batch+1), total_mse_loss/(batch+1)))
    print('-----------train---------------')
    print('v_ccc= ', v_ccc)
    print('a_ccc= ', a_ccc)

    summary_writer.add_scalar('train/v_ccc', v_ccc, epoch)
    summary_writer.add_scalar('train/a_ccc', a_ccc, epoch)
    summary_writer.add_scalar('train/all_loss', total_final_loss/(batch+1), epoch)
    summary_writer.add_scalar('train/v_loss', total_v_loss/(batch+1), epoch)
    summary_writer.add_scalar('train/a_loss', total_a_loss/(batch+1), epoch)
    # summary_writer.add_scalar('train/mse_loss', total_mse_loss/(batch+1), epoch)

    val_v_loss = 0
    val_a_loss = 0
    val_mse_loss = 0
    for batch, (images, image_names, v_labels, a_labels, _, _) in tqdm(enumerate(valid_loader)):
        images = Variable(images.cuda())
        v_labels = Variable(v_labels.cuda().float())
        a_labels = Variable(a_labels.cuda().float())
        v_loss, a_loss, mse_loss = trainer.val(images, image_names, v_labels, a_labels)
        val_v_loss += v_loss
        val_a_loss += a_loss
        val_mse_loss += mse_loss

    v_ccc, a_ccc = trainer.CCC()
    # acc_v, f1_v, acc_a, f1_a = trainer.metric_p()
    print('v_loss = {:.6f}, a_loss = {:.6f}, mse_loss = {:.6f}'.format(val_v_loss/(batch+1), val_a_loss/(batch+1), val_mse_loss/(batch+1)))
    print('-----------val---------------')
    print('v_ccc= ', v_ccc)
    print('a_ccc= ', a_ccc)
    if v_ccc + a_ccc > max_all:
        trainer.save(args.save_path, epoch + iterations, v_ccc, a_ccc)
        max_all = v_ccc + a_ccc
        max_v = v_ccc
        max_a = a_ccc
    summary_writer.add_scalar('val/v_ccc', v_ccc, epoch)
    summary_writer.add_scalar('val/a_ccc', a_ccc, epoch)
    summary_writer.add_scalar('val/v_loss', val_v_loss/(batch+1), epoch)
    summary_writer.add_scalar('val/a_loss', val_a_loss/(batch+1), epoch)
    # summary_writer.add_scalar('val/mse_loss', val_mse_loss/(batch+1), epoch)
summary_writer.close()