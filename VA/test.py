import torch
import math
import random
import argparse
import os

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
                    default='/data02/mxy/CVPR_ABAW/precode/output/resnet_enet/110', type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--save_path', default='/data02/mxy/CVPR_ABAW/precode/results/1.txt')

args = parser.parse_args()


# 加载模型
trainer = Trainer('resnet_enet', phase='test').cuda()
print('模型配置完成')
# trainer.save(args.save_path, 0, 0, 0)

iterations = trainer.resume_test(args.resume)

# 加载数据集
test_dataset = dataset(cfg.label_root, cfg.data_root, mode='test')

test_loader = data.DataLoader(test_dataset, 64,
                               num_workers=4,
                               shuffle=False,
                               pin_memory=True
                            #    sampler=valid_sampler
                               )
print('数据集配置完成')
# 开始训练
for batch, (images, images_name) in tqdm(enumerate(test_loader)):
    images = Variable(images.cuda())

    trainer.test(images, images_name)

# trainer.get_pred_bar()
trainer.write_results(args.save_path)

