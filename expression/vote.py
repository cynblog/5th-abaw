import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import Counter
from torch.utils.data import Dataset
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

from torch.utils.tensorboard import SummaryWriter  

from pathlib import Path
import pandas as pd


import  os
from tqdm import tqdm

#定义一些超参数
BATCHSIZE=64 
EPOCHES=20
LR=0.0005
numclass = 8

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('/home/wjh/ABAW5th/202303131513')
# -------------------------------------------------------------------------------------------
#	模型定义
# -------------------------------------------------------------------------------------------
# mobilenet-v2模型
def mbnet():
	model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
	for param in model.parameters():
		param.requires_grad = True

	fc = nn.Sequential(
		nn.Dropout(0.2),
		nn.Linear(1280, numclass),
	)
	model.classifier = fc
	return model

# mnasnet模型
def mnasnet():
	model = models.MNASNet(alpha=1)

	for param in model.parameters():
		param.requires_grad = True

	fc = nn.Sequential(
		nn.Dropout(0.2),
		nn.Linear(1280, numclass),
	)
	model.classifier = fc

	return model

def densenet121():
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = True

    classifier = nn.Sequential(
                        nn.Linear(1024, 500),
                        nn.ReLU(),
                        nn.Dropout(0.4),
                        nn.Linear(500, numclass)
                        )

    model.classifier = classifier

    return model

def densenet201():
    model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = True

    classifier = nn.Sequential(
                        nn.Linear(1920, 500),
                        nn.ReLU(),
                        nn.Dropout(0.4),
                        nn.Linear(500, numclass)
                        )

    model.classifier = classifier

    return model



#  resnet 18模型
def resnet18(fc_num=256, class_num=numclass):
	model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
	for param in model.parameters():
		param.requires_grad = True

	fc_inputs = model.fc.in_features
	model.fc = nn.Sequential(
		nn.Linear(fc_inputs, fc_num),
		nn.ReLU(),
		nn.Dropout(0.4),
		nn.Linear(fc_num, class_num)
	)

	return model

#  resnet 152模型
def resnet152(fc_num=256, class_num=numclass):
	model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
	for param in model.parameters():
		param.requires_grad = True

	fc_inputs = model.fc.in_features
	model.fc = nn.Sequential(
		nn.Linear(fc_inputs, fc_num),
		nn.ReLU(),
		nn.Dropout(0.4),
		nn.Linear(fc_num, class_num)
	)
	return model

# -------------------------------------------------------------------------------------------
#	数据加载
# -------------------------------------------------------------------------------------------
class dataload(Dataset):
    MEAN = [0.367035294117647, 0.41083294117647057, 0.5066129411764705]  
    STD = [1, 1, 1] 

    def __init__(self, df, dataset_name, img_dir) -> None:
        super(dataload).__init__()
        self.df = df
        self.img_dir = img_dir

        if dataset_name in ['train']:
            self.preprocess = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(224, scale=[0.8, 1.0]),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.5, contrast=0.7, saturation=0.3, hue=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.MEAN, std=self.STD)
            ])
        elif dataset_name in ['valid','test','test']:
            self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD)
            ])  

    def __getitem__(self, index) :
        
        img_name = self.df['img_name'].iloc[index]
        lable = self.df['lable'].iloc[index]

        img_path = self.img_dir / img_name
        X = Image.open(img_path)
        X = self.preprocess(X)


        return X, lable

    def __len__(self):
        return len(self.df)

import matplotlib.pyplot as pl
from sklearn import metrics
# 相关库

def plot_matrix(y_true, y_pred, labels_name, save_dir, title=None, thresh=0.8, axis_labels=None):

    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  
    pl.imshow(cm, interpolation='nearest', cmap=pl.get_cmap('Blues'))
    pl.colorbar() 

    if title is not None:
        pl.title(title)

    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    pl.xticks(num_local, axis_labels, rotation=45)  
    pl.yticks(num_local, axis_labels)  
    pl.ylabel('True label')
    pl.xlabel('Predicted label')


    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) > 0:
                pl.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
                        ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black") 

    pl.savefig(save_dir)
    pl.clf()
    pl.cla()



def data_process(batch_size=32):
	
    csv_dir = Path('/home/wjh/ABAW5th/data_csv')
    img_dir = Path('/data03/cvpr23_competition/cvpr23_competition_data/cropped_aligned_images')

    df_train = pd.read_csv(csv_dir / 'train.csv')
    df_train = df_train.drop(df_train[(df_train['lable'] == -1)].index)


    df_valid = pd.read_csv(csv_dir / 'valid.csv')
    df_valid = df_valid.drop(df_valid[(df_valid['lable'] == -1)].index)
    df_valid = df_valid[:int(len(df_valid)/BATCHSIZE)*BATCHSIZE]

    
    label_conuts = df_train['lable'].value_counts()

    print(label_conuts)
    train_set = dataload(df_train,'train',img_dir)
    valid_set = dataload(df_valid,'valid',img_dir) 

    batch_size = BATCHSIZE
    train_data = DataLoader(train_set, batch_size, shuffle=True, num_workers=8)
    valid_data = DataLoader(valid_set, batch_size, shuffle=False, num_workers=8)

    train_data_size = len(df_train)
    valid_data_size = len(df_valid)

    print("[INFO] Train data / Test data number: ", train_data_size, valid_data_size)
    return train_data, valid_data, label_conuts


class BinaryDiceLoss(nn.Module):
	def __init__(self):
		super(BinaryDiceLoss, self).__init__()
	
	def forward(self, input, targets):
		N = targets.size()[0]
		smooth = 1
		input_flat = input.view(N, -1)
		targets_flat = targets.view(N, -1)
	
		intersection = input_flat * targets_flat 
		N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
		loss = 1 - N_dice_eff.sum() / N
		return loss

class MultiClassDiceLoss(nn.Module):
	def __init__(self, weight=None, ignore_index=None, **kwargs):
		super(MultiClassDiceLoss, self).__init__()
		self.weight = weight
		self.ignore_index = ignore_index
		self.kwargs = kwargs
	
	def forward(self, input, target):
		nclass = input.shape[1]
		target = F.one_hot(target.long(), nclass)

		assert input.shape == target.shape, "predict & target shape do not match"
		
		binaryDiceLoss = BinaryDiceLoss()
		total_loss = 0
		
		logits = F.softmax(input, dim=1)
		C = target.shape[1]
		
		for i in range(C):
			dice_loss = binaryDiceLoss(logits[:, i], target[:, i])
			total_loss += dice_loss
		
		return total_loss / C


# -------------------------------------------------------------------------------------------
#	集成表决
# -------------------------------------------------------------------------------------------
def process(mlps, trainloader, testloader, label_conuts):
    optimizer = torch.optim.Adam([{"params":filter(lambda p: p.requires_grad, mlp.parameters())} for mlp in mlps], lr=LR)

    weight=torch.from_numpy(np.array([1/label_conuts[0], 1/label_conuts[1], 1/label_conuts[2], 1/label_conuts[3], 1/label_conuts[4], 1/label_conuts[5], 1/label_conuts[6],1/label_conuts[7]])).float() 
    
    CE_loss_function = nn.CrossEntropyLoss(weight=weight, reduction='mean').to(device)
    dice_loss_function = MultiClassDiceLoss().to(device)
    
    best_f1 = -1
    for ep in range(EPOCHES):
        print("Epoch: {}/{}".format(ep + 1, EPOCHES))
        print("[INFO] Begin to train")
        mlps_pred_train = [[] for i in range(len(mlps))]
        label_train=[]
        for img, label in tqdm(trainloader):
            img, label = img.to(device), label.to(device)
            label_train.append(label)
            optimizer.zero_grad()  
            for i, mlp in enumerate(mlps):
                mlp.train()
                out = mlp(img)
                mlps_pred_train[i].append(out.to('cpu'))
                
                CEloss = CE_loss_function(out, label)
                Diceloss = dice_loss_function(out, label)
                
                loss = CEloss + 1.5 * Diceloss
                loss.backward()  
            optimizer.step() 
        
        label_train = torch.cat(label_train)
        label_train = label_train.cpu().detach().numpy() 
        
        for idx, mlp_train in enumerate(mlps_pred_train):
            mlp_train = torch.cat(mlp_train)
            mlp_train = mlp_train.detach().numpy()
            mlp_train_c = mlp_train
            mlp_train = mlp_train.argmax(axis=1)
            mlp_acc = accuracy_score(mlp_train, label_train)
            mlp_f1 = f1_score(mlp_train, label_train, average='macro')

            mlp_loss = CE_loss_function(torch.from_numpy(mlp_train_c).float().to(device),torch.from_numpy(label_train).to(device)) + dice_loss_function(torch.from_numpy(mlp_train_c).float(),torch.from_numpy(label_train))

            plot_matrix(y_true=label_train, y_pred=mlp_train, labels_name=[0,1,2,3,4,5,6,7], save_dir='/home/wjh/ABAW5th/confusion_matrix/train-pic-{}-model-{}.png'.format(ep + 1, idx + 1), title='train-pic-{}-model-{}.png'.format(ep + 1, idx + 1), thresh=0.8, axis_labels=['neutral', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'other'])
            
            writer.add_scalar('train_loss/'+str(idx),mlp_loss.item(),ep)
            writer.add_scalar('train_acc/'+str(idx),mlp_acc,ep)
            writer.add_scalar('train_f1/'+str(idx),mlp_f1,ep)
            print("模型" + str(idx) + "的acc=" + str(mlp_acc) + ", f1=" + str(mlp_f1) + ", loss=" + str(mlp_loss.item()))   
        for i, mlp in enumerate(mlps):
            torch.save(mlp.state_dict(), '/home/wjh/ABAW5th/vote_model_state/model'+str(i)+'_202303151833.pth')
        pre = []
        mlps_pred_valid = [[] for i in range(len(mlps))]
        label_valid = []
        vote_valid = []    
        print("[INFO] Begin to valid")
        with torch.no_grad():
            for img, label in tqdm(testloader):
                img = img.to(device)
                label_valid.append(label)
                for i, mlp in enumerate(mlps):
                    mlp.eval()
                    out = mlp(img)
                    mlps_pred_valid[i].append(out.to('cpu'))

                    _, prediction = torch.max(out, 1)  
                    pre_num = prediction.cpu().numpy()   
                    pre.append(pre_num)
                arr = np.array(pre)
                pre.clear()
                
                result = [Counter(arr[:, i]).most_common(1)[0][0] for i in range(BATCHSIZE)]
                for i_res in range(BATCHSIZE):
                    for j_res in [1,5,3,2]:
                        if j_res in arr[:, i_res]:
                            result[i_res] = j_res
                vote_valid.extend(result)
    
        label_valid = torch.cat(label_valid)
        label_valid = label_valid.detach().numpy() 
        vote_valid = np.array(vote_valid)
        vote_acc = accuracy_score(vote_valid, label_valid)
        vote_f1 = f1_score(vote_valid, label_valid, average='macro')
        
        writer.add_scalar('vote_acc',vote_acc,ep)
        writer.add_scalar('vote_f1',vote_f1,ep)

        plot_matrix(y_true=label_valid, y_pred=vote_valid, labels_name=[0,1,2,3,4,5,6,7], save_dir='/home/wjh/ABAW5th/confusion_matrix/vote-pic-{}.png'.format(ep + 1), title='vote-pic-{}.png'.format(ep + 1), thresh=0.8, axis_labels=['neutral', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'other'])

        
        if vote_f1 > best_f1:
            for i, mlp in enumerate(mlps):
                torch.save(mlp.state_dict(), '/home/wjh/ABAW5th/vote_model_state/model'+str(i)+'_202303131513.pth')

        print("epoch:" + str(ep + 1) + "\n集成模型的acc=" + str(vote_acc) + ", f1=" + str(vote_f1) )    
        for idx, mlp_valid in enumerate(mlps_pred_valid):
            mlp_valid = torch.cat(mlp_valid)
            mlp_valid = mlp_valid.detach().numpy()
            mlp_valid_c = mlp_valid
            mlp_valid = mlp_valid.argmax(axis=1)
            mlp_acc = accuracy_score(mlp_valid, label_valid)
            mlp_f1 = f1_score(mlp_valid, label_valid, average='macro')
            mlp_loss = CE_loss_function(torch.from_numpy(mlp_valid_c).float().to(device),torch.from_numpy(label_valid).to(device)) + dice_loss_function(torch.from_numpy(mlp_valid_c).float(),torch.from_numpy(label_valid))
            
            plot_matrix(y_true=label_valid, y_pred=mlp_valid, labels_name=[0,1,2,3,4,5,6,7], save_dir='/home/wjh/ABAW5th/confusion_matrix/valid-pic-{}-model-{}.png'.format(ep + 1, idx + 1), title='valid-pic-{}-model-{}.png'.format(ep + 1, idx + 1), thresh=0.8, axis_labels=['neutral', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'other'])

            writer.add_scalar('valid_loss/'+str(idx),mlp_loss.item(),ep)
            writer.add_scalar('valid_acc/'+str(idx),mlp_acc,ep)
            writer.add_scalar('valid_f1/'+str(idx),mlp_f1,ep)
            print("模型" + str(idx) + "的acc=" + str(mlp_acc) + ", f1=" + str(mlp_f1) + ", loss=" + str(mlp_loss.item()))



def predict_batch(mlps, df):
    img_dir = Path('/data03/cvpr23_competition/cvpr23_competition_data/cropped_aligned_images')
    test_set = dataload(df,'test',img_dir) 
    batch_size = BATCHSIZE
    test_data = DataLoader(test_set, batch_size, shuffle=True, num_workers=8)

    pre = []
    all_result = []
    with torch.no_grad():
        for img, label in tqdm(test_data):
            img = img.to(device)
                
            for i, mlp in enumerate(mlps):
                mlp.eval()
                out = mlp(img)

                _, prediction = torch.max(out, 1)  # 按行取最大值
                pre_num = prediction.cpu().numpy()   
                pre.append(pre_num)
            arr = np.array(pre)
            pre.clear()
                
            result = [Counter(arr[:, i]).most_common(1)[0][0] for i in range(BATCHSIZE)]
            for i in range(BATCHSIZE):
                for j in [1,5,3,2]:
                    if j in arr[:, i]:
                        result[i] = j
                
            all_result.extend(result)
    
    return all_result
    

if __name__ == '__main__':
    mlps = [mbnet().to(device),  resnet152().to(device),  densenet121().to(device), resnet18().to(device), densenet201().to(device)]

    for index, mlp in enumerate(mlps):
        state_saved = torch.load('/home/wjh/ABAW5th/vote_model_state/model'+str(index)+'_202303131513.pth')
        mlp.load_state_dict(state_saved)
    
    train_data, valid_data, label_conuts = data_process()
    process(mlps=mlps, trainloader=train_data , testloader=valid_data, label_conuts=label_conuts)



