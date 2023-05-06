import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import math
from scipy import io as sio
import torch.utils.data
import torchvision.models as models
from imgaug import augmenters as iaa
import pandas as pd
import torchvision.transforms as transforms
import torchvision
import time
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader
from skimage import io
import os
import torchvision.transforms.functional as tf
from PIL import Image
import cv2
import torchvision.models as models
from prefetch_generator import BackgroundGenerator
import matplotlib.pyplot as plt
import lmdb
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from timm.models.vision_transformer import VisionTransformer as VisionTransformer_org

from my_vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from functools import partial
import  copy
from torch.nn.parallel import DataParallel
import torch
from torch.nn.parallel._functions import Scatter
from torch.nn.parallel.parallel_apply import parallel_apply

def scatter(inputs, target_gpus, chunk_sizes, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            try:
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except:
                print('obj', obj.size())
                print('dim', dim)
                print('chunk_sizes', chunk_sizes)
                quit()
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None

def scatter_kwargs(inputs, kwargs, target_gpus, chunk_sizes, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, chunk_sizes, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs

class BalancedDataParallel(DataParallel):
    def __init__(self, gpu0_bsz, *args, **kwargs):
        self.gpu0_bsz = gpu0_bsz
        super().__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        if self.gpu0_bsz == 0:
            device_ids = self.device_ids[1:]
        else:
            device_ids = self.device_ids
        inputs, kwargs = self.scatter(inputs, kwargs, device_ids)

        # print('len(inputs): ', str(len(inputs)))
        # print('self.device_ids[:len(inputs)]', str(self.device_ids[:len(inputs)]))

        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        if self.gpu0_bsz == 0:
            replicas = self.replicate(self.module, self.device_ids)
        else:
            replicas = self.replicate(self.module, self.device_ids[:len(inputs)])

        # replicas = self.replicate(self.module, device_ids[:len(inputs)])
        if self.gpu0_bsz == 0:
            replicas = replicas[1:]

        #print('replicas:', str(len(replicas)))

        outputs = self.parallel_apply(replicas, device_ids, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, device_ids, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, device_ids[:len(inputs)])

    def scatter(self, inputs, kwargs, device_ids):
        bsz = inputs[0].size(self.dim)
        num_dev = len(self.device_ids)
        gpu0_bsz = self.gpu0_bsz
        bsz_unit = (bsz - gpu0_bsz) // (num_dev - 1)
        if gpu0_bsz < bsz_unit:
            chunk_sizes = [gpu0_bsz] + [bsz_unit] * (num_dev - 1)
            delta = bsz - sum(chunk_sizes)
            for i in range(delta):
                chunk_sizes[i + 1] += 1
            if gpu0_bsz == 0:
                chunk_sizes = chunk_sizes[1:]
        else:
            return super().scatter(inputs, kwargs, device_ids)

        # print('bsz: ', bsz)
        # print('num_dev: ', num_dev)
        # print('gpu0_bsz: ', gpu0_bsz)
        # print('bsz_unit: ', bsz_unit)
        # print('chunk_sizes: ', chunk_sizes)
        return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class Mydataset(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = torch.FloatTensor(labels)
    def __getitem__(self, index):
        return torch.from_numpy(self.imgs[index]),self.labels[index]
    def __len__(self):
        return (self.imgs).shape[0]

class Mydataset2(Dataset):
    def __init__(self, imgs, imgs2, labels):
        self.imgs = imgs
        self.imgs2 = imgs2
        self.labels = torch.FloatTensor(labels)

    def __getitem__(self, index):
        return torch.from_numpy(self.imgs[index]), torch.from_numpy(self.imgs2[index]), self.labels[index]

    def __len__(self):
        return (self.imgs).shape[0]


def test_prob(model, test_loader, epoch, device, all_test_loss):
    model.eval()

    op = []
    tg = []
    with torch.no_grad():
        for batch_idx, (data,target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            data = data[:, :, 10:10 + 224, 10:10 + 224]

            data = data.float()
            data /= 255
            data[:, 0] -= 0.485
            data[:, 1] -= 0.456
            data[:, 2] -= 0.406
            data[:, 0] /= 0.229
            data[:, 1] /= 0.224
            data[:, 2] /= 0.225
            output = model(data)
            output=F.softmax(output)
            if batch_idx==0:
                op=output
            else:
                op=torch.cat((op,output),0)


    return all_test_loss,op.cpu().numpy()


def train(model,  train_loader, optimizer,  epoch, device, all_train_loss):
    model.train()
    st = time.time()
    op=[]
    op2=[]
    tg=[]
    for batch_idx, (data,data2, target) in enumerate(train_loader):
        data,data2,  target = data.to(device),data2.to(device),  target.to(device)
        torch.random.manual_seed(len(train_loader) * epoch + batch_idx)
        rd_ps = torch.randint(20, (3,))
        data = data[:, :, rd_ps[0]:rd_ps[0] + 224, rd_ps[1]:rd_ps[1] + 224]
        data2 = data2[:, :, rd_ps[0]:rd_ps[0] + 224, rd_ps[1]:rd_ps[1] + 224]
        if rd_ps[1] < 10:
            data = torch.flip(data, dims=[3])
            data2 = torch.flip(data2, dims=[3])

        data2 = data2.float()
        data2 /= 255
        data2[:, 0] -= 0.485
        data2[:, 1] -= 0.456
        data2[:, 2] -= 0.406
        data2[:, 0] /= 0.229
        data2[:, 1] /= 0.224
        data2[:, 2] /= 0.225

        data = data.float()
        data /= 255
        data[:, 0] -= 0.485
        data[:, 1] -= 0.456
        data[:, 2] -= 0.406
        data[:, 0] /= 0.229
        data[:, 1] /= 0.224
        data[:, 2] /= 0.225

        optimizer.zero_grad()
        output, output2 = model(data, data2)

        loss1 = F.mse_loss(output, target)
        ind=torch.nonzero(target<2)[:,0]
        loss2 = F.mse_loss(output2, target)#+5*F.l1_loss(output2[ind],target[ind])
        loss=loss1+loss2
        all_train_loss.append(loss.item())
        loss.backward()
        optimizer.step()


        op = np.concatenate((op, output[:, 0].detach().cpu().numpy()))
        op2 = np.concatenate((op2, output2[:, 0].detach().cpu().numpy()))
        tg = np.concatenate((tg, target[:, 0].cpu().numpy()))
        p1 = pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="pearson")
        s1 = pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="spearman")

        if batch_idx % 100 == 0:
            print('Train Epoch:{} [({:.0f}%)]\t Loss: {:.4f} L1oss: {:.4f} Loss2: {:.4f}  Pearson:{:.4f} Spearman:{:.4f}'.format(
                epoch, 100. * batch_idx / len(train_loader), loss.item(),  loss1.item(),  loss2.item(), p1, s1))

    print( 'Train ALL Pearson:', pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="pearson"),' ALL Pearson2:', pd.Series((op2[::1])).corr((pd.Series(tg[::1])), method="pearson"))
    print( 'Train  ALL Spearman:', pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="spearman"),' ALL Spearman2:', pd.Series((op2[::1])).corr((pd.Series(tg[::1])), method="spearman"))
    ind = np.where(tg < 2)[ 0]
    print('Low Quality Pearson:', pd.Series((op[ind])).corr((pd.Series(tg[ind])), method="pearson"),'Low Quality Pearson2:', pd.Series((op2[ind])).corr((pd.Series(tg[ind])), method="pearson"))
    print('Low Quality Spearman:', pd.Series((op[ind])).corr((pd.Series(tg[ind])), method="spearman"),'Low Quality Spearman2:', pd.Series((op2[ind])).corr((pd.Series(tg[ind])), method="spearman"))
    return all_train_loss


def test(model,  test_loader, epoch, device, all_test_loss):
    model.eval()
    test_loss = 0
    pearson = 0
    spearman = 0
    op = []
    op2 = []
    tg = []
    with torch.no_grad():
        for batch_idx, (data, data2, target) in enumerate(test_loader):
            data, data2, target = data.to(device),data2.to(device),target.to(device)

            data = data[:, :,10:10+ 224, 10:10+ 224]
            data2 = data2[:, :, 10:10 + 224, 10:10 + 224]


            data2 = data2.float()
            data2 /= 255
            data2[:, 0] -= 0.485
            data2[:, 1] -= 0.456
            data2[:, 2] -= 0.406
            data2[:, 0] /= 0.229
            data2[:, 1] /= 0.224
            data2[:, 2] /= 0.225

            data = data.float()
            data /= 255
            data[:, 0] -= 0.485
            data[:, 1] -= 0.456
            data[:, 2] -= 0.406
            data[:, 0] /= 0.229
            data[:, 1] /= 0.224
            data[:, 2] /= 0.225


            output1,output2=model(data,data2)
            loss1 = F.mse_loss(output1, target)
            ind = torch.nonzero(target < 2)[:, 0]
            loss2 = F.mse_loss(output2, target) #+ 5 * F.l1_loss(output2[ind], target[ind])
            loss = loss1 + loss2
            all_test_loss.append(loss)
            test_loss += loss
            op = np.concatenate((op, output1[:, 0].cpu().numpy()))
            op2 = np.concatenate((op2, output2[:, 0].cpu().numpy()))
            tg = np.concatenate((tg, target[:, 0].cpu().numpy()))
            p1 = pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="pearson")
            s1 = pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="spearman")
            pearson += p1
            spearman += s1
            if batch_idx % 100 == 0:
                print('Test Epoch:{} [({:.0f}%)]\t Loss: {:.4f}  Loss1: {:.4f}  Loss2: {:.4f}  Pearson:{:.4f} Spearman:{:.4f}'.format(
                    epoch, 100. * batch_idx / len(test_loader), loss.item(), loss1.item(), loss2.item(), p1, s1))

    test_loss /= (batch_idx + 1)
    pearson /= (batch_idx + 1)
    spearman /= (batch_idx + 1)
    print('Test : Loss:{:.4f} '.format(test_loss))
    print('Test ALL Pearson:', pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="pearson"), ' ALL Pearson2:',pd.Series((op2[::1])).corr((pd.Series(tg[::1])), method="pearson"))
    print('Test  ALL Spearman:', pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="spearman"), ' ALL Spearman2:',pd.Series((op2[::1])).corr((pd.Series(tg[::1])), method="spearman"))
    ind = np.where(tg < 2)[0]
    print('Low Quality Pearson:', pd.Series((op[ind])).corr((pd.Series(tg[ind])), method="pearson"), 'Low Quality Pearson2:', pd.Series((op2[ind])).corr((pd.Series(tg[ind])), method="pearson"))
    print('Low Quality Spearman:', pd.Series((op[ind])).corr((pd.Series(tg[ind])), method="spearman"),'Low Quality Spearman2:', pd.Series((op2[ind])).corr((pd.Series(tg[ind])), method="spearman"))

    return all_test_loss, pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="pearson"), pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="spearman"), pd.Series((op[ind])).corr((pd.Series(tg[ind])), method="pearson"), pd.Series((op[ind])).corr((pd.Series(tg[ind])), method="spearman")

class MyModel(nn.Module):

    def __init__(self, model1,model2,model3,model4):
        super(MyModel, self).__init__()
        self.model1=model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4

    def forward(self, data,data2,):
        opf1 = self.model1(data)
        opf2 = self.model2(data2)
        op1 = self.model3(torch.cat((opf1, opf1.detach() - opf2.detach()), 1))
        op2 = self.model4(torch.cat((opf1.detach(), opf1.detach() - opf2), 1))

        return op1,op2

class MyModel2(nn.Module):

    def __init__(self, model1,model2,model3,model4):
        super(MyModel2, self).__init__()
        self.model1=model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4

    def forward(self, data,data2):
        opf1 = self.model1(data)
        opf2_1 = self.model2(data)
        opf2_2 = self.model2(data2)
        op1 = self.model3(torch.cat((opf1, (opf2_1- opf2_2).detach()), 1))
        op2 = self.model4(torch.cat((opf2_1, opf2_1 - opf2_2), 1))

        return op1,op2

class MyModel3(nn.Module):

    def __init__(self, model1,model2,model3,model4):
        super(MyModel3, self).__init__()
        self.model1=model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4

    def forward(self, data,data2):
        opf1 = self.model1(data)
        opf2_1 = self.model2(data)
        opf2_2 = self.model2(data2)
        op1 = self.model3(torch.cat((opf1,opf2_1,opf2_2), 1))
        op2 = self.model4(torch.cat((opf2_1, opf2_2), 1))

        return op1,op2

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    device = torch.device("cuda")

    # model2 = VisionTransformer_org(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
    #                           norm_layer=partial(nn.LayerNorm, eps=1e-6))
    # model2.default_cfg = _cfg()
    # model2.load_state_dict(torch.load("deit_small_patch16_224.pth")['model'])
    # model2 = nn.DataParallel(model2.to(device))

    best_plccs=[]
    best_srccs = []
    best_low_plccs = []
    best_low_srccs = []
    all_data = sio.loadmat('E:\Database\LIVEW\livew_244.mat')
    X = all_data['X']
    Y = all_data['Y'].transpose(1, 0)
    Y = Y.reshape(Y.shape[0], 1)
    Y = Y / 25 + 1
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest'].transpose(1, 0)
    Ytest = Ytest / 25 + 1
    del all_data


    all_data = sio.loadmat('E:\Database\LIVEW\livew_244_brt_debulr.mat')
    X2 = all_data['X']
    Xtest2 = all_data['Xtest']
    del all_data

    all_data=np.load('livew_prob_compare.npz')
    op_train=all_data['op_train']
    op_train2 = all_data['op_train2']
    op_test= all_data['op_test']
    op_test2 = all_data['op_test2']
    ind_train=np.where((np.max(op_train2,axis=1)-np.max(op_train,axis=1))<0)[0]
    ind_test=np.where((np.max(op_test2,axis=1)-np.max(op_test,axis=1))<0)[0]
    X2[ind_train]=X[ind_train]
    Xtest2[ind_test] = Xtest[ind_test]




    for i in range(0,10):
        print('Split:',i)
        if i > 0:
            X = np.concatenate((X, Xtest), axis=0)
            Y = np.concatenate((Y, Ytest), axis=0)
            X2 = np.concatenate((X2, Xtest2), axis=0)
            ind = np.arange(0, X.shape[0])
            np.random.seed(i)
            np.random.shuffle(ind)

            Xtest = X[ind[int(len(ind) * 0.8):]]
            Ytest = Y[ind[int(len(ind) * 0.8):]]
            Xtest2 = X2[ind[int(len(ind) * 0.8):]]
            X = X[ind[:int(len(ind) * 0.8)]]
            Y = Y[ind[:int(len(ind) * 0.8)]]
            X2 = X2[ind[:int(len(ind) * 0.8)]]

        
        model1 = VisionTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        model1.default_cfg = _cfg()
        model1.load_state_dict(torch.load("deit_small_patch16_224.pth")['model'])
    
        model2=copy.deepcopy(model1)
    
        model3= nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(384*3, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1))
    
        model4 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(384 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1))
    
        model=MyModel3(model1,model2,model3,model4)
        for param in model.model1.parameters():
            param.requires_grad = False
        for param in model.model2.parameters():
            param.requires_grad = False
    
        for param in model.model3.parameters():
            param.requires_grad = True
        for param in model.model4.parameters():
            param.requires_grad = True
    
        # model = nn.DataParallel(model.to(device))
        model = BalancedDataParallel(32 // 1, model, dim=0).to(device)
        # model.load_state_dict(torch.load("Koniq__imagenet_ftsall_1.pt"))
        ###################################################################
    

        
        
        train_dataset = Mydataset2(X,X2, Y)
        test_dataset = Mydataset2(Xtest, Xtest2,Ytest)
    
        weight_decay = 1e-3
        batch_size = 32 * 4+18
        epochs = 2000
        num_workers_train = 0
        num_workers_test = 0
    
        train_loader = DataLoaderX(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers_train,pin_memory=True)
        test_loader = DataLoaderX(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers_test, pin_memory=True)
    
        all_train_loss = []
        all_test_loss = []
        all_test_loss, _,_,_,_ = test(model,test_loader, -1, device, all_test_loss)
        ct = 0
        lr = 0.01
        max_plsp = -2
    
        for epoch in range(epochs):
            print(lr)
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
            ct += 1
            start = time.time()
            all_train_loss = train(model,train_loader, optimizer, epoch, device, all_train_loss)
            print(time.time() - start)
            all_test_loss, plsp, _ ,_,_ = test(model,test_loader, epoch, device, all_test_loss)
            print("time:", time.time() - start)
            if epoch ==20:
                for param in model.parameters():
                    param.requires_grad = True

                lr = 0.001
    
            if max_plsp < plsp:
                save_nm = 'livew244_deit_enhance_10split'+str(i)+'.pt'
                max_plsp = plsp
                torch.save(model.state_dict(), save_nm)
                ct = 0
    
            if epoch  ==40:
                lr= 0.005
            if epoch == 60:
                lr = 0.03
                ct = 1
    
            if ct > 20 and epoch > 60:
                model.load_state_dict(torch.load(save_nm))
                lr *= 0.3
                ct = 0
                if lr<5e-5:
                    model.load_state_dict(torch.load(save_nm))
                    all_test_loss, plsp,sp,lowpl,lowsr = test(model, test_loader, epoch, device, all_test_loss)
                    best_plccs.append(plsp)
                    best_srccs.append(sp)
                    best_low_plccs.append(lowpl)
                    best_low_srccs.append(lowsr)
                    print('Split:', i, 'End!','PLCC:',best_plccs,'SRCC:',best_srccs,'Low_PLCC:',best_low_plccs,'Low_SRCC:',best_low_srccs)
                    break

if __name__ == '__main__':
    main()

