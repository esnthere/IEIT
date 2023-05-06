# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from scipy import io as sio
import torch.utils.data
import torchvision.models as models
import pandas as pd
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader
import os
import torchvision.transforms.functional as tf
import cv2
from functools import partial
import copy
from my_vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model



class Mydataset(Dataset):
    def __init__(self, imgs, imgs2, labels):
        self.imgs = imgs
        self.imgs2 = imgs2
        self.labels = torch.FloatTensor(labels)

    def __getitem__(self, index):
        return torch.from_numpy(self.imgs[index]), torch.from_numpy(self.imgs2[index]),  self.labels[index]
    def __len__(self):
        return (self.imgs).shape[0]



def test(model,  test_loader, epoch, device, all_test_loss):
    model.eval()

    op = []
    tg = []
    with torch.no_grad():
        for batch_idx, (data, data2, target ) in enumerate(test_loader):
            data, data2, target= data.to(device), data2.to(device), target.to(device)

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


            output1 = model(data, data2)

            op = np.concatenate((op, output1[:, 0].cpu().numpy()))
            tg = np.concatenate((tg, target[:, 0].cpu().numpy()))

    print('Test ALL Pearson:', pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="pearson") )
    print('Test  ALL Spearman:', pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="spearman"))
    ind = np.where(tg < 2.4)[0]
    print('Low Quality Pearson:', pd.Series((op[ind])).corr((pd.Series(tg[ind])), method="pearson"))
    print('Low Quality Spearman:', pd.Series((op[ind])).corr((pd.Series(tg[ind])), method="spearman"))

    return



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


        op1 = self.model3(torch.cat((opf1,opf2_1.detach(),opf2_2.detach()), 1))

        return op1


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'
    device = torch.device("cuda")

    model1 = VisionTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
                               norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model1.default_cfg = _cfg()

    model2 = copy.deepcopy(model1)

    model3 = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(384 * 3, 64),
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

    model = MyModel3(model1, model2, model3, model4)

    model = nn.DataParallel(model.to(device), device_ids=[0])
    model.load_state_dict(torch.load('IEIT_cid2013.pt'))

    batch_size = 64*2
    num_workers_test = 0
################################################################
    all_data = sio.loadmat('E:\Database\CID2013\\cid_244.mat')
    X = all_data['X']
    Y = all_data['Y']
    Y = (Y + 10) / 25 + 1  # It should be Y=Y/25+4 . If Y = (Y+10) /25+ 1  , the low-quality images should be Y<2.4 instead of Y<2!!!!
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest']
    Ytest = (Ytest + 10) / 25 + 1
    del all_data

    all_data = sio.loadmat('E:\Database\CID2013\\cid_244_brt_debulr.mat')
    X2 = all_data['X']
    Xtest2 = all_data['Xtest']
    del all_data

    all_data = np.load('cid_prob_compare.npz')
    op_train = all_data['op_train']
    op_train2 = all_data['op_train2']
    op_test = all_data['op_test']
    op_test2 = all_data['op_test2']
    ind_train = np.where((np.max(op_train2, axis=1) - np.max(op_train, axis=1)) < 0)[0]
    ind_test = np.where((np.max(op_test2, axis=1) - np.max(op_test, axis=1)) < 0)[0]
    X2[ind_train] = X[ind_train]
    Xtest2[ind_test] = Xtest[ind_test]

    for i in range(0,3):
        if i>0:
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


    test_dataset = Mydataset(Xtest, Xtest2,Ytest)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers_test,pin_memory=True)

    print("CID2013 Test Results:")
    all_test_loss = []
    test(model, test_loader, -1, device, all_test_loss)


if __name__ == '__main__':
    main()

