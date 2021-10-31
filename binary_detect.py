import torch
import torch_geometric #torch_geometric == 2.5
import community
import numpy as np
import networkx
import argparse
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.data import Batch
from Sign_OPT import *
import torch_geometric.transforms as T
from Gin import GIN, SAG, GUNet, GCN
from time import time
from torch_geometric.utils import to_networkx, from_networkx
from random import shuffle

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch GNNs for graph detection')
    parser.add_argument('--method', default='Our')  # 'Our' 'Typo'
    #these are parameters for GIN model
    parser.add_argument('--dataset', type=str, default="COIL-DEL")
    parser.add_argument('--model', type=str, default="GIN")  #'GIN' 'SAG' 'GUNet'
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--id', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32, help='social dataset:64 bio dataset:32')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--pooling_ratio', type=float, default=0.8)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--deepth', type=int, default=3)
    parser.add_argument('--model_path', type=str, default='./detection/')
    args = parser.parse_args()
    return args

def train(model, train_loader, device, lr):
    model.train()
    loss_all = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_loader.dataset)

def Train(model, path, train_loader,val_loader, num_epochs, lr, device):
    best_loss, best_val_acc, best_epoch = 0, 0, 0
    for epoch in range(num_epochs):
        if epoch+1 % 50 == 0:
            lr = lr*0.5
        loss = train(model, train_loader, device, lr)
        train_acc = test(model, train_loader, device)
        val_acc = test(model, val_loader, device)
        if val_acc >= best_val_acc:
            best_loss, best_val_acc, best_epoch = loss, val_acc, epoch
            torch.save(model.state_dict(), path)
       # print('Epoch:{:03d}, Loss:{:04f}, Train acc:{:04f}, Val acc:{:04f}'.format(epoch,loss,train_acc, val_acc))
    print('best loss:{:04f}, best acc:{:04f}, epoch:{:03d}'.format(best_loss,best_val_acc,best_epoch))
    
def test(model, test_loader, device):
    model.eval()
    correct = 0
    for data in test_loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(test_loader.dataset)

def Test(model, path, device, test_normal_list, test_advers_list, PR, save_path):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    PR = [x for x in PR if x>0]
    #print('length of PR: {}'.format(len(PR)))
    #print('length of test list: {}'.format(len(test_advers_list)))
    
    FPR = []
    FNR = []
    Acc = []
    Pre = []
    Recall = []
    F1 = []
    for b in range(1,21):
        budget = b/100
        pr_index = [x for x in list(range(len(PR))) if PR[x] <= budget]
        fpr, fnr, acc, pre, recall, f1 = 0,0,0,0,0,0
        if pr_index:
            TP, TN, FP, FN = 0,0,0,0
            for i in pr_index:
                normal = test_normal_list[i]
                advers = test_advers_list[i]
                if model.predict(normal, device) == normal.y[0]:
                    TN += 1
                else:
                    FP += 1
                if model.predict(advers, device) == advers.y[0]:
                    TP += 1
                else:
                    FN += 1
            acc = (TP+TN) / (TP+TN+FP+FN)
            if (TP+FP) > 0:
                pre = TP / (TP+FP)
            if (TP+FN) > 0:
                recall = TP / (TP+FN) 
            if (pre + recall) > 0:
                f1 = 2*pre*recall / (pre + recall)
            if (FP+TN) > 0:
                fpr = FP / (FP + TN)
            if (TP+FN) > 0:
                fnr = FN / (TP + FN)
        FPR.append(fpr)
        FNR.append(fnr)
        Acc.append(acc)
        Pre.append(pre)
        Recall.append(recall)
        F1.append(f1)

    with open(save_path, 'w') as f:
        f.write('FPR'+'-'*20+'\n')
        for i in FPR:
            f.write('{:.4f}\n'.format(i))
        f.write('FNR'+'-'*20+'\n')
        for i in FNR:
            f.write('{:.4f}\n'.format(i))
        f.write('Acc'+'-'*20+'\n')
        for i in Acc:
            f.write('{:.4f}\n'.format(i))
        f.write('Pre'+'-'*20+'\n')
        for i in Pre:
            f.write('{:.4f}\n'.format(i))
        f.write('Recall'+'-'*20+'\n')
        for i in Recall:
            f.write('{:.4f}\n'.format(i))
        f.write('F1'+'-'*20+'\n')
        for i in F1:
            f.write('{:.4f}\n'.format(i))
TUD = {'NCI1':0,'COIL-DEL':0,'IMDB-BINARY':1}

if __name__ == '__main__':
    args = get_args()
    dataset_name = args.dataset
    model_name = args.model
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else torch.device("cpu"))
    batch_size = args.batch_size
    num_epochs = args.epochs
    lr = args.learning_rate
    hidden_dim = args.hidden_dim
    method = args.method
    dropout = args.dropout
    model_path = args.model_path
    pooling_ratio = args.pooling_ratio
    deepth = args.deepth

    detect_path = './detection/{}_{}_'.format(dataset_name, method)
    train_normal_list = torch.load(detect_path+'train_normal.pt')
    train_advers_list = torch.load(detect_path+'train_advers.pt')
    test_path = './detection/{}_{}_'.format(dataset_name, 'Our')
    test_normal_list = torch.load(test_path+'test_normal.pt')
    test_advers_list = torch.load(test_path+'test_advers.pt')
    
    #print('length of train dataset: {}'.format(len(train_advers_list)))
    train_normal_list.extend(train_advers_list)
    shuffle(train_normal_list)
    n = len(train_normal_list) // 5
    val_list = train_normal_list[:n]
    train_list = train_normal_list[n:]
    #print(len(val_list))
    #print(len(train_list))
    train_loader = DataLoader(train_list, batch_size=batch_size)
    val_loader = DataLoader(val_list, batch_size=batch_size)
    #print('load dataset done!')

    input_dim = train_list[0].num_features
    output_dim = 2
    #print('input dim:', input_dim)
    #print('output dim:', output_dim)
    
    our_path = './out/our_{}_{}_{}_{}_'.format(dataset_name,1,1,1)
    with open(our_path+'PR.txt', 'r') as f:
        PR = eval(f.read())
    if model_name == "GIN":
        model = GIN(5,2,input_dim,hidden_dim,output_dim,dropout).to(device)
    if model_name == "SAG":
        model = SAG(5,input_dim,hidden_dim,output_dim,pooling_ratio,dropout).to(device)
    if model_name == "GUNet":
        model = GUNet(input_dim,hidden_dim,output_dim,pooling_ratio,deepth,dropout).to(device)
    if model_name == "GCN":
        model = GCN(5,input_dim, hidden_dim, output_dim, dropout).to(device)
    
    path = model_path + '{}_{}_{}.pt'.format(dataset_name, method, model_name)
    save_path = './detection/out/{}_{}_{}.txt'.format(dataset_name, model_name, args.id)
    Train(model, path, train_loader,val_loader, num_epochs, lr, device)
    Test(model, path, device, test_normal_list, test_advers_list, PR, save_path)

 

