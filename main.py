import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

import torch_geometric.transforms as T
from torch_geometric.nn import GINConv, global_add_pool
import argparse
import numpy as np
from random import shuffle
from Gin import GIN, GCN, SAG, GUNet

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch graph isomorphism network for graph classification')
    parser.add_argument('--dataset', type=str, default="IMDB-BINARY")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32, help='social dataset:64 bio dataset:32')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--model_path', type=str, default='./trained_model/')
    parser.add_argument('--model', type=str, default='GUN')
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

def test(model, test_loader, device):
    model.eval()
    correct = 0
    for data in test_loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(test_loader.dataset)
TUD = {'MUTAG':0,
        'PTC_FM':0,
        'PROTEINS':0,
        'NCI1':0,
        'COIL-DEL':0,
        'COLLAB':1,
        'IMDB-BINARY':1,
        'IMDB-MULTI':1,
        'REDDIT-BINARY':1,
        'REDDIT-MULTI5K':1}
max_degree = {'COLLAB':491,
              'IMDB-BINARY':135,
              'IMDB-MULTI':88}

if __name__ == '__main__':
    args = get_args()
    dataset_name = args.dataset
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else torch.device("cpu"))
    batch_size = args.batch_size
    num_epochs = args.epochs
    lr = args.learning_rate
    hidden_dim = args.hidden_dim
    dropout = args.dropout
    model_path = args.model_path
    model_name = args.model

    if dataset_name in TUD.keys():
        degree_as_attr = TUD[dataset_name]
    else:
        print('invalid dataset!')
        raise(ValueError)

    print(device)
    if degree_as_attr:
        dataset = TUDataset(root='./dataset',name=dataset_name,use_edge_attr='False', use_node_attr=True,
        pre_transform=T.Constant(1,True))
    else:
        dataset = TUDataset(root='./dataset',name=dataset_name,use_edge_attr='False', use_node_attr=True)
 

    print(len(dataset))
    index = list(range(len(dataset)))
    shuffle(index)
    n = len(dataset) // 10
    test_index = index[:n]
    train_index = index[n:]

    index_path = './data_split/' + dataset_name + '_'
    with open(index_path+'train_index.txt', 'w') as f:
        f.write(str(train_index))
    with open(index_path+'test_index.txt', 'w') as f:
        f.write(str(test_index))

    test_dataset = dataset[test_index]
    train_dataset = dataset[train_index]
    test_loader = DataLoader(test_dataset, batch_size=32)
    train_loader = DataLoader(train_dataset, batch_size=32)

    input_dim = dataset.num_node_features
    output_dim = dataset.num_classes
    print('input dim: ', input_dim)
    print('output dim: ', output_dim)
    if model_name=='SAG':
        model = SAG(5,input_dim,hidden_dim,output_dim,0.8,dropout).to(device)
    elif model_name=='GIN':
        model = GIN(5,2,input_dim,hidden_dim,output_dim,dropout).to(device)
    elif model_name=='GUN':
        model = GUNet(input_dim,hidden_dim,output_dim,0.8,3,dropout).to(device)
    

    path = model_path + '{}_{}.pt'.format(dataset_name, model_name)

    best_loss, best_test_acc, best_epoch = 0, 0, 0
    for epoch in range(num_epochs):
        if epoch+1 % 50 == 0:
            lr = lr*0.5
        loss = train(model, train_loader, device, lr)
        train_acc = test(model, train_loader, device)
        test_acc = test(model, test_loader, device)
        if test_acc >= best_test_acc:
            best_loss, best_test_acc, best_epoch = loss, test_acc, epoch
            torch.save(model.state_dict(), path)
        print('Epoch:{:03d}, Loss:{:04f}, Train acc:{:04f}, Test acc:{:04f}'.format(epoch,loss,train_acc,test_acc))
    print('best loss:{:04f}, best acc:{:04f}, epoch:{:03d}'.format(best_loss,best_test_acc,best_epoch))
