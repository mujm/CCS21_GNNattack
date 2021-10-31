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
from time import time
from test import distance
from Gin import GIN, SAG, GUNet

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch graph isomorphism network for graph classification')
    #these are parameters for attack model
    parser.add_argument('--max_query', type=int, default=40000)
    parser.add_argument('--bound', type=float, default=1)
    parser.add_argument('--effective', type=int, default=1)
    parser.add_argument('--id', type=int, default=1)
    parser.add_argument('--search', type=int, default=1)
    #these are parameters for GIN model
    parser.add_argument('--dataset', type=str, default="IMDB-BINARY")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--model_path', type=str, default='./trained_model/')
    parser.add_argument('--model', type=str, default='SAG')
    args = parser.parse_args()
    return args

TUD = {'NCI1':0,'COIL-DEL':0,'IMDB-BINARY':1}

def Random_attack(x0, y0, model, device, count):
    model.to(device)
    model.eval()
    num_nodes = x0.num_nodes
    G = to_networkx(x0, to_undirected=True)
    perturb = float('inf')
    perturb = torch.FloatTensor([perturb]).to(device)
    x_new = copy.deepcopy(x0).to(device)
    adv_x = None
    for k in range(count):
        theta = torch.normal(mean=torch.rand(1).item(), std=0.5, size=(num_nodes,num_nodes)).to(device)
        G_new = new_graph(G, theta)
        x_new.edge_index = from_networkx(G_new).to(device).edge_index.long()
        if model.predict(x_new, device) != y0:
            dis = distance(x_new, x0)
            if dis < perturb:
                perturb = dis
                adv_x = copy.deepcopy(x_new).to(device)
    if adv_x == None:
        print('Random attack failed!')
        return x0, y0, False, -1
    else:
        adv_y = model.predict(adv_x, device)
        return adv_x, adv_y, True, perturb

if __name__ == '__main__':

    args = get_args()
    dataset_name = args.dataset
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else torch.device("cpu"))
    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    bound = args.bound
    dropout = args.dropout
    model_path = args.model_path
    model_name = args.model
    
    if dataset_name in TUD.keys():
        degree_as_attr = TUD[dataset_name]
    else:
        print('invalid dataset!')
        raise(ValueError)

    if degree_as_attr:
        dataset = TUDataset(root='./dataset',name=dataset_name,use_edge_attr='False', use_node_attr=True,
        pre_transform=T.Constant(1, True))
    else:
        dataset = TUDataset(root='./dataset',name=dataset_name,use_edge_attr='False',use_node_attr=True)
    
    index_path = './data_split/' + dataset_name + '_'
    with open(index_path+'test_index.txt', 'r') as f:
        test_index = eval(f.read())
    test_dataset = dataset[test_index]
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
    load_path = model_path + '{}_{}.pt'.format(dataset_name, model_name)
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.eval()

    our_path = './out1/our_{}_{}_{}_{}_'.format(dataset_name, args.id, args.effective, args.search)
    with open(our_path+'Q.txt', 'r') as f:
        num_query = eval(f.read())
    
    assert len(num_query) == len(test_dataset)
    num_test = len(num_query)
    perturbation = [] #perturbation for each poisoned graph
    perturbation_ratio = []
    success_index = [] 
    success_count = 0
    no_need_count = 0
    fail_count = 0
    distortion = []
    attack_time = []

    for i in range(num_test):
        print('begin to attack instance {}'.format(i))
        x0 = test_dataset[i].to(device)
        y0 = x0.y[0]
        count = num_query[i]
        num_nodes = x0.num_nodes
        space = num_nodes * (num_nodes - 1) / 2
        max_perturb = bound * space
        if count > 0:
            time_start = time()
            adv_x0, adv_y0, success, perturb = Random_attack(x0, y0, model, device, count)
            time_end = time()
            if success:
                perturbation.append(perturb)
                attack_time.append(time_end-time_start)
                perturbation_ratio.append(perturb/space)
            else:
                fail_count += 1
                perturbation.append(-1)
                attack_time.append(-1)
                perturbation_ratio.append(-1)
        else:
            print('instance {} is wrongly classified, No Need to Attack'.format(i))
            no_need_count += 1 
            perturbation.append(0)
            attack_time.append(0)
            perturbation_ratio.append(0)
    '''
    print('{} instances don\'t need to be attacked'.format(no_need_count))
    print('Random fails to attack {} instance'.format(fail_count))
    success_ratio = success_count / (num_test - no_need_count)*100
    avg_perturbation = sum(perturbation) / success_count
    print("Random: the success rate of black-box attack is {}/{} = {}".format(success_count,num_test-no_need_count, success_ratio))
    print('Random: the average perturbation is {}'.format(avg_perturbation))
    print('Random: the average perturbation ratio is {}'.format(sum(perturbation_ratio) / success_count*100))
    print('Random: the average attacking time is {}'.format(sum(attack_time)/success_count))
    print('Random: detail perturbation are: {}'.format(perturbation))
    print('Random: detail perturbation ratio are: {}'.format(perturbation_ratio))
    print('dataset: {}'.format(dataset_name))
    '''

    random_path = './out1/ran_{}_{}_{}_{}_'.format(dataset_name, args.id, args.effective, args.search)
    with open(random_path+'P.txt', 'w') as f:
        f.write(str(perturbation))
    with open(random_path+'PR.txt', 'w') as f:
        f.write(str(perturbation_ratio))
    with open(random_path+'T.txt', 'w') as f:
        f.write(str(attack_time))