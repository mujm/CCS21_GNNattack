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
from Gin import GIN
from time import time


def get_args():
    parser = argparse.ArgumentParser(description='Pytorch graph isomorphism network for graph classification')
    #these are parameters for attack model
    parser.add_argument('--svm', type=int, default=0)
    parser.add_argument('--max_query', type=int, default=40000)
    parser.add_argument('--effective', type=int, default=1)
    #these are parameters for GIN model
    parser.add_argument('--dataset', type=str, default="IMDB-BINARY")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32, help='social dataset:64 bio dataset:32')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--model_path', type=str, default='./trained_model/')
    args = parser.parse_args()
    return args

def distance(x_adv, x):
    adj_adv = nx.adjacency_matrix(to_networkx(x_adv, to_undirected=True))
    adj_x = nx.adjacency_matrix(to_networkx(x, to_undirected=True))
    return np.sum(np.abs(adj_adv-adj_x)) / 2

TUD = {'NCI1':0,'COIL-DEL':0,'IMDB-BINARY':1}
Num = {'NCI1':318,'COIL-DEL':304,'IMDB-BINARY':77}
if __name__ == '__main__':

    args = get_args()
    dataset_name = args.dataset
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else torch.device("cpu"))
    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    dropout = args.dropout
    model_path = args.model_path
    
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
    
    n = (len(dataset) // 10) * 2
    
    print('length of training dataset of detection classifier:', n)

    index_path = './data_split/' + dataset_name + '_'
    with open(index_path+'train_index.txt', 'r') as f:
        train_index = eval(f.read())
    #detect_train_index = random.sample(train_index, n)
    train_dataset = dataset[train_index]
    #index_path = './detection/' + dataset_name+'_'
    #with open(index_path+'train_index.txt', 'w') as f:
       # f.write(str(detect_train_index)) 
    input_dim = dataset.num_node_features
    output_dim = dataset.num_classes
    print('input dim: ', input_dim)
    print('output dim: ', output_dim)
    model = GIN(5,2,input_dim,hidden_dim,output_dim,dropout).to(device)
    load_path = model_path + dataset_name + '.pt'
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.eval()

    attacker = OPT_attack_sign_SGD(model, device, args.effective)
    num_train = len(train_dataset)
    perturbation = [] #perturbation for each poisoned graph
    perturbation_ratio = [] #perturbation ratio for each poisoned graph
    success_index = [] 
    success_count = 0
    no_need_count = 0
    num_query = []    
    fail_count = 0
    distortion = []
    attack_time = []
    
    detect_train_normal = []
    detect_train_advers = []
    count = 0
    for i in range(num_train):
        print('begin to attack instance {}'.format(i))
        x0 = train_dataset[i].to(device)
        y0 = x0.y[0]
        y1 = model.predict(x0, device)
        num_nodes = x0.num_nodes
        space = num_nodes * (num_nodes - 1) / 2

        if y0 == y1:
            time_start = time()
            adv_x0, adv_y0, query, success, dis, init = attacker.attack_untargeted(x0, y0, svm=args.svm, query_limit=args.max_query)
            time_end = time()
            attack_time.append(time_end-time_start)
            num_query.append(query)
            if success :
                perturb = distance(adv_x0, x0)
                success_count += 1
                perturbation.append(perturb)
                perturbation_ratio.append(perturb/space)
                distortion.append(dis)
                x0.y = torch.tensor([0])
                adv_x0.y = torch.tensor([1])
                detect_train_normal.append(x0)
                detect_train_advers.append(adv_x0)
                count += 1
            else:
                fail_count += 1
        else:
            print('instance {} is wrongly classified, No Need to Attack'.format(i))
            no_need_count += 1
            num_query.append(0)

        if count == 3*Num[dataset_name]:
            break
    '''
    print('{} instances don\'t need to be attacked'.format(no_need_count))
    print('Sign-Opt fails to attack {} instance'.format(fail_count))
    success_ratio = success_count / (num_train - no_need_count)
    avg_perturbation = sum(perturbation) / success_count
    print("Sign-Opt: the success rate of black-box attack is {}/{} = {:.4f}".format(success_count,num_train-no_need_count, success_ratio))
    print('Sign-Opt: the average perturbation is {:.4f}'.format(avg_perturbation))
    print('Sign-Opt: the average perturbation ratio is {:.4f}'.format(sum(perturbation_ratio) / success_count))
    print('Sign-Opt: the average query count is {:.4f}'.format(sum(num_query)/(num_train-no_need_count)))
    print('Sign-Opt: the average attacking time is {:.4f}'.format(sum(attack_time)/(num_train-no_need_count)))
    print('Sign-Opt: the average distortion is {:.4f}'.format(sum(distortion)/success_count))
    print('Sign-Opt: detail perturbation are: {}'.format(perturbation))
    print('Sign-Opt: detail perturbation ratio are: {}'.format(perturbation_ratio))
    print('dataset: {}'.format(dataset_name))
    '''
    detect_train_path = './detection/'+dataset_name+'_Our_'
    torch.save(detect_train_normal, detect_train_path+'train_normal.pt')
    torch.save(detect_train_advers,detect_train_path+'train_advers.pt')
    
    '''
    query_path = './detection/' + dataset_name + '_query.txt'
    with open(query_path, 'w') as f:
        f.write(str(num_query))
    '''
    '''
    out_path = './detection/{}_Opt_{}.txt'.format(dataset_name, bound)  
    with open(out_path, 'w') as f:
        f.write('{} instances don\'t need to be attacked\n'.format(no_need_count))
        f.write('Sign-Opt fails to attack {} instance\n'.format(fail_count))
        f.write("Sign-Opt: the success rate of black-box attack is {}/{} = {:.4f}\n".format(success_count,num_train-no_need_count, success_ratio))
        f.write('Sign-Opt: the average perturbation is {:.4f}\n'.format(avg_perturbation))
        f.write('Sign-Opt: the average perturbation ratio is {:.4f}\n'.format(sum(perturbation_ratio) / success_count*100))
        f.write('Sign-Opt: the average query count is {:.4f}\n'.format(sum(num_query)/(num_train-no_need_count)))
        f.write('Sign-Opt: the average attacking time is {:.4f}\n'.format(sum(attack_time)/(num_train-no_need_count)))
        f.write('Sign-Opt: the average distortion is {:.4f}\n'.format(sum(distortion)/success_count))
        f.write('Sign-Opt: detail perturbation are: {}\n'.format(perturbation))
        f.write('Sign-Opt: detail perturbation ratio are: {}\n'.format(perturbation_ratio))
    '''
