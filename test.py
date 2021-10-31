import torch
import torch_geometric #torch_geometric == 1.6.1
import community
import numpy as np
import networkx
import argparse
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.data import Batch
from Sign_OPT import *
from torch_geometric.utils import to_networkx, from_networkx
import torch_geometric.transforms as T
from Gin import GIN, SAG, GUNet
from time import time

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch graph isomorphism network for graph classification')
    #these are parameters for attack model

    parser.add_argument('--effective', type=int, default=1)
    parser.add_argument('--max_query', type=int, default=40000)
    parser.add_argument('--id', type= int, default=1)
    parser.add_argument('--search', type=int, default=1)
    #these are parameters for GIN model
    parser.add_argument('--dataset', type=str, default="IMDB-BINARY")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32, help='social dataset:64 bio dataset:32')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--model_path', type=str, default='./trained_model/')
    parser.add_argument('--model', type=str, default='GUN')
    args = parser.parse_args()
    return args

def distance(x_adv, x):
    adj_adv = nx.adjacency_matrix(to_networkx(x_adv, to_undirected=True))
    adj_x = nx.adjacency_matrix(to_networkx(x, to_undirected=True))
    return np.sum(np.abs(adj_adv-adj_x)) / 2
    
def count_edges(x_adv, x):
    adj_adv = nx.adjacency_matrix(to_networkx(x_adv, to_undirected=True)).todense().A
    adj_x = nx.adjacency_matrix(to_networkx(x, to_undirected=True)).todense().A
    difference = adj_adv - adj_x
    num_add = sum(sum(difference==1)) / 2
    num_delete = sum(sum(difference==-1)) / 2
    return num_add, num_delete

TUD = {'NCI1':0,'COIL-DEL':0,'IMDB-BINARY':1}

if __name__ == '__main__':

    args = get_args()
    dataset_name = args.dataset
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else torch.device("cpu"))
    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
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
        load_path = model_path + '{}_{}.pt'.format(dataset_name, model_name)
    elif model_name=='GIN':
        model = GIN(5,2,input_dim,hidden_dim,output_dim,dropout).to(device)
        load_path = model_path + '{}.pt'.format(dataset_name)
    elif model_name=='GUN':
        model = GUNet(input_dim,hidden_dim,output_dim,0.8,3,dropout).to(device)
        load_path = model_path + '{}_{}.pt'.format(dataset_name, model_name)
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.eval()

    attacker = OPT_attack_sign_SGD(model, device, args.effective)
    num_test = len(test_dataset)
    perturbation = [] #perturbation for each poisoned graph
    perturbation_ratio = [] #perturbation ratio for each poisoned graph

    no_need_count = 0
    num_query = []    
    fail_count = 0
    distortion = []
    attack_time = []

    init_perturbation = [] #perturbation for each poisoned graph
    init_perturbation_ratio = [] #perturbation ratio for each poisoned graph
    init_num_query = []    
    init_distortion = []
    init_attack_time = []
    search_type = []

    detect_test_normal = []
    detect_test_advers = []

    num_add_edge, num_delete_edge = [], []
    for i in range(num_test):
        print('begin to attack instance {}'.format(i))
        x0 = test_dataset[i].to(device)
        y0 = x0.y[0]
        y1 = model.predict(x0, device)
        num_nodes = x0.num_nodes
        space = num_nodes * (num_nodes - 1) / 2
        if y0 == y1:
            time_start = time()
            adv_x0, adv_y0, query, success, dis, init = attacker.attack_untargeted(x0, y0, query_limit=args.max_query)
            time_end = time()
            init_num_query.append(init[2])
            num_query.append(query)
            init_attack_time.append(init[3])
            attack_time.append(time_end-time_start)
            if success:
                #process results in Stage 1
                init_perturb, init_dis, init_query, init_time, s_type = init
                init_ratio = init_perturb / space
                init_perturbation.append(init_perturb)
                init_distortion.append(init_dis)
                search_type.append(s_type)
                init_perturbation_ratio.append(init_ratio)

                #process results in Stage 2
                perturb = distance(adv_x0, x0)
                perturbation.append(perturb)
                perturbation_ratio.append(perturb/space)
                distortion.append(dis)
                
                add_edge, delete_edge = count_edges(adv_x0, x0)
                num_delete_edge.append(delete_edge)
                num_add_edge.append(add_edge)

                #test dataset for defense
                #x0.y = torch.tensor([0])
                #adv_x0.y = torch.tensor([1])
                adv_x0.y = x0.y
                detect_test_advers.append(adv_x0)
                detect_test_normal.append(x0)
            else:
                detect_test_advers.append(x0)
                detect_test_normal.append(x0)
                init_distortion.append(-1)
                init_perturbation.append(-1)
                init_perturbation_ratio.append(-1)
                search_type.append(-1)

                perturbation.append(-1)
                perturbation_ratio.append(-1)
                distortion.append(-1) 
        else:
            print('instance {} is wrongly classified, No Need to Attack'.format(i))
            no_need_count += 1
            num_query.append(0)
            attack_time.append(0)
            perturbation.append(0)
            perturbation_ratio.append(0)
            distortion.append(0)
            
            init_perturbation.append(0)
            init_distortion.append(0)
            init_num_query.append(0)
            init_attack_time.append(0)
            search_type.append(0)
            init_perturbation_ratio.append(0)

    
    print('{} instances don\'t need to be attacked'.format(no_need_count))
    '''
    success_ratio = success_count / (num_test - no_need_count)*100
    avg_perturbation = sum(perturbation) / success_count
    print("Sign-Opt: the success rate of black-box attack is {}/{} = {:.4f}".format(success_count,num_test-no_need_count, success_ratio))
    print('Sign-Opt: the average perturbation is {:.4f}'.format(avg_perturbation))
    print('Sign-Opt: the average perturbation ratio is {:.4f}'.format(sum(perturbation_ratio) / success_count*100))
    print('Sign-Opt: the average query count is {:.4f}'.format(sum(num_query)/(num_test-no_need_count)))
    print('Sign-Opt: the average attacking time is {:.4f}'.format(sum(attack_time)/(num_test-no_need_count)))
    print('Sign-Opt: the average distortion is {:.4f}'.format(sum(distortion)/success_count))
    print('dataset: {}'.format(dataset_name))
    '''
    if args.search == 1 and args.effective == 1 and args.id ==1: 
        detect_test_path = './defense/'+dataset_name+'_'+model_name+'_Our_'
        torch.save(detect_test_normal, detect_test_path+'test_normal.pt')
        torch.save(detect_test_advers, detect_test_path+'test_advers.pt')
        print('test dataset for defense saved!')
  
    
    init_path = './out1/init_{}_{}_{}_{}_'.format(dataset_name, args.id, args.effective , args.search)
    with open(init_path+'search_type.txt', 'w') as f:
        f.write(str(search_type))
    with open(init_path+'P.txt', 'w') as f:
        f.write(str(init_perturbation))
    with open(init_path+'PR.txt', 'w') as f:
        f.write(str(init_perturbation_ratio))
    with open(init_path+'D.txt', 'w') as f:
        f.write(str(init_distortion))
    with open(init_path+'Q.txt', 'w') as f:
        f.write(str(init_num_query))
    with open(init_path+'T.txt', 'w') as f:
        f.write(str(init_attack_time))  
    
    
    our_path = './out1/our_{}_{}_{}_{}_'.format(dataset_name, args.id, args.effective , args.search)
    with open(our_path+'Q.txt', 'w') as f:
        f.write(str(num_query))
    with open(our_path+'T.txt', 'w') as f:
        f.write(str(attack_time))
    with open(our_path+'P.txt', 'w') as f:
        f.write(str(perturbation))
    with open(our_path+'PR.txt', 'w') as f:
        f.write(str(perturbation_ratio))
    with open(our_path+'D.txt', 'w') as f:
        f.write(str(distortion))
    with open(our_path+'ADD.txt', 'w') as f:
        f.write(str(num_delete_edge))
    with open(our_path+'DEL.txt', 'w') as f:
        f.write(str(num_add_edge))
            
    print("the numbers of deleted edges are:", num_delete_edge)
    print("the numbers od added edges are:", num_add_edge)
    print("the average number of deleted edges for %s: %d"%(dataset_name, float(sum(num_delete_edge)/len(num_delete_edge))))
    print("the average number of added edges for %s: %d"%(dataset_name, float(sum(num_add_edge)/len(num_add_edge))))
    '''
    out_path = './out/{}_Opt_{}.txt'.format(dataset_name, bound)  
    with open(out_path, 'w') as f:
        f.write('{} instances don\'t need to be attacked\n'.format(no_need_count))
        f.write('Sign-Opt fails to attack {} instance\n'.format(fail_count))
        f.write("Sign-Opt: the success rate of black-box attack is {}/{} = {:.4f}\n".format(success_count,num_test-no_need_count, success_ratio))
        f.write('Sign-Opt: the average perturbation is {:.4f}\n'.format(avg_perturbation))
        f.write('Sign-Opt: the average perturbation ratio is {:.4f}\n'.format(sum(perturbation_ratio) / success_count*100))
        f.write('Sign-Opt: the average query count is {:.4f}\n'.format(sum(num_query)/(num_test-no_need_count)))
        f.write('Sign-Opt: the average attacking time is {:.4f}\n'.format(sum(attack_time)/(num_test-no_need_count)))
        f.write('Sign-Opt: the average distortion is {:.4f}\n'.format(sum(distortion)/success_count))
        f.write('Sign-Opt: detail perturbation are: {}\n'.format(perturbation))
        f.write('Sign-Opt: detail perturbation ratio are: {}\n'.format(perturbation_ratio))
    '''
    
