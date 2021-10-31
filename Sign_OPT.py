'''
refer from https://github.com/LeMinhThong/blackbox-attack
'''
from time import time
import numpy as np 
from numpy import linalg as LA
import torch
import scipy.spatial
import copy
from scipy.linalg import qr
import random
import networkx as nx
from networkx.classes.function import is_path
import community
import torch_geometric
from torch_geometric.utils import to_networkx, from_networkx
import matplotlib.pyplot as plt
from test import distance

def L0_norm(theta, device):
    theta[theta==float('inf')]=1
    theta[theta==float('-inf')]=1
    theta = torch.triu(theta, diagonal=1).to(device)  
    theta = torch.where(theta>0.5, torch.FloatTensor([1]).to(device), torch.FloatTensor([0]).to(device))
    return torch.sum(theta)

def L1_norm(theta, device):
    theta[theta==float('inf')]=1
    theta[theta==float('-inf')]=1
    theta = torch.triu(theta-0.5, diagonal=1).to(device)
    theta = torch.clamp(theta, 0.0, 1.0)
    return torch.sum(theta)

def new_graph(G, perturb, index1=None, index2=None):
    #generate a perturbed graph
    G_new = copy.deepcopy(G)
    if index2 == None:
        if index1 == None:  #change the edges in the whole graph
            num_nodes = len(G_new)
            for i in range(num_nodes-1):
                for j in range(i+1, num_nodes):
                    if perturb[i, j] > 0.5:  #we will change this link
                        if is_path(G_new, [i, j]):
                            G_new.remove_edge(i, j)
                        else:
                            G_new.add_edge(i, j)
        else:  #change the edges in one cluster
            num_nodes = len(index1)
            for i in range(num_nodes-1):
                for j in range(i+1, num_nodes):
                    if perturb[i, j] > 0.5:  #we will change this link
                        if is_path(G_new, [index1[i], index1[j]]):
                            G_new.remove_edge(index1[i], index1[j])
                        else:
                            G_new.add_edge(index1[i], index1[j])
    else:  #change the edges between clusters
        num_nodes1, num_nodes2 = len(index1), len(index2)
        for i in range(num_nodes1):
            for j in range(num_nodes2):
                if perturb[i, j] > 0.5:
                    if is_path(G_new, [index1[i], index2[j]]):
                        G_new.remove_edge(index1[i], index2[j])
                    else:
                        G_new.add_edge(index1[i], index2[j])
    return G_new 

def quad_solver(Q, b, device):
    """
    Solve min_a  0.5*aQa + b^T a s.t. a>=0
    """
    #Q: torch(M,N)
    #b: torch([K])
    K = Q.shape[0]
    alpha = torch.zeros((K,)).to(device)
    g = b
    Qdiag = torch.diag(Q).to(device)
    for i in range(20000):
        delta = torch.max(alpha - torch.div(g,Qdiag),torch.zeros_like(alpha)) - alpha
        idx = torch.argmax(torch.abs(delta)).item()
        val = delta[idx]

        if torch.abs(val) < 1e-7:
            break
        g = g + val*Q[:,idx]
        alpha[idx] += val
    return alpha.to(device)

class OPT_attack_sign_SGD(object):
    def __init__(self, model, device, effective, Q=100):
        self.model = model
        self.Q = Q
        self.device = device
        self.effective = effective
        model.eval()

    
    def initial_search(self,x0, y0):
        model = self.model
        model.eval()
        device = self.device
        num_query = 0
        num_nodes = x0.num_nodes
        G = to_networkx(x0, to_undirected=True) #tansfer from PyG data to networkx
        partition = community.best_partition(G) #decompose G into clusters
        num_cluster = len(list(set(partition.values())))
        cluster = {}
        for i in range(num_cluster):
            cluster[i] = list(np.where(np.array(list(partition.values())) == i)[0])

        g_theta = float('inf')  #initial g_theta
        g_theta = torch.FloatTensor([g_theta]).to(device)
        F_theta = float('inf')  #initial F_theta
        F_theta = torch.FloatTensor([F_theta]).to(device)
        x_new = copy.deepcopy(x0).to(device)
        flag_inner, flag_outer = 0, 0
        
        #final_theta = torch.zeros((num_nodes, num_nodes)).to(device)
        final_theta = torch.normal(mean=0.5,std=0.1,size=(num_nodes,num_nodes)).to(self.device)
        final_theta = torch.clamp(final_theta, 0.0, 0.5)
        search_type = -1       
        
        #inner cluster perturbation
        for i in range(num_cluster):
            nodes = cluster[i]
            num_cluster_nodes = len(nodes)
            if num_cluster_nodes > 1:
                for j in range(10*num_cluster_nodes): #search initial directions 
                    theta = torch.normal(mean=torch.rand(1).item(),std=0.5,size=(num_cluster_nodes,num_cluster_nodes)).to(self.device)
                    theta = torch.triu(theta, diagonal=1).to(device)
                    G_new = new_graph(G, theta, index1=nodes)
                    x_new.edge_index = from_networkx(G_new).to(device).edge_index.long()
                    if model.predict(x_new, device) != y0:  #we find a direction
                        F_lbd = distance(x_new, x0)
                        if F_lbd < F_theta:
                            F_theta = F_lbd
                            flag_inner = 1
                            search_type = 0
                            for p in range(num_cluster_nodes-1):
                                for q in range(p+1, num_cluster_nodes):
                                    final_theta[nodes[p], nodes[q]] = theta[p, q]
                                    final_theta[nodes[q], nodes[p]] = theta[p, q]   
                    num_query += 1   
        
        ##perturbations between clusters
        if (num_cluster > 1) and (flag_inner == 0):
            for i in range(num_cluster - 1):
                for j in range(i+1, num_cluster):
                    nodes1, nodes2 = cluster[i], cluster[j]
                    num_cluster_nodes1, num_cluster_nodes2 = len(nodes1), len(nodes2)
                    for k in range(10*(num_cluster_nodes1+num_cluster_nodes2)):
                        theta = torch.normal(mean=torch.rand(1).item(), std=0.5, size=(num_cluster_nodes1,num_cluster_nodes2)).to(self.device)
                        G_new = new_graph(G, theta, nodes1, nodes2)
                        x_new.edge_index = from_networkx(G_new).to(device).edge_index.long()
                        if model.predict(x_new, device) != y0:
                            F_lbd = distance(x_new, x0)
                            if F_lbd < F_theta:  
                                F_theta = F_lbd
                                flag_outer = 1
                                search_type = 1
                                for p in range(num_cluster_nodes1):
                                    for q in range(num_cluster_nodes2):
                                        final_theta[nodes1[p], nodes2[q]] = theta[p, q]     
                                        final_theta[nodes2[q], nodes1[p]] = theta[p, q]     
                        num_query += 1   
        
        #perturbations on the whole graph
        if (flag_inner == 0) and (flag_outer == 0):
            for k in range(10*num_nodes):
                theta = torch.normal(mean=torch.rand(1).item(), std=0.5, size=(num_nodes,num_nodes)).to(self.device)
                theta = torch.triu(theta, diagonal=1).to(device)
                G_new = new_graph(G, theta)
                x_new.edge_index = from_networkx(G_new).to(device).edge_index.long()
                if model.predict(x_new, device) != y0:
                    F_lbd = distance(x_new, x0)
                    if F_lbd < F_theta:
                        search_type = 2
                        F_theta = F_lbd
                        #g_theta = lbd
                        final_theta = theta    
                num_query += 1
        
        if F_theta.item() == float('inf'):  #can not find an initial direction
            return final_theta, F_theta, g_theta, num_query, search_type
        else:  #find initial direction
            final_theta = torch.triu(final_theta, diagonal=1).to(device)
            init_lbd_whole = torch.norm(final_theta)
            final_theta_norm = torch.div(final_theta, init_lbd_whole)
            g_theta_whole, c = self.fine_grained_binary_search(model, x0, y0, final_theta_norm, init_lbd_whole, torch.FloatTensor([float('inf')]).to(device))
            F_theta_whole = L1_norm(g_theta_whole*final_theta_norm, self.device)
            return final_theta_norm, F_theta_whole, g_theta_whole, num_query + c, search_type
    
    def test_initial_search(self, x0, y0):
        model = self.model
        query_count = 0
        if (model.predict(x0, self.device) != y0):
            print("Fail to classify the graph. No need to attack.")
            return x0, y0, -1, False, 0
        G0 = to_networkx(x0, to_undirected=True)
        time_start = time()
        initial_theta, inital_F, initial_g, num_query = self.initial_search(x0, y0)
        query_count += num_query
        time_end = time()
        if inital_F.item() == float('inf'):
            print("Couldn't find valid initial direction, failed")
            return x0, y0, query_count, False, 0
        print("==========> Found best initial distortion %.4f perturbation %d in %.4f seconds "
              "using %d queries" % (inital_F.item(), L0_norm(initial_g*initial_theta,self.device),time_end-time_start, query_count))
        G_final = new_graph(G0, torch.clamp(initial_g * initial_theta, 0.0, 1.0))
        x_final = copy.deepcopy(x0).to(self.device)
        x_final.edge_index = from_networkx(G_final).to(self.device).edge_index.long()
        target = model.predict(x_final, self.device)
        #print("Attack succeed: distortion %.4f perturbation %d queries %d \nTime: %.4f seconds" % (g_theta.item(), L0_norm(g_theta*theta,self.device).item(), query_count, time_end-time_start))
        return x_final, target, query_count, True, initial_g.item()

    def attack_untargeted(self, x0, y0, alpha = 0.2, beta = 0.005, iterations = 1000, query_limit=20000,
                          seed=None):
        #this is untargeted attack to GNN model
        #outputs: adv_x, adv_y, num_query, success, g_theta
        #adv_x, adv_y : the perturbed graph
        #num_query : number of total queries
        #success : a bool variable. whether the attack succeed
        #g_theta : distortion between adv_x and x0

        model = self.model
        model.eval()
        query_count = 0
        
        if (model.predict(x0, self.device) != y0):
            print("Fail to classify the graph. No need to attack.")
            return x0, y0, 1, False, 0

        if seed is not None:
            np.random.seed(seed)

        G0 = to_networkx(x0, to_undirected=True)
        time_start = time()
        initial_theta, inital_F, initial_g, num_query, search_type = self.initial_search(x0, y0)
        
        query_count += num_query
        time_end = time()
        if inital_F.item() == float('inf'):
            print("Couldn't find valid initial direction, failed")
            return x0, y0, query_count, False, 0, (0,0,num_query,time_end-time_start,search_type)
        #record the results in Stage 1
        initial_perturb = L0_norm(initial_g*initial_theta, self.device)
        init = (initial_perturb.item(), inital_F.item(), num_query, time_end-time_start, search_type)
        print("==========> Found best initial distortion %.4f perturbation %d in %.4f seconds "
              "using %d queries" % (inital_F.item(), initial_perturb,time_end-time_start, query_count))
        

        # Begin Gradient Descent.
        time_start = time()
        theta, g_theta, F_theta = initial_theta, initial_g, inital_F
        vg = torch.zeros_like(theta).to(self.device)

        for i in range(iterations):

            sign_gradient, grad_queries = self.sign_grad_v1(x0, y0, theta, initial_F=F_theta, initial_g=g_theta)

            # Line search
            ls_count = 0
            min_theta = theta
            min_g = g_theta
            min_F = F_theta

            for _ in range(15):
                new_theta = theta - alpha * sign_gradient
                new_theta = torch.div(new_theta, torch.norm(new_theta))
                new_g, count = self.fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd = min_g)
                new_F = L1_norm(new_g*new_theta, self.device)
                ls_count += count
                alpha = alpha * 2 
                if (new_F  < min_F) and (L0_norm(new_g*new_theta, self.device) <= L0_norm(min_g*min_theta,self.device)):
                    min_theta = new_theta 
                    min_g = new_g
                    min_F = new_F
                else:
                    break     

            if min_F >= F_theta :
                for _ in range(15):
                    alpha = alpha * 0.25  
                    new_theta = theta - alpha * sign_gradient
                    new_theta = torch.div(new_theta, torch.norm(new_theta))
                    new_g, count = self.fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd = min_g) 
                    new_F = L1_norm(new_g*new_theta, self.device)
                    ls_count += count
                    if (new_F  < min_F) and (L0_norm(new_g*new_theta, self.device) <= L0_norm(min_g*min_theta,self.device)):
                        min_theta = new_theta 
                        min_g = new_g
                        min_F = new_F
                        break
            if alpha < 1e-4:  #1e-4
                alpha = 1.0
                print("Warning: not moving")
                beta = beta*0.1
                if (beta < 1e-5):  #1e-8
                    break
            
            theta, g_theta, F_theta = min_theta, min_g, min_F

            query_count += (grad_queries + ls_count)

            if query_count > query_limit:
               break
            
            if (i+1)% 5 == 0:
                print("Iteration %3d distortion %.4f  pertubation %.2f  num_queries %d" % (i+1, F_theta.item(), L0_norm(g_theta*theta,self.device).item(), query_count))

        G_final = new_graph(G0, torch.clamp(g_theta * theta, 0.0, 1.0))
        x_final = copy.deepcopy(x0).to(self.device)
        x_final.edge_index = from_networkx(G_final).to(self.device).edge_index.long()
        target = model.predict(x_final, self.device)
        time_end = time()
        print("Attack succeed: distortion %.4f perturbation %d queries %d \nTime: %.4f seconds" % (F_theta.item(), L0_norm(g_theta*theta,self.device).item(), query_count, time_end-time_start))
        return x_final, target, query_count, True, F_theta.item(), init

    def sign_grad_v1(self, x0, y0, theta, initial_F,initial_g, h=0.1):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        G = to_networkx(x0, to_undirected=True)
        x_new = copy.deepcopy(x0).to(self.device)
        Q = self.Q
        sign_grad = torch.zeros(theta.shape).to(self.device)
        queries = 0
        for iii in range(Q):
            u = torch.randn(*theta.shape).to(self.device)
            u = torch.triu(u, diagonal=1)
            u = torch.div(u, torch.norm(u))
            sign = 1
            new_theta = torch.clamp(theta + h*u, 0, 1)
            theta_norm = torch.norm(new_theta)
            new_theta = torch.div(new_theta, theta_norm)

            if self.effective:
                p0 = L1_norm(theta*initial_g, self.device)
                p0_new = L1_norm(new_theta*initial_g, self.device)
                if p0_new >= p0:
                    g_left = initial_g
                    g_right = initial_g
                    while L1_norm(new_theta*g_left,self.device) >= p0:
                        g_left = g_left*0.98
                else:
                    g_left = initial_g
                    g_right = initial_g
                    while L1_norm(new_theta*g_right,self.device) < p0:
                        g_right = g_right*1.02

                while g_right - g_left > 1e-4:
                    g_mid = (g_right + g_left) / 2
                    p1 = L1_norm(new_theta*g_mid, self.device)
                    if p1 < p0:
                        g_left = g_mid
                    else:
                        g_right = g_mid
                g_new = (g_right + g_left) / 2
                G_new = new_graph(G, torch.clamp(g_new*new_theta, 0.0, 1.0))
                x_new.edge_index = from_networkx(G_new).to(self.device).edge_index.long()
                if (self.model.predict(x_new, self.device) != y0):
                    sign = -1
                queries += 1
            else:
                new_g, count = self.fine_grained_binary_search_local(self.model, x0, y0, new_theta, initial_lbd = initial_g)
                new_F = L1_norm(new_theta*new_g, self.device)
                if new_F < initial_F:
                    sign = -1
                queries += count

            sign_grad += torch.sign(u)*sign
        sign_grad = sign_grad / Q
        return sign_grad, queries

   
    def fine_grained_binary_search_local(self, model, x0, y0, theta, initial_lbd, tol=1e-2):
        nquery = 0
        lbd = initial_lbd
        device = self.device
        G0 = to_networkx(x0, to_undirected=True)
        x_new = copy.deepcopy(x0).to(self.device)
        G1 = new_graph(G0, torch.clamp(lbd * theta, 0.0, 1.0))
        x_new.edge_index = from_networkx(G1).to(self.device).edge_index.long()
        if model.predict(x_new, device) == y0:
            lbd_lo = lbd
            lbd_hi = lbd*1.02  
            nquery += 1
            G2 = new_graph(G0, torch.clamp(lbd_hi * theta, 0.0, 1.0))
            x_new.edge_index = from_networkx(G2).to(self.device).edge_index.long()
            while model.predict(x_new, device) == y0:
                lbd_hi = lbd_hi*1.02
                nquery += 1
                G2 = new_graph(G0, torch.clamp(lbd_hi * theta, 0.0, 1.0))
                x_new.edge_index = from_networkx(G2).to(self.device).edge_index.long()
                if lbd_hi > 20:
                    return torch.FloatTensor([float('inf')]).to(self.device), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.98  #0.99
            nquery += 1
            G3 = new_graph(G0, torch.clamp(lbd_lo * theta, 0.0, 1.0))
            x_new.edge_index = from_networkx(G3).to(self.device).edge_index.long()
            while model.predict(x_new, device) != y0 :
                lbd_lo = lbd_lo*0.98
                G3 = new_graph(G0, torch.clamp(lbd_lo * theta, 0.0, 1.0))
                x_new.edge_index = from_networkx(G3).to(self.device).edge_index.long()
                nquery += 1

        while lbd_hi - lbd_lo > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            G_mid = new_graph(G0, torch.clamp(lbd_mid * theta, 0.0, 1.0))
            x_new.edge_index = from_networkx(G_mid).to(self.device).edge_index.long()
            if model.predict(x_new, device) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        g_theta = lbd_hi
        return g_theta, nquery


    def fine_grained_binary_search(self, model, x0, y0, theta, initial_lbd,current_best, index1=None, index2=None):
        #theta:  torch(N,N)
        #initial_lbd: torch([1])
        #current_best: torch([1])
        
        x_new = copy.deepcopy(x0).to(self.device)
        G0 = to_networkx(x0, to_undirected=True)
        nquery = 0
        
        if current_best < initial_lbd:
            G_new  = new_graph(G0, torch.clamp(current_best*theta,0.0,1.0), index1, index2)
            x_new.edge_index = from_networkx(G_new).to(self.device).edge_index.long()
            if model.predict(x_new, self.device) == y0:
                nquery += 1
                return torch.FloatTensor([float('inf')]).to(self.device), nquery
            lbd = current_best.to(self.device)
        else:
            lbd = initial_lbd.to(self.device)
        
        lbd_hi = lbd
        lbd_lo = torch.FloatTensor([0.0]).to(self.device)

        while lbd_hi-lbd_lo > 1e-2: 
            lbd_mid = (lbd_lo+lbd_hi)/2.0
            nquery += 1
            G_new = new_graph(G0, torch.clamp(lbd_mid * theta, 0.0, 1.0), index1, index2)
            x_new.edge_index = from_networkx(G_new).to(self.device).edge_index.long()
            if model.predict(x_new, self.device) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        g_theta = lbd_hi
        
        return g_theta, nquery


        
