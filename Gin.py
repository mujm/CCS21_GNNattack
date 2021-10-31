import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, GCNConv, TopKPooling, SAGPooling
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool as gap
from torch_geometric.nn.models import GraphUNet
from torch_geometric.data import Batch
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch.nn import Parameter

class MLP(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''
    
        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = torch.nn.Linear(input_dim, output_dim)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
        
            self.linears.append(torch.nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(torch.nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(torch.nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)

class GIN(torch.nn.Module):
    def __init__(self, num_layers, num_mlp_layers,input_dim, hidden_dim, output_dim, dropout):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.num_mlp_layers = num_mlp_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        
        self.liners_prediction = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.liners_prediction.append(torch.nn.Linear(input_dim, output_dim))
            else:
                self.liners_prediction.append(torch.nn.Linear(hidden_dim, output_dim))
        
        self.ginconv = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            self.ginconv.append(GINConv(self.mlps[layer], train_eps=True))
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
            
        hidden_rep = [x]

        for layer in range(self.num_layers - 1):
            x = self.ginconv[layer](x, edge_index)
            x = F.relu(self.batch_norms[layer](x))
            hidden_rep.append(x)
            
        score_over_layer = 0

        for layer, h in enumerate(hidden_rep):
            pooled_h = global_add_pool(h, batch)
            score_over_layer += F.dropout(self.liners_prediction[layer](pooled_h),self.dropout,training=self.training)
        return score_over_layer

    def predict(self, data, device):
        #this is the prediction for single graph
        self.eval() 
        graph = Batch.from_data_list([data]).to(device)
        output = self(graph)  #logits of graph: [[0.2,0.3,0.5]]
        pred = output.max(1, keepdim = True)[1] #final predicted label: [[1]]
        return pred[0][0]
        
    def predict_vector(self, data, device):
        self.eval()
        graph = Batch.from_data_list([data]).to(device)
        output = self(graph)
        vector = output[0]
        return torch.nn.functional.softmax(vector, dim=0)
        #return vector

class GCN(torch.nn.Module):
    def __init__(self, num_layers,input_dim, hidden_dim, output_dim, dropout):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
        self.gcns = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.liners_prediction = torch.nn.ModuleList()

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.gcns.append(GCNConv(input_dim, hidden_dim))
            else:
                self.gcns.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        
        for layer in range(self.num_layers):
            if layer == 0:
                self.liners_prediction.append(torch.nn.Linear(input_dim, output_dim))
            else:
                self.liners_prediction.append(torch.nn.Linear(hidden_dim, output_dim))
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        hidden_rep = [x]
        for layer in range(self.num_layers - 1):
            x = self.gcns[layer](x, edge_index)
            x = F.relu(self.batch_norms[layer](x))
            hidden_rep.append(x)
       

        score_over_layer = 0
        for layer, h in enumerate(hidden_rep):
            pooled_h = global_add_pool(h, batch)
            score_over_layer += F.dropout(self.liners_prediction[layer](pooled_h),self.dropout,training=self.training)
        return score_over_layer

    
    def predict(self, data, device):
        self.eval()
        graph = Batch.from_data_list([data]).to(device)
        output = self(graph)
        pred = output.max(1, keepdim = True)[1]
        return pred[0][0]

class SAGPool(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=GCNConv,non_linearity=torch.tanh):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.score_layer(x,edge_index).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm

class SAG(torch.nn.Module):
    def __init__(self,num_layers, input_dim, hidden_dim, output_dim, pooling_ratio, dropout):
        super(SAG, self).__init__()
        self.num_layers = num_layers
        self.num_features = input_dim
        self.nhid = hidden_dim
        self.num_classes = output_dim
        self.pooling_ratio = pooling_ratio
        self.dropout_ratio = dropout
        
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers-1):
            if layer == 0:
                self.convs.append(GCNConv(self.num_features, self.nhid))
            else:
                self.convs.append(GCNConv(self.nhid, self.nhid))
            self.batch_norms.append(torch.nn.BatchNorm1d(self.nhid))

        self.liners_prediction = torch.nn.ModuleList()
        
        for layer in range(num_layers):
            if layer == 0:
                self.liners_prediction.append(torch.nn.Linear(input_dim, output_dim))
            else:
                self.liners_prediction.append(torch.nn.Linear(hidden_dim, output_dim)) 

        self.sagpool = torch.nn.ModuleList()
        for layer in range(self.num_layers-1):
            self.sagpool.append(SAGPool(self.nhid, ratio=self.pooling_ratio))       

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        hidden_rep = [x]
        batch_rep = [batch]
        for layer in range(self.num_layers-1):
            x = self.convs[layer](x, edge_index)
            x, edge_index, _, batch, _ = self.sagpool[layer](x, edge_index, None, batch)
            x = F.relu(self.batch_norms[layer](x))
            hidden_rep.append(x)
            batch_rep.append(batch)
           
        score_over_layer = 0
        
        for layer, h in enumerate(hidden_rep):
            pooled_h = global_add_pool(h, batch_rep[layer])
            score_over_layer += F.dropout(self.liners_prediction[layer](pooled_h),self.dropout_ratio,training=self.training)
    
        return score_over_layer       
 
    def predict(self, data, device):
        #this is the prediction for single graph
        self.eval() 
        graph = Batch.from_data_list([data]).to(device)
        output = self(graph)  #logits of graph: [[0.2,0.3,0.5]]
        pred = output.max(1, keepdim = True)[1] #final predicted label: [[1]]
        return pred[0][0]


class GUNet(torch.nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim, pooling_ratio, deepth, dropout):
        super(GUNet, self).__init__()
        self.num_features = input_dim
        self.nhid = hidden_dim
        self.num_classes = output_dim
        self.pooling_ratio = pooling_ratio
        self.dropout_ratio = dropout
        self.deepth = deepth
        self.num_layers = deepth

        self.liners_prediction = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            if layer == 0:
                self.liners_prediction.append(torch.nn.Linear(input_dim, output_dim))
            else:
                self.liners_prediction.append(torch.nn.Linear(hidden_dim, output_dim)) 

        self.gunpool = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers-1):
            if layer ==0:
                self.gunpool.append(GraphUNet(self.num_features, 32, self.nhid, 2, self.pooling_ratio))      
            else:
                self.gunpool.append(GraphUNet(self.nhid, 32, self.nhid, 2, self.pooling_ratio))
            self.batch_norms.append(torch.nn.BatchNorm1d(self.nhid))

       # self.pool = GraphUNet(self.num_features,32, self.nhid, self.deepth,self.pooling_ratio)

        #self.lin1 = torch.nn.Linear(self.num_features, self.num_classes)
       # self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)
      #  self.lin3 = torch.nn.Linear(self.nhid//2, self. num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        hidden_rep = [x]

        for layer in range(self.num_layers-1):   
            x = self.gunpool[layer](x, edge_index)
            x = F.relu(self.batch_norms[layer](x))
            hidden_rep.append(x)
  
        score_over_layer = 0
        
        for layer, h in enumerate(hidden_rep):
            pooled_h = global_add_pool(h, batch)
            score_over_layer += F.dropout(self.liners_prediction[layer](pooled_h),self.dropout_ratio,training=self.training)
        
        return score_over_layer

    def predict(self, data, device):
        #this is the prediction for single graph
        self.eval() 
        graph = Batch.from_data_list([data]).to(device)
        output = self(graph)  #logits of graph: [[0.2,0.3,0.5]]
        pred = output.max(1, keepdim = True)[1] #final predicted label: [[1]]
        return pred[0][0]
#g.x: [num_node, num_node_features] 节点的特征矩阵X
#g.edge_index:[2, num_edges] torch.long
#g.edge_attr:[num_edges, num_edge_features] 边的特征矩阵
#g.y: node-level:[num_nodes, *], graph-level:[1,*] 标签
#g.pos: [num_nodes, num_dimensions]

#例如print(graph)
#Data(edge_attr=[38, 4], edge_index=[2, 38], x=[17, 7], y=[1])
#返回的都是相关参数的shape


#edge = graph.edge_index
#edge = edge.transpose(0, 1)
#edge = torch.cat((edge, torch.LongTensor([[3,16]]).to(device)), dim=0)
#edge = torch.cat((edge, torch.LongTensor([[16,3]]).to(device)), dim=0)
#graph.edge_index = edge.transpose(0, 1)
