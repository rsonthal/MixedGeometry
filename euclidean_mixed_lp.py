import networkx as nx
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# torch geometric
try:
    import torch_geometric
except ModuleNotFoundError:
    # Installing torch geometric packages with specific CUDA+PyTorch version.
    # See https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html for details
    TORCH = '2.0.0'
    CUDA = 'cu' + torch.version.cuda.replace('.','')
    !pip install torch-scatter     -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
    !pip install torch-sparse      -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
    !pip install torch-cluster     -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
    !pip install torch-spline-conv -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
    !pip install torch-geometric
    import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data

import pandas as pd
import json
from math import ceil
from tqdm import tqdm
import os
from csv import writer

import torch
import random
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T
import torch_geometric
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import train_test_split_edges
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from torch_geometric.transforms import RandomLinkSplit

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

#hyperbolic version linear layer
class LorentzLinear(nn.Module):
    """
        Perform the Lorentz linear transformation.

        args:
            in_features, out_features, bias: Same as nn.Linear
            dropout: Dropout rate in lorentz linear
            manifold: THe manifold that the linear layer operated in.
            nonlin: Non-linear function before the linear operation.
            merge: If set to True, it means that the input has the shape of [..., head_num, head_dim], and the output will has the shape of [..., head_num * head_dim]. The heads are merged.
            head_num: If `merge` is set to True, then head_num specifies the number of heads in input, otherwise it means that the output should be split into `head_num` heads, i.e., [..., head_num, head_dim]. If set to 0, then it is a normal lorentz linear layer.
    """
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 dropout=0.0,
                 nonlin=None):
        super().__init__()
        self.nonlin = nonlin
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = nn.Linear(
            self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * 2.3)

    def forward(self, x, bias=None):
        if self.nonlin is not None:
            x = self.nonlin(x)

        x = self.weight(self.dropout(x))

        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        if bias is not None:
            x = x + bias
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1
        scale = (time * time - 1.0) / \
            (x_narrow * x_narrow).sum(dim=-1, keepdim=True)
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        return x

    def reset_parameters(self):
        stdv = 0.02
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        step = self.in_features // self.in_features
        if self.bias:
            nn.init.uniform_(self.weight.bias, -stdv, stdv)

class GATLayer(nn.Module):

    def __init__(self, c_in, c_out, node_feats, edge_index, num_heads=1, concat_heads=True, alpha=0.2):
        """
        Inputs:
            c_in - Dimensionality of input features
            c_out - Dimensionality of output features
            num_heads - Number of heads, i.e. attention mechanisms to apply in parallel. The
                        output features are equally split up over the heads if concat_heads=True.
            concat_heads - If True, the output of the different heads is concatenated instead of averaged.
            alpha - Negative slope of the LeakyReLU activation.
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            c_out = c_out // num_heads

        # Sub-modules and parameters needed in the layer
        self.projection = nn.Linear(c_in, c_out * num_heads)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * c_out)) # One per head
        self.leakyrelu = nn.LeakyReLU(alpha)

        # Initialization from the original implementation
        nn.init.xavier_uniform_(self.projection.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        adj_matrix = to_dense_adj(edge_index)
        self.adj = adj_matrix
        self.adj = self.adj.to(node_feats.device)

    def forward(self, node_feats, print_attn_probs=False):
        """
        Inputs:
            node_feats - Input features of the node. Shape: [batch_size, c_in]
            adj_matrix - Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
            print_attn_probs - If True, the attention weights are printed during the forward pass (for debugging purposes)
        """
        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)

        adj_matrix = self.adj
        # Apply linear layer and sort nodes by head
        node_feats = self.projection(node_feats)
        node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)

        # We need to calculate the attention logits for every edge in the adjacency matrix
        # Doing this on all possible combinations of nodes is very expensive
        # => Create a tensor of [W*h_i||W*h_j] with i and j being the indices of all edges
        edges = adj_matrix.nonzero(as_tuple=False) # Returns indices where the adjacency matrix is not 0 => edges
        node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)
        edge_indices_row = edges[:,0] * num_nodes + edges[:,1]
        edge_indices_col = edges[:,0] * num_nodes + edges[:,2]
        a_input = torch.cat([
            torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0),
            torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0)
        ], dim=-1) # Index select returns a tensor with node_feats_flat being indexed at the desired positions along dim=0

        # Calculate attention MLP output (independent for each head)
        attn_logits = torch.einsum('bhc,hc->bh', a_input, self.a)
        attn_logits = self.leakyrelu(attn_logits)

        # Map list of attention values back into a matrix
        attn_matrix = attn_logits.new_zeros(adj_matrix.shape+(self.num_heads,)).fill_(-9e15)
        attn_matrix[adj_matrix[...,None].repeat(1,1,1,self.num_heads) == 1] = attn_logits.reshape(-1)

        # Weighted average of attention
        attn_probs = F.softmax(attn_matrix, dim=2)
        if print_attn_probs:
            print("Attention probs\n", attn_probs.permute(0, 3, 1, 2))
        node_feats = torch.einsum('bijh,bjhc->bihc', attn_probs, node_feats)

        # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)

        return node_feats


class HGATLayer(nn.Module):

  def __init__(self, c_in, c_out, node_feats, edge_index, num_heads=1, concat_heads=True, alpha=0.2):
      """
      Inputs:
          c_in - Dimensionality of input features
          c_out - Dimensionality of output features
          num_heads - Number of heads, i.e. attention mechanisms to apply in parallel. The
                      output features are equally split up over the heads if concat_heads=True.
          concat_heads - If True, the output of the different heads is concatenated instead of averaged.
          alpha - Negative slope of the LeakyReLU activation.
      """
      super().__init__()
      self.num_heads = num_heads
      self.concat_heads = concat_heads
      if self.concat_heads:
          assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
          c_out = c_out // num_heads

      # Sub-modules and parameters needed in the layer
      self.projection = LorentzLinear(c_in, c_out * num_heads)
      self.a = nn.Parameter(torch.Tensor(num_heads, 2 * c_out)) # One per head
      self.leakyrelu = nn.LeakyReLU(alpha)

      # Initialization from the original implementation
      nn.init.xavier_uniform_(self.a.data, gain=1.414)

      adj_matrix = to_dense_adj(edge_index)
      self.adj = adj_matrix
      self.adj = self.adj.to(node_feats.device)

  def forward(self, node_feats, print_attn_probs=False):
      """
      Inputs:
          node_feats - Input features of the node. Shape: [batch_size, c_in]
          adj_matrix - Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
          print_attn_probs - If True, the attention weights are printed during the forward pass (for debugging purposes)
      """
      batch_size, num_nodes = node_feats.size(0), node_feats.size(1)

      adj_matrix = self.adj
      # Apply linear layer and sort nodes by head
      node_feats = self.projection(node_feats)
      node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)

      # We need to calculate the attention logits for every edge in the adjacency matrix
      # Doing this on all possible combinations of nodes is very expensive
      # => Create a tensor of [W*h_i||W*h_j] with i and j being the indices of all edges
      edges = adj_matrix.nonzero(as_tuple=False) # Returns indices where the adjacency matrix is not 0 => edges
      node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)
      edge_indices_row = edges[:,0] * num_nodes + edges[:,1]
      edge_indices_col = edges[:,0] * num_nodes + edges[:,2]
      a_input = torch.cat([
          torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0),
          torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0)
      ], dim=-1) # Index select returns a tensor with node_feats_flat being indexed at the desired positions along dim=0

      # Calculate attention MLP output (independent for each head)
      attn_logits = torch.einsum('bhc,hc->bh', a_input, self.a)
      attn_logits = self.leakyrelu(attn_logits)

      # Map list of attention values back into a matrix
      attn_matrix = attn_logits.new_zeros(adj_matrix.shape+(self.num_heads,)).fill_(-9e15)
      attn_matrix[adj_matrix[...,None].repeat(1,1,1,self.num_heads) == 1] = attn_logits.reshape(-1)

      # Weighted average of attention
      attn_probs = F.softmax(attn_matrix, dim=2)
      if print_attn_probs:
          print("Attention probs\n", attn_probs.permute(0, 3, 1, 2))
      node_feats = torch.einsum('bijh,bjhc->bihc', attn_probs, node_feats)

      # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
      if self.concat_heads:
          node_feats = node_feats.reshape(batch_size, num_nodes, -1)
      else:
          node_feats = node_feats.mean(dim=2)

      return node_feats

class GCNLayer(nn.Module):

    def __init__(self, c_in, c_out, node_feats, edge_index):
        super().__init__()
        self.w = nn.Linear(c_in, c_out)
        edge_index = torch.cat((edge_index, edge_index[[1,0]]), dim=1)
        adj_matrix = to_dense_adj(edge_index)[0]
        self.adj = adj_matrix
        self.adj = self.adj.to(node_feats.device)

    def forward(self, node_feats):
        """
        Inputs:
            node_feats - Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                         Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        node_feats = node_feats[0]
        adj_matrix = self.adj
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(dim=-1)
        node_feats = self.w(node_feats)
        mask = (num_neighbours > 0)
        adj_matrix = adj_matrix[mask,:]
        adj_matrix = adj_matrix[:,mask]
        node_feats[mask,:] = adj_matrix.mm(node_feats[mask,:])
        node_feats[mask,:] =  node_feats[mask,:] / num_neighbours[mask].view(-1,1)
        node_feats = torch.unsqueeze(node_feats, dim=0)
        return node_feats


class HGCNLayer(nn.Module):

    def __init__(self, c_in, c_out, node_feats, edge_index):
        super().__init__()
        # self.w = nn.Linear(c_in, c_out)
        self.w = LorentzLinear(c_in, c_out)
        # make graph undirected
        edge_index = torch.cat((edge_index, edge_index[[1,0]]), dim=1)
        adj_matrix = to_dense_adj(edge_index)[0]
        self.adj = adj_matrix
        self.adj = self.adj.to(node_feats.device)

    def forward(self, node_feats):
        """
        Inputs:
            node_feats - Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                         Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        node_feats = node_feats[0]
        adj_matrix = self.adj
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(dim=-1)
        node_feats = self.w(node_feats)
        mask = (num_neighbours > 0)
        adj_matrix = adj_matrix[mask,:]
        adj_matrix = adj_matrix[:,mask]
        node_feats[mask,:] = adj_matrix.mm(node_feats[mask,:])
        node_feats[mask,:] =  node_feats[mask,:] / num_neighbours[mask].view(-1,1)
        node_feats = torch.unsqueeze(node_feats, dim=0)
        return node_feats

class GraphConvLayer(nn.Module):

    def __init__(self, c_in, c_out, node_feats, edge_index):
        super().__init__()
        self.w1 = nn.Linear(c_in, c_out)
        self.w2 = nn.Linear(c_in, c_out, bias=False)
        adj_matrix = to_dense_adj(edge_index)[0]
        self.adj = adj_matrix
        self.adj = self.adj.to(node_feats.device)

    def forward(self, node_feats):
        """
        Inputs:
            node_feats - Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                         Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
                         Shape: [batch_size, num_nodes, num_nodes]
        """

        adj_matrix = self.adj
        node_feats1 = self.w1(node_feats)
        node_feats2 = torch.matmul(adj_matrix, node_feats)
        node_feats2 = self.w2(node_feats2)
        return node_feats1 + node_feats2

class HGraphConvLayer(nn.Module):
    def __init__(self, c_in, c_out, node_feats, edge_index):
        super().__init__()
        self.w1 = LorentzLinear(c_in, c_out)
        self.w2 = LorentzLinear(c_in, c_out)
        adj_matrix = to_dense_adj(edge_index)[0]

        self.adj = adj_matrix
        self.adj = self.adj.to(node_feats.device)



    def forward(self, node_feats):
        """
        Inputs:
            node_feats - Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                         Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
                         Shape: [batch_size, num_nodes, num_nodes]
        """

        adj_matrix = self.adj
        node_feats1 = self.w1(node_feats)
        node_feats2 = torch.matmul(adj_matrix, node_feats)
        node_feats2 = self.w2(node_feats2)
        return node_feats1 + node_feats2

class HResGatedGraphConvLayer(nn.Module):

    def __init__(self, c_in, c_out, node_feats, edge_index):
        super().__init__()
        self.w1 = LorentzLinear(c_in, c_out)
        self.w2 = LorentzLinear(c_in, c_out)
        self.w3 = LorentzLinear(c_in, c_out)
        self.w4 = LorentzLinear(c_in, c_out)
        self.sigmoid = nn.Sigmoid()
        adj_matrix = to_dense_adj(edge_index)[0]
        self.adj = adj_matrix
        self.adj = self.adj.to(node_feats.device)

    def forward(self, node_feats):
        """
        Inputs:
            node_feats - Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                         Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
                         Shape: [batch_size, num_nodes, num_nodes]
        """

        adj_matrix = self.adj
        node_feats_i = torch.matmul(adj_matrix, node_feats)
        node_feats3 = self.w3(node_feats_i)
        node_feats4 = self.w4(node_feats)
        eta = self.sigmoid(node_feats3 + node_feats4)
        node_feats2 = self.w2(node_feats)
        node_feats2 = torch.mul(eta, node_feats2)
        node_feats1 = self.w1(node_feats_i)
        node_feats1 = node_feats1 + node_feats2
        return node_feats1


class ResGatedGraphConvLayer(nn.Module):

    def __init__(self, c_in, c_out, node_feats, edge_index):
        super().__init__()
        self.w1 = nn.Linear(c_in, c_out)
        self.w2 = nn.Linear(c_in, c_out)
        self.w3 = nn.Linear(c_in, c_out)
        self.w4 = nn.Linear(c_in, c_out)
        self.sigmoid = nn.Sigmoid()
        adj_matrix = to_dense_adj(edge_index)[0]
        self.adj = adj_matrix
        self.adj = self.adj.to(node_feats.device)

    def forward(self, node_feats):
        """
        Inputs:
            node_feats - Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                         Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
                         Shape: [batch_size, num_nodes, num_nodes]
        """

        adj_matrix = self.adj
        node_feats_i = torch.matmul(adj_matrix, node_feats)
        node_feats3 = self.w3(node_feats_i)
        node_feats4 = self.w4(node_feats)
        eta = self.sigmoid(node_feats3 + node_feats4)
        node_feats2 = self.w2(node_feats)
        node_feats2 = torch.mul(eta, node_feats2)
        node_feats1 = self.w1(node_feats_i)
        node_feats1 = node_feats1 + node_feats2
        return node_feats1

class Net(torch.nn.Module):
    def __init__(self, layer, d, data, num_layers):
        super(Net, self).__init__()
        if layer[0] == "H":
            self.input_net = LorentzLinear(data['train'].num_features, d, dropout = 0.0)
            # self.output_net = LorentzLinear(d, d, dropout = 0.0)
        else:
            self.input_net = nn.Linear(data['train'].num_features, d)
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            if layer == "GCN":
                self.layers.append(GCNLayer(d, d, data['train'].x, data['train'].edge_label_index))
            elif layer == "GC":
                self.layers.append(GraphConvLayer(d, d, data['train'].x, data['train'].edge_label_index))
            elif layer == "RGGC":
                self.layers.append(ResGatedGraphConvLayer(d, d, data['train'].x, data['train'].edge_label_index))
            elif layer == "GAT":
                self.layers.append(GATLayer(d, d, data['train'].x, data['train'].edge_label_index))
            elif layer == "HGCN":
                self.layers.append(HGCNLayer(d, d, data['train'].x, data['train'].edge_label_index))
            elif layer == "HGC":
                self.layers.append(HGraphConvLayer(d, d, data['train'].x, data['train'].edge_label_index))
            elif layer == "HRGGC":
                self.layers.append(HResGatedGraphConvLayer(d, d, data['train'].x, data['train'].edge_label_index))
            elif layer == "HGAT":
                self.layers.append(HGATLayer(d, d, data['train'].x, data['train'].edge_label_index))
        self.data = data['train']
        self.num_layers = num_layers
        self.layer = layer

    def forward(self, pos_edge_index, neg_edge_index):
        x = self.data.x
        x = self.input_net(x.unsqueeze(0))
        for layer in self.layers:
            x = F.relu(layer(x))
        x = x.squeeze(0)
        pos_neg_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1) # concatenate pos and neg edges
        if self.layer[0] == "H":
            logits = (x[pos_neg_edge_index[0],1:] * x[pos_neg_edge_index[1],1:]).sum(dim=-1) #- (x[pos_neg_edge_index[0],0] * x[pos_neg_edge_index[1],0]) # dot product
            logits = logits
        else:
            logits = (x[pos_neg_edge_index[0]] * x[pos_neg_edge_index[1]]).sum(dim=-1)
        return logits

def initialize_empty_dict(dict_model):
  d ={}
  d[dict_model] = {}
  for layers in num_layers:
    d[dict_model][layers]={}
    for dim in dims:
      d[dict_model][layers][dim] = {}
      for rate in learning_rates:
        d[dict_model][layers][dim][rate] ={}
        for trial in range(1,trials+1):
          d[dict_model][layers][dim][rate][trial] ={}
          for e in range(1,epochs+1):
            d[dict_model][layers][dim][rate][trial][e] ={}
            for acc in ['auc_train', 'auc_val', 'auc_test']:
              d[dict_model][layers][dim][rate][trial][e][acc] = {}
              d[dict_model][layers][dim][rate][trial][e][acc] = 0

  return d

def export_dict(data_name,model_name, d):
  file_name = f'models_report/{data_name}_{model_name}_report-new-version.json'
  with open(file_name, "w") as write_file:
    json.dump(d, write_file, indent = "")

def train(model, data, optimizer):
  with torch.autograd.set_detect_anomaly(True):
    model.train()
    neg_edge_index = negative_sampling(
        edge_index=data['train'].edge_label_index, #positive edges
        num_nodes=data['train'].num_nodes, # number of nodes
        num_neg_samples=data['train'].edge_label_index.size(1)) # number of neg_sample equal to number of pos_edges
    optimizer.zero_grad()
    link_logits = model(data['train'].edge_label_index, neg_edge_index)
    link_labels = torch.zeros(data['train'].edge_label_index.size(1)*2, dtype=torch.float, device=device)
    link_labels[:data['train'].edge_label_index.size(1)] = 1.
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def test(model, data):
    model.eval()
    perfs = {}
    for prefix in ["train", "val", "test"]:
        data_split = data[prefix]
        if prefix == "train":
            pos_edge_index = data_split.edge_label_index
            neg_edge_index = negative_sampling(
                edge_index=data_split.edge_label_index, #positive edges
                num_nodes=data_split.num_nodes, # number of nodes
                num_neg_samples=data_split.edge_label_index.size(1)) # number of neg_sample equal to number of pos_edges
            link_labels = torch.zeros(pos_edge_index.size(1)*2, dtype=torch.float, device=device)
            link_labels[:pos_edge_index.size(1)] = 1.
        else:
            pos_edge_index, neg_edge_index = data_split.edge_label_index.chunk(2, dim=1)
            link_labels = data_split.edge_label

        link_logits = model(pos_edge_index, neg_edge_index)
        link_probs = link_logits.sigmoid() # apply sigmoid

        perfs[f'{prefix}_auc'] = roc_auc_score(link_labels.cpu(), link_probs.cpu())

    return perfs

def split_edges(data):
    # train/test split
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    transform = RandomLinkSplit(num_val=0.05, num_test=0.1, add_negative_train_samples=False)
    train_data, val_data, test_data = transform(data)
    data = {'train':train_data, 'val':val_data, 'test':test_data}
    return data

# citation
data_cora = torch.load('data_cora')
data_citeseer = torch.load('data_citeseer')
data_pubmed = torch.load('data_pubmed')
# # airports
data_usa = torch.load('data_usa')
data_brazil = torch.load('data_brazil')
data_europe = torch.load('data_europe')
# # WebKB
data_cornell = torch.load('data_cornell')
data_texas = torch.load('data_texas')
data_wisconsin = torch.load('data_wisconsin')

trials = 10
epochs = 200
model_names = ["GCN", "GC", "GAT", "HGCN", "HGC", "HGAT"]
datasets = {"Cora": data_cora, "CiteSeer": data_citeseer, \
            "AirportsUSA": data_usa, "AirportsBrazil": data_brazil,"AirportsEurope": data_europe,\
            "WebKB-Cornell": data_cornell, "WebKB-Wisconsin": data_wisconsin, "WebKB-Texas": data_texas}

num_layers = [2, 3, 4]
dims = [32, 64, 128, 256]
learning_rates = [2e-4, 1e-3, 5e-3]

def train_model(dic, model_name, data, data_name, num_layer, d, lr, config):
  filename = "dataset=" + data_name + "_layers=" + str(num_layer) + \
              "_dims=" + str(d) + "_lr=" + str(lr) + ".pt"
  model_weights = []
  aucs = []
  for trial in range(1, trials+1):
      print("training", model_name + "/models/" + filename + " trial " + str(trial) + ", config " + str(config))
      model = Net(model_name, d, data, num_layer).to(device)
      optimizer = getattr(torch.optim, "Adam")(model.parameters(), lr=lr)
      best_val_auc = 0
      best_model = None
      best_results = None
      best_epoch = None
      for epoch in tqdm(range(1, epochs+1)):
        loss = train(model, data, optimizer)
        results = test(model, data)
        results['loss'] = loss.item()
        dic[model_name][num_layer][d][lr][trial][epoch]['auc_train'] = results['train_auc']
        dic[model_name][num_layer][d][lr][trial][epoch]['auc_val'] = results['val_auc']
        dic[model_name][num_layer][d][lr][trial][epoch]['auc_test'] = results['test_auc']

        # save best model
        if results['val_auc'] > best_val_auc:
          best_val_auc = results['val_auc']
          best_model = model
          best_results = results
          best_epoch = epoch
      model_weights.append(best_model.state_dict())
      aucs.append(tuple(best_results))
  # save 10 trials in one file
  torch.save(model_weights, model_name + "/models/" + filename)
  torch.save(aucs, model_name + "/results/" + filename)

for model_name in model_names:
  for data_name, data in datasets.items():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for key, val in data.items():
        val = val.to(device)
    # print(data)
    # data.num_classes = int(data.y.max()+1)
    dic = initialize_empty_dict(model_name)
    # path_model = path + model_name + "/"
    if os.path.exists('models_report/'+data_name+'_'+model_name+'_report-new-version.json'):
        print("Finished training "+model_name+" on "+data_name)
        continue
    config = 1
    for num_layer in num_layers:
      if data_name == "PubMed" and num_layer > 2:
        print("==========")
        print("Skipping PubMed on >2 layer " + model_name + " for now...")
        print("==========")
        continue
      for d in dims:
        for lr in learning_rates:
          train_model(dic, model_name, data, data_name, num_layer, d, lr, config)
          config += 1
    export_dict(data_name,model_name,dic)
