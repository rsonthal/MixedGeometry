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
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T
import torch_geometric

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
            self.input_net = LorentzLinear(data.num_features, d, dropout = 0.0)
        else:
            self.input_net = nn.Linear(data.num_features, d)
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            if layer == "GCN":
                self.layers.append(GCNLayer(d, d, data.x, data.edge_index))
            elif layer == "GC":
                self.layers.append(GraphConvLayer(d, d, data.x, data.edge_index))
            elif layer == "RGGC":
                self.layers.append(ResGatedGraphConvLayer(d, d, data.x, data.edge_index))
            elif layer == "GAT":
                self.layers.append(GATLayer(d, d, data.x, data.edge_index))
            elif layer == "HGCN":
                self.layers.append(HGCNLayer(d, d, data.x, data.edge_index))
            elif layer == "HGC":
                self.layers.append(HGraphConvLayer(d, d, data.x, data.edge_index))
            elif layer == "HRGGC":
                self.layers.append(HResGatedGraphConvLayer(d, d, data.x, data.edge_index))
            elif layer == "HGAT":
                self.layers.append(HGATLayer(d, d, data.x, data.edge_index))
        self.output_net = nn.Linear(d, int(data.num_classes))
        self.data = data
        self.num_layers = num_layers

    def forward(self):
        x, edge_index = self.data.x, self.data.edge_index
        x = self.input_net(x.unsqueeze(0))
        for layer in self.layers:
            x = F.relu(layer(x))
        x = x.squeeze(0)
        x = self.output_net(x)
        return F.log_softmax(x, dim=1)

def initialize_empty_dict():
  d ={}
  for model in model_names:
    d[model] = {}
    for layers in num_layers:
      d[model][layers]={}
      for dim in dims:
        d[model][layers][dim] = {}
        for rate in learning_rates:
          d[model][layers][dim][rate] ={}
          for trial in range(1,trials+1):
            d[model][layers][dim][rate][trial] ={}
            for e in range(1,epochs+1):
              d[model][layers][dim][rate][trial][e] ={}
              for acc in ['acc_train', 'acc_val', 'acc_test']:
                d[model][layers][dim][rate][trial][e][acc] = {}
                d[model][layers][dim][rate][trial][e][acc] = 0
  return d

def export_dict(data_name,model_name, d):
  file_name = f'models_report_2/{data_name}_{model_name}_report.json'
  with open(file_name, "w") as write_file:
    json.dump(d, write_file, indent = "")

def train(model, data, optimizer):
  with torch.autograd.set_detect_anomaly(True):
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model()[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

@torch.no_grad()
def test(model, data):
  model.eval()
  logits = model()
  mask_train = data['train_mask']
  pred_train = logits[mask_train].max(1)[1]
  acc_train = pred_train.eq(data.y[mask_train]).sum().item() / mask_train.sum().item()

  mask_val = data['val_mask']
  pred_val = logits[mask_val].max(1)[1]
  acc_val = pred_val.eq(data.y[mask_val]).sum().item() / mask_val.sum().item()

  mask_test = data['test_mask']
  pred_test = logits[mask_test].max(1)[1]
  acc_test = pred_test.eq(data.y[mask_test]).sum().item() / mask_test.sum().item()
  return acc_train, acc_val, acc_test

def create_masks(data):
  total = data.y.size(dim=0)
  train_size = ceil(total * .16)
  val_size = ceil(total * .14)
  test_size = total - train_size - val_size
  data['train_mask'] = torch.zeros_like(data.y)
  data['train_mask'][:train_size] = torch.ones(train_size)
  data['train_mask'] = data['train_mask'].type(torch.BoolTensor)
  data['val_mask'] = torch.zeros_like(data.y)
  data['val_mask'][train_size:(train_size+val_size)] = torch.ones(val_size)
  data['val_mask'] = data['val_mask'].type(torch.BoolTensor)
  data['test_mask'] = torch.zeros_like(data.y)
  data['test_mask'][(train_size+val_size):] = torch.ones(test_size)
  data['test_mask'] = data['test_mask'].type(torch.BoolTensor)
  data.num_classes = int(data.y.max()+1)

europe_index = np.array([299,  65, 246, 172,   3, 163, 347, 135, 147, 116, 318, 323, 160,
       209, 189,  88, 279,  40, 385, 127,  89,  59, 239, 320, 101, 391,
       196, 199, 373, 254, 112,  26, 321, 309, 136,  20, 227, 219, 224,
       336, 368, 298, 118, 343, 256,  54,  47, 387, 324, 208,  70, 183,
       128, 226, 398, 355,  34, 281,  43,  63, 229,  93,  23, 255, 222,
        27, 346, 105, 206, 359,  85,  28,  15, 149, 362, 364,  66, 273,
       153, 284, 204, 213, 134, 150, 316,  52,  60, 151, 168, 214, 173,
       365, 220, 397, 317, 325, 187, 231, 123, 269, 125, 390, 225,  81,
       280,  68, 392, 212, 286, 170, 285,  62, 109, 263, 122, 251, 345,
       319, 303, 148, 312, 238, 244, 126, 250, 201, 282, 193, 270, 330,
       175, 367,  41, 165,   9, 278, 129,  92, 245, 366, 258, 124, 395,
       156,  39, 302, 252, 296,  94,  48,  96, 339,   0, 333,  16, 356,
       188,  83, 253, 232,  30, 342, 247, 191, 203,  51,   6, 314, 133,
       138,  45, 274, 360, 237, 332, 143, 257,  22, 337, 115, 381,   7,
        61, 164, 389,  44, 315,  55, 375, 304, 113,  33, 380,  46,  69,
       266,  97, 334,  84,  57,  12, 207, 240, 396, 192, 155,  98,  76,
       107, 108, 119, 262, 197, 200, 357, 394,  75, 377, 361, 327,  10,
        24, 142, 161,  78,   1, 353, 243,  71,  17, 233,  99, 144, 313,
       393,  58, 205,  77,  90, 152,  73, 221, 341, 111, 210, 202,   4,
         2,  80, 372, 288, 369,  91, 379, 184, 384, 383, 216, 198, 131,
       358, 194, 120, 349, 328, 275, 264,  11, 267, 300,  36, 331, 363,
       110,  49, 169, 181, 311, 145, 167, 146,  21, 177, 241, 291, 166,
       190, 293, 371, 159, 276, 211, 230,  95, 158, 215, 297, 272, 106,
        14,  13, 310, 374, 308, 289, 295, 179, 121, 162, 292, 260, 294,
       307, 306, 351, 322, 248, 234, 261, 249, 157, 388, 185,  35, 305,
       338,  67, 141, 277,  32, 287,  72, 259, 344, 104,  87, 378, 326,
       217, 301, 140, 376,  64,  82, 370, 235,  25, 386, 268, 242,  18,
       223, 265, 335,   8, 130,  53, 139,  29,  50, 171, 178, 329, 340,
       103, 174,   5, 350,  74, 218,  37,  42,  56, 176, 382, 117, 182,
       352, 228, 290, 186, 195, 236,  19, 283, 180, 271, 154, 137, 354,
        38, 100, 348,  79, 114,  86, 102,  31, 132])

def create_masks_airports_europe(data):
  total = data.y.size(dim=0)
  train_size = ceil(total * .16)
  val_size = ceil(total * .14)
  test_size = total - train_size - val_size

  data['train_mask'] = torch.zeros_like(data.y)
  for a in europe_index[:train_size]:
    data['train_mask'][a] = torch.ones(1)
  data['train_mask'] = data['train_mask'].type(torch.BoolTensor)

  data['val_mask'] = torch.zeros_like(data.y)
  for b in europe_index[train_size:(train_size+val_size)]:
    data['val_mask'][b] = torch.ones(1)
  data['val_mask'] = data['val_mask'].type(torch.BoolTensor)

  data['test_mask'] = torch.zeros_like(data.y)
  for c in europe_index[(train_size+val_size):]:
    data['test_mask'][c] = torch.ones(1)
  data['test_mask'] = data['test_mask'].type(torch.BoolTensor)

data_cora = torch_geometric.datasets.Planetoid(root="./", name="Cora")[0]
create_masks(data_cora)
data_citeseer = torch_geometric.datasets.Planetoid(root="./", name="CiteSeer")[0]
create_masks(data_citeseer)
data_pubmed = torch_geometric.datasets.Planetoid(root="./", name="PubMed")[0]
create_masks(data_pubmed)
# airports
data_airports_usa = torch_geometric.datasets.Airports(root="./", name="USA")[0]
create_masks(data_airports_usa)
data_airports_brazil = torch_geometric.datasets.Airports(root="./", name="Brazil")[0]
create_masks(data_airports_brazil)
data_airports_europe = torch_geometric.datasets.Airports(root="./", name="Europe")[0]
create_masks_airports_europe(data_airports_europe)
#WebKB
data_cornell = torch_geometric.datasets.WebKB(root="./", name="Cornell")[0]
create_masks(data_cornell)
data_texas = torch_geometric.datasets.WebKB(root="./", name="Texas")[0]
create_masks(data_texas)
data_wisconsin = torch_geometric.datasets.WebKB(root="./", name="Wisconsin")[0]
create_masks(data_wisconsin)

trials = 10
epochs = 100
model_names = ["GCN", "GC", "GAT", "HGCN", "HGC", "HGAT"]
# model_names = ["GCN", "HGCN"]
datasets = {"Cora": data_cora, "CiteSeer": data_citeseer, "PubMed": data_pubmed, \
            "AirportsUSA": data_airports_usa, "AirportsBrazil": data_airports_brazil,"AirportsEurope": data_airports_europe, \
            "WebKB-Corenell": data_cornell, "WebKB-Texas": data_texas, "WebKB-Wisconsin": data_wisconsin}
num_layers = [2, 3, 4]
dims = [32, 64, 128, 256]
learning_rates = [2e-4, 1e-3, 5e-3]

def train_model(dic, model_name, data, data_name, num_layer, d, lr, config):
  filename = "dataset=" + data_name + "_layers=" + str(num_layer) + \
              "_dims=" + str(d) + "_lr=" + str(lr) + "_set2" + ".pt"
  model_weights = []
  accs = []
  for trial in range(1, trials+1):
      print("training", model_name + "/models/" + filename + " trial " + str(trial) + ", config " + str(config))
      model = Net(model_name, d, data, num_layer).to(device)
      optimizer = getattr(torch.optim, "Adam")(model.parameters(), lr=lr)
      best_val_acc = 0
      best_model = None
      best_results = None
      best_epoch = None
      for epoch in tqdm(range(1, epochs+1)):
        train(model, data, optimizer)
        results = test(model, data)
        dic[model_name][num_layer][d][lr][trial][epoch]['acc_train'] = results[0]
        dic[model_name][num_layer][d][lr][trial][epoch]['acc_val'] = results[1]
        dic[model_name][num_layer][d][lr][trial][epoch]['acc_test'] = results[2]

        # save best model
        if results[1] > best_val_acc:
          best_val_acc = results[1]
          best_model = model
          best_results = results
          best_epoch = epoch
        if epoch % 10 == 0:
          print(results)
      model_weights.append(best_model.state_dict())
      accs.append(tuple(best_results))
      print("Best val accuracy =", str(best_val_acc), ", at epoch", str(best_epoch))
  # save 10 trials in one file
  # torch.save(model_weights, model_name + "/models/" + filename)
  # torch.save(accs, model_name + "/results/" + filename)

for data_name, data in datasets.items():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  data = data.to(device)
  data.num_classes = int(data.y.max()+1)
  dic = initialize_empty_dict()
  for model_name in model_names:
    config = 1
    for num_layer in num_layers:
      if data_name == "PubMed" and (model_name == "GAT" or model_name == "HGAT") and num_layer > 2:
        print("==========")
        print("Skipping PubMed on >2 layer " + model_name + " for now...")
        print("==========")
        continue
      for d in dims:
        for lr in learning_rates:
          train_model(dic, model_name, data, data_name, num_layer, d, lr, config)
          config += 1
    # export_dict(data_name,model_name,dic)
