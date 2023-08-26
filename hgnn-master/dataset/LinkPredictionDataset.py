#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import save_npz, load_npz
from scipy.sparse.linalg import eigsh
import sys
from torch.utils.data import Dataset, DataLoader
from utils import *
import torch
import torch_geometric
from torch_geometric.utils import to_dense_adj
from torch.nn.functional import one_hot
from math import ceil
# from torch.nn.utils.rnn import pad_sequence

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1)).astype(float)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

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

def create_masks_europe(data, europe_index):
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

class LinkPredictionDataset(Dataset):
    """
    Extend the Dataset class for graph datasets
    """
    def __init__(self, args, logger):
        self.args = args
        self.load_data(self.args.dataset_str)

    def load_airport_data(self, dataset_str):
        
        data = torch_geometric.datasets.Airports(root="../data/", name=dataset_str)[0]
        # shuffle airport Europe
        if dataset_str == 'Europe':
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
            create_masks_europe(data, europe_index)
        else:
            create_masks(data)

        adj = sp.csr_matrix(to_dense_adj(data.edge_index)[0]).astype('float64')
        features = sp.csr_matrix(data.x).tolil()
        labels = one_hot(data.y).numpy()
        idx_train = torch.squeeze(np.nonzero(data.train_mask), axis=1).tolist()
        idx_val = torch.squeeze(np.nonzero(data.val_mask), axis=1).tolist()
        idx_test = torch.squeeze(np.nonzero(data.test_mask), axis=1).tolist()
        # create graph
        edge_index = data.edge_index
        graph = {}
        for i in range(edge_index.size(1)):
            a = edge_index[0][i].item()
            b = edge_index[1][i].item()
            if a in graph:
                graph[a].append(b)
            else:
                graph[a] = [b]
            if b in graph:
                graph[b].append(a)
            else:
                graph[b] = [a]
        return adj, features, labels, idx_train, idx_val, idx_test, graph

    def load_webKB_data(self, dataset_str):
        data = torch_geometric.datasets.WebKB(root="data/", name=dataset_str)[0]
        create_masks(data)
        adj = sp.csr_matrix(to_dense_adj(data.edge_index)[0]).astype('float64')
        features = sp.csr_matrix(data.x).tolil()
        labels = one_hot(data.y).numpy()
        idx_train = torch.squeeze(np.nonzero(data.train_mask), axis=1).tolist()
        idx_val = torch.squeeze(np.nonzero(data.val_mask), axis=1).tolist()
        idx_test = torch.squeeze(np.nonzero(data.test_mask), axis=1).tolist()
        # create graph
        edge_index = data.edge_index
        graph = {}
        for i in range(edge_index.size(1)):
            a = edge_index[0][i].item()
            b = edge_index[1][i].item()
            if a in graph:
                graph[a].append(b)
            else:
                graph[a] = [b]
            if b in graph:
                graph[b].append(a)
            else:
                graph[b] = [a]
        return adj, features, labels, idx_train, idx_val, idx_test, graph

    def load_data(self, dataset_str):
        # """
        # Loads input data from data directory
        # ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        # ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        # ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        #     (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        # ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        # ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
        # ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        # ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        #     object;
        # ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
        # All objects above must be saved using python pickle module.
        # :param dataset_str: Dataset name
        # :return: All data input files loaded (as well the training/test data).
        # """
        # if dataset_str in ['USA', 'Brazil', 'Europe']:
        #     adj, features, labels, idx_train, idx_val, idx_test, graph = self.load_airport_data(dataset_str)
        # elif dataset_str in ['Cornell', 'Texas', 'Wisconsin']:
        #     adj, features, labels, idx_train, idx_val, idx_test, graph = self.load_webKB_data(dataset_str)
        # else:
        #     names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        #     objects = []
        #     for i in range(len(names)):
        #         with open("data/node/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
        #             if sys.version_info > (3, 0):
        #                 objects.append(pkl.load(f, encoding='latin1'))
        #             else:
        #                 objects.append(pkl.load(f))

        #     x, y, tx, ty, allx, ally, graph = tuple(objects)
        #     test_idx_reorder = parse_index_file("data/node/ind.{}.test.index".format(dataset_str))
        #     test_idx_range = np.sort(test_idx_reorder)
        #     if dataset_str == 'citeseer':  # or dataset_str == 'nell.0.001'
        #         # Fix citeseer dataset (there are some isolated nodes in the graph)
        #         # Find isolated nodes, add them as zero-vecs into the right position
        #         test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        #         tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        #         tx_extended[test_idx_range-min(test_idx_range), :] = tx
        #         tx = tx_extended
        #         ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        #         ty_extended[test_idx_range-min(test_idx_range), :] = ty
        #         ty = ty_extended
        #     features = sp.vstack((allx, tx)).tolil()
        #     features[test_idx_reorder, :] = features[test_idx_range, :]
        #     adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        #     labels = np.vstack((ally, ty))
        #     labels[test_idx_reorder, :] = labels[test_idx_range, :]

        #     idx_test = test_idx_range.tolist()
        #     idx_train = range(len(y))
        #     idx_val = range(len(y), len(y)+500)

        # adj = [[i] for i in range(features.shape[0])]
        # weight = [[1] for i in range(features.shape[0])]
        # for node, neighbor in graph.items():
        #     for n in neighbor:
        #         adj[node].append(n)
        #         weight[node].append(1)

        # max_len = max([len(i) for i in adj])
        # normalize_weight(adj, weight)

        # adj_label = []
        # for i in range(len(adj)):
        #     for j in range(len(adj)):
        #         if j in adj[i]:
        #             adj_label.append(1)
        #         else:
        #             adj_label.append(0)
        # adj = pad_sequence(adj, max_len)
        # weight = pad_sequence(weight, max_len)

        # train_mask = sample_mask(idx_train, labels.shape[0])
        # val_mask = sample_mask(idx_val, labels.shape[0])
        # test_mask = sample_mask(idx_test, labels.shape[0])

        # y_train = np.zeros(labels.shape)
        # y_val = np.zeros(labels.shape)
        # y_test = np.zeros(labels.shape)
        # y_train[train_mask, :] = labels[train_mask, :]
        # y_val[val_mask, :] = labels[val_mask, :]
        # y_test[test_mask, :] = labels[test_mask, :]

        # self.adj = np.array(adj)
        # self.weight = np.array(weight)
        # features = np.array(features.todense().tolist())

        # self.features = preprocess_features(features)
        # self.y_train = y_train
        # self.y_val = y_val
        # self.y_test = y_test
        # self.train_mask = train_mask.astype(int)
        # self.val_mask = val_mask.astype(int)
        # self.test_mask = test_mask.astype(int)
        # # set up paramaters
        # self.args.node_num = features.shape[0]
        # self.args.input_dim = features.shape[1]
        # self.args.num_class = y_train.shape[1]
        # self.adj_label = np.array(adj_label)
        data = torch.load(f'/content/drive/MyDrive/HGNNs_Link_Prediction/data_{dataset_str}')
        self.train = data['train']
        self.val = data['val']
        self.test = data['test']
        edge_index = data['train'].edge_index
        features = sp.lil_matrix(data['train'].x)
        graph = {}
        for i in range(edge_index.size(1)):
            a = edge_index[0][i].item()
            b = edge_index[1][i].item()
            if a in graph:
                graph[a].append(b)
            else:
                graph[a] = [b]
            if b in graph:
                graph[b].append(a)
            else:
                graph[b] = [a]
        adj = [[i] for i in range(features.shape[0])]
        weight = [[1] for i in range(features.shape[0])]
        for node, neighbor in graph.items():
            for n in neighbor:
                adj[node].append(n)
                weight[node].append(1)
        max_len = max([len(i) for i in adj])
        adj_label = []
        for i in range(len(adj)):
            for j in range(len(adj)):
                if j in adj[i]:
                    adj_label.append(1)
                else:
                    adj_label.append(0)
        adj = pad_sequence(adj, max_len)
        weight = pad_sequence(weight, max_len)
        self.adj = np.array(adj)
        self.weight = np.array(weight)
        features = np.array(features.todense().tolist())
        self.features = features
        self.edge_label_index_train = data['train'].edge_label_index
        self.edge_label_index_val = data['val'].edge_label_index
        self.edge_label_index_test = data['test'].edge_label_index
        self.edge_label_val = data['val'].edge_label
        self.edge_label_test = data['test'].edge_label
        # set up paramaters
        self.args.node_num = features.shape[0]
        self.args.input_dim = features.shape[1]
        self.adj_label = np.array(adj_label)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return  {
                  'adj': self.adj,
                  'weight': self.weight,
                  'features': self.features,
                  'edges_train': self.edge_label_index_train,
                  'edges_val': self.edge_label_index_val,
                  'edges_test': self.edge_label_index_test,
                  'edge_label_val': self.edge_label_val,
                  'edge_label_test': self.edge_label_test,
                }
