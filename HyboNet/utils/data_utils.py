"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric
from torch_geometric.utils import to_dense_adj
from math import ceil


def load_data(args, datapath):
    if args.task == 'nc':
        data = load_data_nc(args.dataset, args.use_feats, datapath, args.split_seed)
    else:
        # data = load_data_lp(args.dataset, args.use_feats, datapath)
        # adj = data['adj_train']
        # if args.task == 'lp':
            # print('start to mask edges')
            # adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
            #         adj, args.val_prop, args.test_prop, args.split_seed
            # )
            # data['adj_train'] = adj_train
            # data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
            # data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
            # data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
        
        data = torch.load(f'/content/myDrive/MyDrive/HGNNs_Link_Prediction/data_{args.dataset}')
        data = split_edges(data)
    
    data['adj_train_norm'], data['features'] = process(
            data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
    )
    if args.dataset == 'airport':
        data['features'] = augment(data['adj_train'], data['features'])
    return data


# ############### FEATURES PROCESSING ####################################


def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


# ############### DATA SPLITS #####################################################

def split_edges(data):
    data_new = {}
    num_nodes = data['train'].num_nodes
    train_edge_label_index = data['train'].edge_label_index  # positive only
    val_edge_label_index = data['val'].edge_label_index  # positive+negative
    test_edge_label_index = data['test'].edge_label_index  # positive+negative
    
    data_new['adj_train'] = to_dense_adj(train_edge_label_index, max_num_nodes=num_nodes)[0]
    data_new['features'] = data['train'].x
    data_new['train_edges'] = torch.transpose(train_edge_label_index, 0, 1)
    data_new['train_edges_false'] = (torch.ones((num_nodes, num_nodes)) - data_new['adj_train']).nonzero()
    val_edges, val_edges_false = val_edge_label_index.chunk(2, dim=1)
    data_new['val_edges'] = torch.transpose(val_edges, 0, 1)
    data_new['val_edges_false'] = torch.transpose(val_edges_false, 0, 1)
    test_edges, test_edges_false = test_edge_label_index.chunk(2, dim=1)
    data_new['test_edges'] = torch.transpose(test_edges, 0, 1)
    data_new['test_edges_false'] = torch.transpose(test_edges_false, 0, 1)
    
    data_new['adj_train'] = sp.csr_matrix(data_new['adj_train'])
    data_new['features'] = sp.lil_matrix(data_new['features'])

    return data_new

def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)  


def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


# ############### LINK PREDICTION DATA LOADERS ####################################


def load_data_lp(dataset, use_feats, data_path):
    if dataset in ['cora', 'pubmed']:
        adj, features = load_citation_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'disease_lp':
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'airport':
        adj, features = load_data_airport(dataset, data_path, return_label=False)
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    data = {'adj_train': adj, 'features': features}
    return data


# ############### NODE CLASSIFICATION DATA LOADERS ####################################


def load_data_nc(dataset, use_feats, data_path, split_seed):
    if dataset in ['cora', 'pubmed', 'citeseer']:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
            dataset, use_feats, data_path, split_seed
        )
    # else:
    #     if dataset == 'disease_nc':
    #         adj, features, labels = load_synthetic_data(dataset, use_feats, data_path)
    #         val_prop, test_prop = 0.10, 0.60
    #     elif dataset == 'airport':
    #         adj, features, labels = load_data_airport(dataset, data_path, return_label=True)
    #         val_prop, test_prop = 0.15, 0.15
    #     else:
    #         raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    #     idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)
        labels = torch.LongTensor(labels)
    elif dataset in ['USA', 'Brazil', 'Europe']:
        adj, features, labels, idx_train, idx_val, idx_test = load_airport_data(dataset)
    elif dataset in ['Cornell', 'Texas', 'Wisconsin']:
        adj, features, labels, idx_train, idx_val, idx_test = load_webKB_data(dataset)
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
    return data


# ############### DATASETS ####################################


def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + min(1000, len(labels) - len(y) - len(idx_test)))

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not use_feats:
        features = sp.eye(adj.shape[0])
    return adj, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels


# def load_data_airport(dataset_str, data_path, return_label=False):
#     graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
#     adj = nx.adjacency_matrix(graph)
    
#     features = np.array([graph.nodes[u]['feat'] for u in graph.nodes()])
#     if return_label:
#         label_idx = 4
#         labels = features[:, label_idx]
#         features = features[:, :label_idx]
#         labels = bin_feat(labels, bins=[7.0/7, 8.0/7, 9.0/7])
#         return sp.csr_matrix(adj), features, labels
#     else:
#         return sp.csr_matrix(adj), features

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

def load_airport_data(dataset_str):
    data = torch_geometric.datasets.Airports(root="data/", name=dataset_str)[0]
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
    labels = data.y
    idx_train = torch.squeeze(np.nonzero(data.train_mask), axis=1).tolist()
    idx_val = torch.squeeze(np.nonzero(data.val_mask), axis=1).tolist()
    idx_test = torch.squeeze(np.nonzero(data.test_mask), axis=1).tolist()
    return adj, features, labels, idx_train, idx_val, idx_test

def load_webKB_data(dataset_str):
    data = torch_geometric.datasets.WebKB(root="data/", name=dataset_str)[0]
    create_masks(data)
    adj = sp.csr_matrix(to_dense_adj(data.edge_index)[0]).astype('float64')
    features = sp.csr_matrix(data.x).tolil()
    labels = data.y
    idx_train = torch.squeeze(np.nonzero(data.train_mask), axis=1).tolist()
    idx_val = torch.squeeze(np.nonzero(data.val_mask), axis=1).tolist()
    idx_test = torch.squeeze(np.nonzero(data.test_mask), axis=1).tolist()
    return adj, features, labels, idx_train, idx_val, idx_test