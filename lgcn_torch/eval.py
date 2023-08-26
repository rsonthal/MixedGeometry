from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time

import numpy as np
import optimizers
import torch
import torch.nn as nn
from config import parser
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics
import json

def initialize_empty_dict(args):
  d ={}
  trials = 10
  epochs = 200
  layers = args.num_layers
  dim = args.dim-1
  lr = args.lr
  model = 'LGCN'
  # print("Initializing", model, layers, dim, lr)
  d[model] = {}
  d[model][layers]={}
  d[model][layers][dim] = {}
  d[model][layers][dim][lr] ={}
  for trial in range(1,trials+1):
    d[model][layers][dim][lr][trial] ={}
    for e in range(1,epochs+1):
      d[model][layers][dim][lr][trial][e] ={}
      for acc in ['auc_train', 'auc_val', 'auc_test']:
        d[model][layers][dim][lr][trial][e][acc] = {}
        d[model][layers][dim][lr][trial][e][acc] = 0
  # print("Init", d['LGCN'][2][32][0.0002][1][1]['acc_train'])

  return d

def export_dict(data_name, model_name, num_layers, dim, lr, d):
  file_name = f'/content/myDrive/MyDrive/HGNNs_Link_Prediction/models_report/{data_name}_{model_name}_{num_layers}_{dim}_{lr}_report.json'
  with open(file_name, "w") as write_file:
    json.dump(d, write_file, indent = "")
  print("Saved json file")

def train(args):
    if args.dataset in ['disease_lp', 'disease_nc']:
        args.normalize_feats = 0
    if args.task == 'nc':
    	args.num_layers += 1
    if args.manifold == 'Lorentzian' or 'Euclidean':
        args.dim = args.dim + 1
    args.c = float(args.c) if args.c != None else None
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join('save', args.task, date)
            save_dir = get_dir_name(models_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        else:
            save_dir = args.save_dir
        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                logging.StreamHandler()
                            ])

    logging.info(f'Using cuda: {args.cuda}')
    logging.info("Using seed {}.".format(args.seed))
    logging.info("Using dataset {}.".format(args.dataset))

    # Load data
    data = load_data(args, os.path.join('data/', args.dataset))
    # print(data['idx_test'])
    args.n_nodes, args.feat_dim = data['features'].shape
    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
        logging.info(f'Num classes: {args.n_classes}')
    else:
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        if args.task == 'lp':
            Model = LPModel
        else:
            raise NotImplementedError
            Model = RECModel
            # No validation for reconstruction task
            args.eval_freq = args.epochs + 1

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    dic = initialize_empty_dict(args)
    num_layer = args.num_layers
    d = args.dim-1
    lr = args.lr
    model_name = 'LGCN'
    # dataset2name = {'cora':'Cora', 'citeseer':'CiteSeer', 'pubmed':'PubMed',\
    #                 'USA':'AirportUSA', 'Brazil':'AirportBrazil', 'Europe':'AirportEurope',\
    #                 'Cornell':'WebKB-Cornell', 'Texas':'WebKB-Texas', 'Wisconsin':'WebKB-Wisconsin'}
    filename = "dataset=" + args.dataset + "_layers=" + str(num_layer) + \
              "_dims=" + str(d) + "_lr=" + str(lr) + ".pt"
    model_weights = []
    aucs = []
    # Model and optimizer
    model = Model(args)
    model_dicts = torch.load(f'../HGNNs_Link_Prediction/{model_name}/models/{filename}')
    # print(filename)
    if args.cuda is not None and int(args.cuda) >= 0:
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)
    for iter_i in range(1,args.run_times+1):
        model.eval()
        model.load_state_dict(model_dicts[iter_i-1])
        model = model.to(args.device)
        embeddings = model.encode(data['features'], data['adj_train_norm'])
        prf = model.compute_metrics_prf(embeddings, data, 'test')
        print(prf)
        torch.save(prf, f'../HGNNs_Link_Prediction/{model_name}/results/prf/{filename}'.replace('.pt', f'_trial={iter_i}.pt'))
if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
