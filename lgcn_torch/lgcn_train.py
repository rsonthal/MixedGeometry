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
import warnings

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
    warnings.filterwarnings('ignore')
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

    #logging.info(f'Using cuda: {args.cuda}')
    #logging.info("Using seed {}.".format(args.seed))
    #logging.info("Using dataset {}.".format(args.dataset))

    # Load data
    data = load_data(args, os.path.join('data/', args.dataset))
    # print(data['idx_test'])
    args.n_nodes, args.feat_dim = data['features'].shape
    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
        #logging.info(f'Num classes: {args.n_classes}')
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
    filename = "dataset=" + args.dataset + "_layers=" + str(num_layer) + \
              "_dims=" + str(d) + "_lr=" + str(lr) + ".pt"
    model_weights = []
    aucs = []
    for iter_i in range(1,args.run_times+1):
        # Model and optimizer
        model = Model(args)
        if args.cuda is not None and int(args.cuda) >= 0:
            model = model.to(args.device)
            for x, val in data.items():
                if torch.is_tensor(data[x]):
                    data[x] = data[x].to(args.device)
        iter_i = str(iter_i)
        optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(args.lr_reduce_freq),
            gamma=float(args.gamma)
        )
        # Train model
        t_total = time.time()
        counter = 0
        best_train_metrics = None
        best_val_metrics = model.init_metric_dict()
        best_test_metrics = None
        best_emb = None
        best_model = None
        for epoch in range(args.epochs):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            embeddings = model.encode(data['features'], data['adj_train_norm'])
            train_metrics = model.compute_metrics(embeddings, data, 'train')
            train_metrics['loss'].backward()
            if args.grad_clip is not None:
                max_norm = float(args.grad_clip)
                all_params = list(model.parameters())
                for param in all_params:
                    torch.nn.utils.clip_grad_norm_(param, max_norm)
            optimizer.step()
            lr_scheduler.step()
            # if (epoch + 1) % args.log_freq == 0:
                # logging.info(" ".join(['{} {} Epoch: {} {:04d}'.format(args.dataset, args.task, iter_i, epoch + 1),
                                      #  'lr: {}'.format(lr_scheduler.get_last_lr()[0]),
                                      #  format_metrics(train_metrics, 'train'),
                                      #  'time: {:.4f}s'.format(time.time() - t)
                                      #  ]))
            if (epoch + 1) % args.eval_freq == 0:
                model.eval()
                embeddings = model.encode(data['features'], data['adj_train_norm'])
                val_metrics = model.compute_metrics(embeddings, data, 'val')
                test_metrics = model.compute_metrics(embeddings, data, 'test')
                #if (epoch + 1) % args.log_freq == 0:
                    #logging.info(" ".join(['{} {} Epoch: {} {:04d}'.format(args.dataset, args.task, iter_i, epoch + 1), format_metrics(val_metrics, 'val')]))
                if model.has_improved(best_val_metrics, val_metrics):
                    best_train_metrics = model.compute_metrics(embeddings, data, 'train')
                    best_test_metrics = model.compute_metrics(embeddings, data, 'test')
                    best_emb = embeddings.cpu()
                    best_model = model
                    #if args.save:
                        #np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().numpy())
                    best_val_metrics = val_metrics
                    counter = 0
                else:
                    counter += 1
                    #if counter == args.patience and epoch > args.min_epochs:
                        #logging.info("Early stopping")
                        #break
            
            trial = int(iter_i)
            #print(train_metrics)
            #print(test_metrics)
            #print(model_name, num_layer, d, lr, trial, epoch+1)
            #dic[model_name][num_layer][d][lr][trial][epoch+1]['auc_train'] = train_metrics['roc']
            #dic[model_name][num_layer][d][lr][trial][epoch+1]['auc_val'] = val_metrics['roc']
            #dic[model_name][num_layer][d][lr][trial][epoch+1]['auc_test'] = test_metrics['roc']

        #logging.info("Optimization Finished!")
        #logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        if not best_test_metrics:
            model.eval()
            best_emb = model.encode(data['features'], data['adj_train_norm'])
            best_test_metrics = model.compute_metrics(best_emb, data, 'test')
        #logging.info(" ".join(["Val set results:", format_metrics(best_val_metrics, 'val')]))
        #logging.info(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))

        #aucs.append((best_train_metrics['roc'], best_val_metrics['roc'], best_test_metrics['roc']))
        model_weights.append(best_model.state_dict())

        #if args.save:
            #print('Saved path', os.path.join(save_dir, 'embeddings.npy'))
            #np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
            #if hasattr(model.encoder, 'att_adj'):
                #filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
                #pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
                #print('Dumped attention adj: ' + filename)
            #json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))
            #torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
            #logging.info(f"Saved model in {save_dir}")

    #export_dict(args.dataset, model_name, num_layer, d, lr, dic)
    #torch.save(model_weights, "/content/myDrive/MyDrive/HGNNs_Link_Prediction/" + model_name + "/models/" + filename)
    #torch.save(aucs, "/content/myDrive/MyDrive/HGNNs_Link_Prediction/" + model_name + "/results/" + filename)
    # print("Saved json file")

if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
