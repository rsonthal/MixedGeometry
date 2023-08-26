#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils import * 
from torch.utils.data import DataLoader
import torch.optim as optim
from task.BaseTask import BaseTask
import numpy as np
from dataset.LinkPredictionDataset import LinkPredictionDataset
from task.LinkPrediction import LinkPrediction
import time
import json
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

def initialize_empty_dict(args):
  d ={}
  trials = 10
  epochs = 200
  layers = args.gnn_layer
  dim = args.embed_size-1
  lr = args.lr
  model = 'HGNN'
  trial = args.trial_number
  # print("Initializing", model, layers, dim, lr)
  d[model] = {}
  d[model][layers]={}
  d[model][layers][dim] = {}
  d[model][layers][dim][lr] ={}
  d[model][layers][dim][lr][trial] = {}
  for e in range(1,epochs+1):
    d[model][layers][dim][lr][trial][e] ={}
    for acc in ['auc_train', 'auc_val', 'auc_test']:
      d[model][layers][dim][lr][trial][e][acc] = {}
      d[model][layers][dim][lr][trial][e][acc] = 0
  # print("Init", d['LGCN'][2][32][0.0002][1][1]['acc_train'])

  return d

def export_dict(data_name, model_name, num_layers, dim, lr, trial, d):
  file_name = f'../HGNNs_Link_Prediction/models_report/{data_name}_{model_name}_{num_layers}_{dim}_{lr}_{trial}_report.json'
  with open(file_name, "w") as write_file:
    json.dump(d, write_file, indent = "")
  print("Saved json file")

def cross_entropy(log_prob, label, mask):
	label, mask = label.squeeze(), mask.squeeze()
	negative_log_prob = -th.sum(label * log_prob, dim=1)
	return th.sum(mask * negative_log_prob, dim=0) / th.sum(mask)

def get_accuracy(label, logits):
  auc = roc_auc_score(label.detach().cpu().numpy(), logits.sigmoid().detach().cpu().numpy())
	# label = label.squeeze()
	# pred_class = th.argmax(log_prob, dim=1)
	# real_class = th.argmax(label, dim=1)
	# acc = th.eq(pred_class, real_class).float() * mask
	# return (th.sum(acc) / th.sum(mask)).cpu().detach().numpy()
  return auc

def get_prf(label, logits):
	label = label.detach().cpu().numpy()
	probs = logits.sigmoid()
	preds = th.where(probs > 0.2, 1, 0).detach().cpu().numpy()
	auc = roc_auc_score(label, logits.sigmoid().detach().cpu().numpy())
	# print(len(label))
	# print(preds)

	p = precision_score(label, preds, zero_division=0.0)
	r = recall_score(label, preds, zero_division=0.0)
	f = f1_score(label, preds, zero_division=0.0)
	return p, r, f

class LinkPredictionTaskEval(BaseTask):

	def __init__(self, args, logger, rgnn, manifold):
		super(LinkPredictionTaskEval, self).__init__(args, logger, criterion='max')
		self.args = args
		self.logger = logger
		self.manifold = manifold
		self.hyperbolic = False if args.select_manifold == "euclidean" else True
		self.rgnn = rgnn

	def forward(self, model, sample, prefix):
		if prefix == 'train':
			pos_edge_index = sample['edges_train'].squeeze(dim=0)
			neg_edge_index = negative_sampling(
				edge_index=pos_edge_index, #positive edges
				num_nodes=sample['features'].size(1), # number of nodes
				num_neg_samples=pos_edge_index.size(1)) # number of neg_sample equal to number of pos_edges
			link_labels = th.zeros(pos_edge_index.size(1)*2, dtype=th.float, device=th.device('cuda'))
			link_labels[:pos_edge_index.size(1)] = 1.
		else:
			pos_edge_index, neg_edge_index = sample[f'edges_{prefix}'].squeeze(dim=0).chunk(2, dim=1)
			link_labels = sample[f'edge_label_{prefix}'].squeeze(dim=0).cuda()
		scores = model(
				sample['adj'].cuda().long(),
				sample['weight'].cuda().float(),
				sample['features'].cuda().float(),
				pos_edge_index.cuda().float(),
				neg_edge_index.cuda().float(),
              )
		# loss = loss_function(scores,
		# 				 sample['y_train'].cuda().float(), 
		# 				 sample['train_mask'].cuda().float()) 
		loss = F.binary_cross_entropy_with_logits(scores, link_labels)
		return scores, loss

	def run_gnn(self, trial):
		loader = self.load_data()

		model = LinkPrediction(self.args, self.logger, self.rgnn, self.manifold).cuda()

		# loss_function = cross_entropy

		optimizer, lr_scheduler, hyperbolic_optimizer, hyperbolic_lr_scheduler = \
								set_up_optimizer_scheduler(self.hyperbolic, self.args, model)

		args = self.args
		d = initialize_empty_dict(args)
		best_model = None
		best_val = 0.0
		best_aucs = None
		num_layer = args.gnn_layer
		dim = args.embed_size-1
		lr = args.lr
		trial = args.trial_number
		model_name = 'HGNN'
		filename = "dataset=" + args.dataset_str + "_layers=" + str(num_layer) + \
              "_dims=" + str(dim) + "_lr=" + str(lr) + "_trial=" + str(trial) + ".pt"
		model_dict = th.load(f'/content/drive/MyDrive/HGNNs_Link_Prediction/{model_name}/models/{filename}')
		model.load_state_dict(model_dict)
		model.eval()
			# scores, loss = self.forward(model, sample, loss_function)
				# th.save(accuracy, "../HGNN_comparison/PubMed/HGNN/results/train_trial"+str(trial)+"of10_epoch"+str(epoch+1)+"of100.pt")
			# th.save(model, "../HGNN_comparison/PubMed/HGNN/trial"+str(trial)+"of10_epoch"+str(epoch+1)+"of100.pt")
		epoch = 200
		p, r, f = self.evaluate(trial, epoch, loader, 'test', model)
		prf = {'precision':p, 'recall':r, 'f1':f}
		print(prf)
		print(filename)
		th.save(prf, f'/content/drive/MyDrive/HGNNs_Link_Prediction/{model_name}/results/prf/{filename}')

	def evaluate(self, trial, epoch, data_loader, prefix, model):
		model.eval()
		with th.no_grad():
			for i, sample in enumerate(data_loader):
				scores, _ = self.forward(model, sample, prefix)
				# auc = get_accuracy(sample[f'edge_label_{prefix}'].squeeze(0), scores)
				# if prefix == 'test':
				# 	self.logger.info("%s epoch %d: AUC %.4f \n" % (
				# 		prefix, 
				# 		epoch, 
				# 		auc))
				p, r, f = get_prf(sample[f'edge_label_{prefix}'].squeeze(0), scores)
		return p, r, f

	def load_data(self):
		dataset = LinkPredictionDataset(self.args, self.logger)
		return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

