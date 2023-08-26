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
from dataset.NodeClassificationDataset import NodeClassificationDataset
from task.NodeClassification import NodeClassification
import time
import json

# def initialize_empty_dict(args):
#   d ={}
#   trials = 10
#   epochs = 100
#   layers = args.gnn_layer
#   dim = args.embed_size-1
#   lr = args.lr
#   model = 'HGNN'
#   trial = args.trial_number
#   # print("Initializing", model, layers, dim, lr)
#   d[model] = {}
#   d[model][layers]={}
#   d[model][layers][dim] = {}
#   d[model][layers][dim][lr] ={}
#   d[model][layers][dim][lr][trial] = {}
#   for e in range(1,epochs+1):
#     d[model][layers][dim][lr][trial][e] ={}
#     for acc in ['acc_train', 'acc_val', 'acc_test']:
#       d[model][layers][dim][lr][trial][e][acc] = {}
#       d[model][layers][dim][lr][trial][e][acc] = 0
#   # print("Init", d['LGCN'][2][32][0.0002][1][1]['acc_train'])

#   return d

# def export_dict(data_name, model_name, num_layers, dim, lr, trial, d):
#   file_name = f'../HGNNs/models_report_2/{data_name}_{model_name}_{num_layers}_{dim}_{lr}_{trial}_report.json'
#   with open(file_name, "w") as write_file:
#     json.dump(d, write_file, indent = "")
#   print("Saved json file")

def cross_entropy(log_prob, label, mask):
	label, mask = label.squeeze(), mask.squeeze()
	negative_log_prob = -th.sum(label * log_prob, dim=1)
	return th.sum(mask * negative_log_prob, dim=0) / th.sum(mask)

def get_accuracy(label, log_prob, mask):
	label = label.squeeze()
	pred_class = th.argmax(log_prob, dim=1)
	real_class = th.argmax(label, dim=1)
	acc = th.eq(pred_class, real_class).float() * mask
	return (th.sum(acc) / th.sum(mask)).cpu().detach().numpy()

class NodeClassificationTask(BaseTask):

	def __init__(self, args, logger, rgnn, manifold):
		super(NodeClassificationTask, self).__init__(args, logger, criterion='max')
		self.args = args
		self.logger = logger
		self.manifold = manifold
		self.hyperbolic = False if args.select_manifold == "euclidean" else True
		self.rgnn = rgnn

	def forward(self, model, sample, loss_function):
		scores = model(
					sample['adj'].cuda().long(),
			        sample['weight'].cuda().float(),
			        sample['features'].cuda().float(),
					)
		loss = loss_function(scores,
						 sample['y_train'].cuda().float(), 
						 sample['train_mask'].cuda().float()) 
		return scores, loss

	def run_gnn(self, trial):
		loader = self.load_data()

		model = NodeClassification(self.args, self.logger, self.rgnn, self.manifold).cuda()

		loss_function = cross_entropy

		optimizer, lr_scheduler, hyperbolic_optimizer, hyperbolic_lr_scheduler = \
								set_up_optimizer_scheduler(self.hyperbolic, self.args, model)

		args = self.args
		#d = initialize_empty_dict(args)
		best_model = None
		best_val = 0.0
		best_accs = None
		num_layer = args.gnn_layer
		dim = args.embed_size-1
		lr = args.lr
		trial = args.trial_number
		model_name = 'HGNN'
		# filename = "dataset=" + 'WebKB-' + args.dataset_str + "_layers=" + str(num_layer) + \
    #           "_dims=" + str(dim) + "_lr=" + str(lr) + "_trial=" + str(trial) + "_set2" + ".pt"
		for epoch in range(self.args.max_epochs):
			model.train()
			for i, sample in enumerate(loader):
				model.zero_grad()
				scores, loss = self.forward(model, sample, loss_function)
				loss.backward()
				if self.args.grad_clip > 0.0:
					th.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip)
				optimizer.step()
				if self.hyperbolic and len(self.args.hyp_vars) != 0:
					hyperbolic_optimizer.step()
				accuracy = get_accuracy(
									sample['y_train'].cuda().float(), 
									scores, 
									sample['train_mask'].cuda().float())
				# self.logger.info("%s epoch %d: accuracy %.4f \n" % (
				# 	'train', 
				# 	epoch, 
				# 	accuracy))
				# th.save(accuracy, "../HGNN_comparison/PubMed/HGNN/results/train_trial"+str(trial)+"of10_epoch"+str(epoch+1)+"of100.pt")
			# th.save(model, "../HGNN_comparison/PubMed/HGNN/trial"+str(trial)+"of10_epoch"+str(epoch+1)+"of100.pt")
			dev_acc = self.evaluate(trial, epoch, loader, 'dev', model, loss_function)
			test_acc = self.evaluate(trial, epoch, loader, 'test', model, loss_function)
			#d[model_name][num_layer][dim][lr][int(trial)][int(epoch+1)]['acc_train'] = accuracy.item(0)
			#d[model_name][num_layer][dim][lr][int(trial)][int(epoch+1)]['acc_val'] = dev_acc.item(0)
			#d[model_name][num_layer][dim][lr][int(trial)][int(epoch+1)]['acc_test'] = test_acc.item(0)
			if dev_acc.item(0) > best_val:
				best_val = dev_acc.item(0)
				best_model = model
				best_accs = (accuracy.item(0), dev_acc.item(0), test_acc.item(0))
			lr_scheduler.step()
			if self.hyperbolic and len(self.args.hyp_vars) != 0:
				hyperbolic_lr_scheduler.step()
			if not self.early_stop.step(dev_acc, test_acc, epoch):		
				break
		self.report_best()
		#export_dict('WebKB-' + args.dataset_str, model_name, num_layer, dim, lr, trial, d)
		#th.save(best_model.state_dict(), "../HGNNs/" + model_name + "/models/" + filename)
		#th.save(best_accs, "../HGNNs/" + model_name + "/results/" + filename)
			
	def evaluate(self, trial, epoch, data_loader, prefix, model, loss_function):
		model.eval()
		with th.no_grad():
			for i, sample in enumerate(data_loader):
				scores, _ = self.forward(model, sample, loss_function)
				if prefix == 'dev':
					accuracy = get_accuracy(
									sample['y_val'].cuda().float(), 
									scores, 
									sample['val_mask'].cuda().float())
				elif prefix == 'test':
					accuracy = get_accuracy(
									sample['y_test'].cuda().float(), 
									scores, 
									sample['test_mask'].cuda().float())
					# th.save(accuracy, "../HGNN_comparison/PubMed/HGNN/results/test_trial"+str(trial)+"of10_epoch"+str(epoch+1)+"of100.pt")
				# if prefix == 'test' and epoch%1==0:
				# 	self.logger.info("%s epoch %d: accuracy %.4f \n" % (
				# 		prefix, 
				# 		epoch, 
				# 		accuracy))
		return accuracy

	def load_data(self):
		dataset = NodeClassificationDataset(self.args, self.logger)
		return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

