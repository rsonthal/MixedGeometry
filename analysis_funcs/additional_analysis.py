"""
This document contains defintions of additional functions 
used to analyze results from the appendix. Specifically,
it studies the sensitivity to hyper-parameters.
"""

import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# tensor containing node classification training values
tensor = np.load('link_prediction_tensor.npy')

# defintions for indexing

dataset_categories = {
    'Airports': ['AirportUSA', 'AirportEurope', 'AirportBrazil'],
    'Publications': ['Cora', 'PubMed', 'CiteSeer'],
    'WebKB': ['WebKB-Texas', 'WebKB-Cornell', 'WebKB-Wisconsin']
}

model_categories = {
    'Attention Network':['GAT', 'HGAT', 'HyboNet'],
    'Convolution Network': ['GCN', 'HGCN', 'LGCN'],
    'Graph Convolution': ['GC', 'HGC', 'HGNN']
}

dataset_indices = {
    'AirportBrazil': 0,
    'AirportEurope': 1,
    'AirportUSA': 2,
    'CiteSeer': 3,
    'Cora': 4,
    'PubMed': 5,
    'WebKB-Cornell': 6,
    'WebKB-Texas': 7,
    'WebKB-Wisconsin': 8
}

model_indices = {
    'GAT': 0,
    'GC': 1,
    'GCN': 2,
    'HGAT': 3,
    'HGC': 4,
    'HGCN': 5,
    'HGNN': 6,
    'HyboNet': 7,
    'LGCN': 8
}

layers = [2,3,4]
dimensions = [32,64,128,256]
learning_rates = [0.0002,0.001,0.005]


configs = {}
configs['keys'] = ['layers', 'dimensions', 'learning_rates']
i = 0
for layer in [2,3,4]:
    for dim in [32,64,128,256]:
        for lr in [0.0002,0.001,0.005]:
            configs[i] = [layer,dim,lr]
            i+=1


def stability_across_all_configs(dir):
    '''
    Returns a dataframe containing the average standard deviation of test accuracies across configurations
    for every combination of dataset and model. 

    Parameters:
    dir (str): file path containing the CSVs with the structure of 
    ...\\{dataset}\\{dataset}_{model}_report.csv

    Returns:
    dataframe: the desired table
    '''
    df_main = pd.DataFrame({'model_name': ['GAT','HGAT','HyboNet', 'GC','HGC', 'HGNN','GCN', 'HGCN', 'LGCN']})
    for dataset in dataset_indices.keys():
        directory = f'{dir}{dataset}\\'
        df_dataset = pd.DataFrame()
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            df = pd.read_csv(f, index_col = 'Unnamed: 0')
            if (dataset == 'PubMed'):
                df = df[df['num_layers'] == 2]
            df_dataset = pd.concat([df,df_dataset], ignore_index = True)

        df_epoch_max = df_dataset[df_dataset['acc_val'] == df_dataset.groupby(['model_name', 'num_layers', 'dimensions', 'learning_rate','trial'])['acc_val'].transform(max)].drop_duplicates(['model_name', 'num_layers', 'dimensions', 'learning_rate','trial'], keep = 'first')
        df_avg_acc_val_per_config = df_epoch_max.groupby(['model_name', 'num_layers', 'dimensions', 'learning_rate']).aggregate({'acc_val': ['mean', np.std], 'acc_test': ['mean', np.std]}).reset_index()
        df_avg_acc_val_per_config.columns = df_avg_acc_val_per_config.columns.map('_'.join).str.strip('_') 
        df_std_across_configs_per_model = df_avg_acc_val_per_config.groupby(['model_name']).agg({'acc_test_mean': np.std}).reset_index()
        df_std_across_configs_per_model[str(dataset)] = df_std_across_configs_per_model['acc_test_mean'].round(3)

        mask = (df_std_across_configs_per_model['model_name'] != 'RGGC') & (df_std_across_configs_per_model['model_name'] != 'HRGGC')
        df_std_across_configs_per_model = df_std_across_configs_per_model[mask].reset_index(drop = True)

        df_std_across_configs_per_model = df_std_across_configs_per_model.drop(columns = ['acc_test_mean'])
        df_main = df_main.merge(df_std_across_configs_per_model, how = 'left')
    return df_main



def stability_agg_geometry(dir):
    '''
    Aggregates the values from the stability_across_all_configs function
    across geometries so for each dataset, the results from Euclidean,
    Mixed, and Hyperbolic can be compared.

    Parameters:
    dir (str): file path containing the CSVs with the structure of 
    ...\\{dataset}\\{dataset}_{model}_report.csv

    Returns:
    dataframe: the desired table
    '''

    df_stability_across_configs = stability_across_all_configs(dir)

    data = {'model_name': ['Euclidean', 'Mixed', 'Hyperbolic']}

    for dataset_name in list(df_stability_across_configs.columns[1:]):
        vals = []
        for j in [0,1,2]:
            vals.append(np.round(np.mean([float(np.array(df_stability_across_configs[dataset_name])[i]) for i in [0+j,3+j,6+j]]),2))

        data[dataset_name] = vals

    df_agg_best_acc = pd.DataFrame(data)

    return df_agg_best_acc


def stability_agg_architecture(dir):
    '''
    Aggregates the values from the stability_across_all_configs function
    across architectures so for each dataset, the results from GraphConv,
    Convolution Network, and Attention Networks can be compared.

    Parameters:
    dir (str): file path containing the CSVs with the structure of 
    ...\\{dataset}\\{dataset}_{model}_report.csv

    Returns:
    dataframe: the desired table
    '''

    df_stability_across_configs = stability_across_all_configs(dir)

    data = {'model_name': ['Attention Network', 'Graph Convolution', 'Convolution Network']}

    for dataset_name in list(df_stability_across_configs.columns[1:]):
        vals = []
        for j in [0,3,6]:
            vals.append(np.round(np.max([float(np.array(df_stability_across_configs[dataset_name])[i]) for i in [0+j,1+j,2+j]]),2))

        data[dataset_name] = vals

    df_agg_best_acc = pd.DataFrame(data)

    return df_agg_best_acc


def stability_agg_models(dir):
    '''
    Aggregates and prints the values from stability_across_all_configs
    function across datasets so for each model, results can be compared.

    Parameters:
    dir (str): file path containing the CSVs with the structure of 
    ...\\{dataset}\\{dataset}_{model}_report.csv

    Returns:
    dataframe: the desired table
    '''
    df_stability_across_configs = stability_across_all_configs(dir)

    for i in range(9):
        print(df_stability_across_configs['model_name'][i],
        np.round(df_stability_across_configs.mean(axis=1)[i],4))

    return
