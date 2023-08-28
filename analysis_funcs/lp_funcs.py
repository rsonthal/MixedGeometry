"""
This document contains defintions of functions used for 
the analysis of the link prediction data.

Notes:
- Line 25 requires access to link_prediction_tensor.py
which can be found in this github repository
- Additionally, a file path to the CSVs found on the github
repository is required with the same structure
- Some of the functions are defined using the CSVs while
some make use of the tensor. Note that the data is identical
in both

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

def auc_for_best_configs(dir):
    '''
    Returns a dataframe containing the average auc for the best performing
    hyper-parameter configuration for each model on each dataset

    Process:
    - For each model/dataset/config, the val_auc is averaged across trials
    - The config with the highest val_auc is chosen
    - The test_auc for each trial of the best config is averaged

    Parameters:
    dir (str): file path containing the CSVs with the structure of 
    ...\\{dataset}\\{dataset}_{model}_report.csv

    Returns:
    dataframe: the desired table
    '''
    df_main = pd.DataFrame({'model_name': ['GAT','HGAT','HyboNet', 'GC','HGC','HGNN','GCN', 'HGCN', 'LGCN']})
    for dataset in dataset_indices.keys():
        directory = f'{dir}{dataset}\\'
        df_dataset = pd.DataFrame()
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            df = pd.read_csv(f, index_col = 'Unnamed: 0')
            if (dataset == 'PubMed'):
                df = df[df['num_layers'] == 2]
            df_dataset = pd.concat([df,df_dataset], ignore_index = True)
    
        df_epoch_max = df_dataset[df_dataset['auc_val'] == df_dataset.groupby(['model_name', 'num_layers', 'dimensions', 'learning_rate','trial'])['auc_val'].transform(max)].drop_duplicates(['model_name', 'num_layers', 'dimensions', 'learning_rate','trial'], keep = 'first')
        df_avg_auc_val_per_config = df_epoch_max.groupby(['model_name', 'num_layers', 'dimensions', 'learning_rate']).aggregate({'auc_val': ['mean', np.std], 'auc_test': ['mean', np.std]}).reset_index()
        df_avg_auc_val_per_config.columns = df_avg_auc_val_per_config.columns.map('_'.join).str.strip('_')
        df_best_config = df_avg_auc_val_per_config[df_avg_auc_val_per_config['auc_val_mean'] == df_avg_auc_val_per_config.groupby(['model_name'])['auc_val_mean'].transform(max)]
        df_best_config = df_best_config[df_best_config['auc_val_std'] == df_best_config.groupby(['model_name'])['auc_val_std'].transform(max)]
        df_best_config = df_best_config[df_best_config.duplicated(subset = ['model_name']) == False]
        df_best_config['auc_test_mean'] = df_best_config['auc_test_mean'].round(2).astype(str)
        df_best_config['auc_test_std'] = df_best_config['auc_test_std'].round(2).astype(str)
        df_best_config[str(dataset)] = df_best_config['auc_test_mean']

        df_best_config = df_best_config.drop(columns = ['num_layers', 'dimensions', 'learning_rate', 'auc_val_mean', 'auc_test_mean', 'auc_test_std', 'auc_val_std'])
        df_best_config = df_best_config.reset_index(drop = True)
        df_main = df_main.merge(df_best_config, how = 'left')
    return df_main


def auc_for_all_configs(dir):
    '''
    Returns a dataframe containing the average auc for each model on each dataset

    Process:
    - For each model/dataset, the test_auc is averaged across all trials of all configs

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

        df_epoch_max = df_dataset[df_dataset['auc_val'] == df_dataset.groupby(['model_name', 'num_layers', 'dimensions', 'learning_rate','trial'])['auc_val'].transform(max)].drop_duplicates(['model_name', 'num_layers', 'dimensions', 'learning_rate','trial'], keep = 'first')
        df_avg_auc_val_per_config = df_epoch_max.groupby(['model_name', 'num_layers', 'dimensions', 'learning_rate']).aggregate({'auc_val': ['mean', np.std], 'auc_test': ['mean', np.std]}).reset_index()
        df_avg_auc_val_per_config.columns = df_avg_auc_val_per_config.columns.map('_'.join).str.strip('_') 

        df_across_configs = df_avg_auc_val_per_config.groupby(['model_name']).aggregate({'auc_test_mean': ['mean', np.std]}).reset_index()
        df_across_configs.columns = df_across_configs.columns.map('_'.join).str.strip('_')
        df_across_configs = df_across_configs.reset_index(drop = True)

        df_across_configs['auc_test_mean_mean'] = df_across_configs['auc_test_mean_mean'].round(2).astype(str)
        df_across_configs['auc_test_mean_std'] = df_across_configs['auc_test_mean_std'].round(2).astype(str)
        df_across_configs[str(dataset)] = df_across_configs['auc_test_mean_mean']
        df_across_configs = df_across_configs.drop(columns = ['auc_test_mean_mean', 'auc_test_mean_std'])
        df_main = df_main.merge(df_across_configs, how = 'left')
    return df_main


def auc_as_heatmap(agg = 'best'):
    '''
    Plots the auc dataframe as a heatmap

    Parameters:
    agg (string): either 'best' or 'all' for best/all configuration(s) respectively

    Returns:
    N/A
    '''
    if agg == 'best':
        df_auc = auc_for_best_configs().set_index('model_name', drop = True).astype(float)

    elif agg == 'all':
        df_auc = auc_for_all_configs().set_index('model_name', drop = True).astype(float)

    else:
        return 'invalid agg entry'

    plt.figure(figsize=(11, 8))
    sns.heatmap(df_auc, annot=True, cmap="Greens", fmt=".2f", linewidths=0.5
                ,linecolor='black', square=True)

    plt.axhline(3, color='black', linewidth=2)
    plt.axvline(3, color='black', linewidth=2)
    plt.axhline(6, color='black', linewidth=2)
    plt.axvline(6, color='black', linewidth=2)

    plt.title('AUC for best configurations' if agg == 'best' else 'AUC for all configurations')
    plt.show()


def create_blank_df(hyper_parameter):
    '''
    Utility function for generating a blank multi-indexed dataframe split 
    by model against dataset, hyper-parameter value 

    Parameters:
    hyper_parameter (list): either layers, dimensions, learning_rates

    Returns:
    dataframe: desired table
    '''
    headers = []
    row_index = []

    for category in dataset_categories.keys():
        for dataset in dataset_categories[category]:
            for config in hyper_parameter:
                headers.append((category,dataset,config))


    for model_type in model_categories.keys():
        for model in model_categories[model_type]:
            row_index.append((model_type, model))

    df = pd.DataFrame(index = pd.MultiIndex.from_tuples(row_index),columns = pd.MultiIndex.from_tuples(headers))

    return df


def generate_config_indices(hyper_parameter):
    '''
    For a given hyper_parameter, generates a dict with all configuration indices
    (see variable definition for configs) split across all values of the
    hyper_parameter

    Parameters:
    hyper_parameter (list): either layers, dimensions, learning_rates

    Returns:
    dict: desired dict
    '''
    result_dict = {}
    config_idx = [layers,dimensions,learning_rates].index(hyper_parameter)
    for value in hyper_parameter:
        matching_indices = [idx for idx, config in configs.items() if config[config_idx] == value]
        result_dict[value] = matching_indices
    return result_dict


def stopping_epoch(dataset_idx, model_idx, config, trial):
    '''
    For a given dataset/model/configuration id/trial number,
    generates the test auc at the stopping epoch. Stopping epoch
    is defined as the first instance where val auc is maxed. 

    Parameters:
    dataset_idx (int): index of the desired dataset
    model_idx (int): index of the desired model
    config (int): index of the deisred configuration
    trial (int): index of the desired trial number

    Returns:
    float: the test_acc at the stopping epoch
    '''
    sub_tensor = tensor[dataset_idx, model_idx,config,trial]
    max_index = np.argmax(sub_tensor[:,2])
    test_auc_stopping_epoch = sub_tensor[max_index,0]
    return test_auc_stopping_epoch


def auc_across_configs(dataset, model):
    '''
    For a given dataset/model, returns the average test auc
    across trials configuration as a list

    Parameters:
    dataset (str): desired dataset
    model (str): desired model

    Returns:
    list: all test aucs ordered by config id
    '''
    dataset_idx = dataset_indices[dataset]
    model_idx = model_indices[model]

    config_aucs = []

    for config in [x for x in range(36)]:
        trial_aucs = []
        for trial in [x for x in range(10)]:
            auc = stopping_epoch(dataset_idx, model_idx, config, trial)
            trial_aucs.append(auc)
        config_auc = np.mean(trial_aucs)
        config_aucs.append(config_auc)

    return config_aucs


def best_configuration(dataset, model):
    '''
    For a given dataset/model, returns the best configuration

    Parameters:
    dataset (str): desired dataset
    model (str): desired model

    Returns:
    int: index of best performing configuration (see defintion of 
        configs for indexing)
    '''
    def val_at_stopping_epoch(dataset_idx, model_idx, config, trial):
        '''
        Returns the val auc for a given dataset/model/ 
        config/trial at the stopping epoch. Stopping epoch is
        defined as the first instance where val auc is maxed. 

        Parameters:
        dataset_idx (int): index of the desired dataset
        model_idx (int): index of the desired model
        config (int): index of the deisred configuration
        trial (int): index of the desired trial number

        Returns:
        float: the val_auc at the stopping epoch
        '''
        sub_tensor = tensor[dataset_idx, model_idx,config,trial]
        max_index = np.argmax(sub_tensor[:,2])
        val_auc_stopping_epoch = sub_tensor[max_index,2]
        return val_auc_stopping_epoch
    
    def val_auc_across_configs(dataset, model):
        '''
        Returns the val auc for each config of a given  
        dataset/model at the stopping epoch. Stopping epoch is
        defined as the first instance where val auc is maxed. 

        Parameters:
        dataset (str): desired dataset
        model (str): desired model

        Returns:
        list: all val aucs ordered by config id
        '''
        dataset_idx = dataset_indices[dataset]
        model_idx = model_indices[model]

        config_aucs = []

        for config in [x for x in range(36)]:
            trial_aucs = []
            for trial in [x for x in range(10)]:
                auc = val_at_stopping_epoch(dataset_idx, model_idx, config, trial)
                trial_aucs.append(auc)
            config_auc = np.mean(trial_aucs)
            config_aucs.append(config_auc)

        return config_aucs
    
    val_aucs = val_auc_across_configs(dataset, model)

    val_auc_arr = np.array(val_aucs)
    index_of_max = np.argmax(val_auc_arr)

    return index_of_max


def df_auc_split_by_parameter(hyper_parameter):
    '''
    Returns a table that, for a given hyper parameter, outputs the percent
    difference of a parameter value vs the average acorss all parameters
    for each model/dataset

    Process:
    - For a given model/dataset, calculates the average test_auc using
    auc_across_configs (see above definition) as all_aucs
    - For each parameter value of the specified hyper-parameter, averages the
    test aucs as parameter_auc
    - Calculates the percent difference between all_aucs and parameter_aucs
    - For Pubmed, sets values to 0 on GAT, HGAT if hyper parameter is layers
    given that only layers 2 was trained.

    Parameters:
    hyper_parameter (list): either layers, dimensions, learning_rates

    Returns:
    dataframe: desired multi-index table
    '''
    df = create_blank_df(hyper_parameter)
    config_indices = generate_config_indices(hyper_parameter)

    for dataset_category in dataset_categories.keys():
        for dataset in dataset_categories[dataset_category]:
            for model_type in model_categories.keys():
                for model in model_categories[model_type]:
                    all_aucs = auc_across_configs(dataset, model)
                    config_auc = np.mean(all_aucs)
                    for parameter in hyper_parameter:
                        parameter_aucs = [all_aucs[i] for i in config_indices[parameter]]
                        parameter_auc = np.mean(parameter_aucs)
                        relative_auc = np.round(((parameter_auc/config_auc)*100)-100,2)
                        df.at[(model_type, model), (dataset_category, dataset,parameter)] = relative_auc

    if hyper_parameter == layers:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df.at[('Attention Network', 'GAT'),('Publications', 'PubMed')] = 0
            df.at[('Attention Network', 'HGAT'),('Publications', 'PubMed')] = 0
    return df
    

def best_parameter(hyper_parameter):
    '''
    For each model/dataset, outputs the best performing parameter value
    for the specified hyper parameter

    Parameters:
    hyper_parameter (list): either layers, dimensions, learning_rates

    Returns:
    dataframe: desired table
    '''
    df_main = pd.DataFrame(index = ['GAT', 'HGAT', 'HyboNet', 'GC', 'HGC', 'HGNN', 'GCN', 'HGCN', 'LGCN'], columns = dataset_indices.keys())
    config_indices = generate_config_indices(hyper_parameter)

    for dataset in dataset_indices.keys():
        for model in model_indices.keys():
            all_aucs = auc_across_configs(dataset, model)
            best_parameter = -1
            best_parameter_auc = 0

            for parameter in hyper_parameter:
                parameter_aucs = [all_aucs[i] for i in config_indices[parameter]]
                parameter_auc = np.mean(parameter_aucs)

                if parameter_auc > best_parameter_auc:
                    best_parameter_auc = parameter_auc
                    best_parameter = parameter
            df_main.at[model, dataset] = best_parameter

    return df_main


def best_hyper_parameter_heatmap(hyper_parameter):
    '''
    Stitches a heatmap for results from best_parameter (see definition above)

    Parameters:
    hyper_parameter (list): either layers, dimensions, learning_rates

    Returns:
    N/A
    '''
    if hyper_parameter == layers:
        df_hyper_parameter = best_parameter(hyper_parameter)
        custom_cmap = {
            0: '#000000',  
            2: '#ffffbf',  
            3: '#a6d96a',  
            4: '#1a9850'   
        }
        for model in model_indices.keys():
            df_hyper_parameter.at[model, 'PubMed'] = 0
        df_hyper_parameter = df_hyper_parameter.apply(pd.to_numeric, errors='coerce', downcast='integer')
        map_title = 'Best Performing Layer'
        f = 'd'

    elif hyper_parameter == dimensions:
        df_hyper_parameter = best_parameter(hyper_parameter)
        custom_cmap = {
            32: '#A5DEF2',  
            64: '#5BAEB7',  
            128: '#1E80C1',  
            256: '#414C6B'   
        }
        df_hyper_parameter = df_hyper_parameter.apply(pd.to_numeric, errors='coerce', downcast='integer')
        map_title = 'Best Performing Dimension'
        f = 'd'

    elif hyper_parameter == learning_rates:
        df_hyper_parameter = best_parameter(hyper_parameter)
        custom_cmap = {
            0.0002: '#d3ffb3',  
            0.001: '#389f0a',  
            0.005: '#1F6200'   
        }
        df_hyper_parameter = df_hyper_parameter.astype(float)
        map_title = 'Best Performing Learning Rate'
        f = '.4f'

    else:
        return 'Invalid hyper-parameter'



    cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', list(custom_cmap.values()))

    plt.figure(figsize=(13, 8))
    sns.heatmap(df_hyper_parameter, annot=True, cmap=cmap, cbar=False, fmt= f , linewidths=0.5,
                linecolor='black', square=True)


    plt.axhline(3, color='black', linewidth=2)
    plt.axvline(3, color='black', linewidth=2)
    plt.axhline(6, color='black', linewidth=2)
    plt.axvline(6, color='black', linewidth=2)

    plt.title(map_title)
    plt.show()
    

def df_stopping_epoch():
    '''
    Returns the average stopping epoch across all trials for the best performing 
    configuration on each model/dataset

    Parameters:
    N/A

    Returns:
    dataframe: a table containing the values
    '''
    df_main = pd.DataFrame(index = ['GAT', 'HGAT', 'HyboNet', 'GC', 'HGC', 'HGNN', 'GCN', 'HGCN', 'LGCN'], columns = dataset_indices.keys())
    
    for dataset, dataset_idx in dataset_indices.items():
        for model, model_idx in model_indices.items():
            best_config = best_configuration(dataset,model)
            stopping_epoch_across_trials = []
            for trial in [x for x in range(10)]:
                sub_tensor = tensor[dataset_idx, model_idx,best_config,trial]
                epoch = np.argmax(sub_tensor[:,2])
                stopping_epoch_across_trials.append(epoch)
            stopping_epoch_for_config = np.mean(stopping_epoch_across_trials)
            df_main.at[model, dataset] = np.round(stopping_epoch_for_config,1)

    return df_main


def heatmap_stopping_epoch():
    '''
    Stitches a heatmap containing the values of df_stopping_epoch

    Inputs:
    N/A

    Returns:
    N/A
    '''
    df_stopping_epochs = df_stopping_epoch()
    df_stopping_epochs = df_stopping_epochs.astype(float)
    plt.figure(figsize=(11, 8))
    sns.heatmap(df_stopping_epochs, annot=True, cmap="Reds", fmt=".1f", linewidths=0.5
                ,linecolor='black', square=True)

    plt.axhline(3, color='black', linewidth=2)
    plt.axvline(3, color='black', linewidth=2)
    plt.axhline(6, color='black', linewidth=2)
    plt.axvline(6, color='black', linewidth=2)

    plt.title("Stopping Epochs Heatmap")
    plt.show()


def stability_of_best_config(dir):
    '''
    For the best configuration of each dataset/model, returns the std dev. of
    test aucs across trials

    Process:
    - For each dataset/model/configuration/trial, chooses the 'stopping epoch'
    where val auc is first maximized
    - Then averages the val aucs at stopping epochs across all trials of a 
    given dataset/model/configuration
    - Chooses the configuration that has the highest average
    - Once the best config is chosen, collects the test aucs at the
    stopping epoch for each trial
    - Outputs the standard deviation amongst the 10 trials

    Parameters:
    dir (str): file path containing the CSVs with the structure of 
    ...\\{dataset}\\{dataset}_{model}_report.csv

    Returns:
    dataframe: desired table containing the standard deviations
    '''
    df_main = pd.DataFrame({'model_name': ['GAT', 'HGAT', 'HyboNet', 'GC', 'HGC', 'HGNN', 'GCN', 'HGCN', 'LGCN']})
    for dataset in dataset_indices.keys():
        directory = f'{dir}{dataset}\\'
        df_dataset = pd.DataFrame()
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            df = pd.read_csv(f, index_col = 'Unnamed: 0')
            #if (dataset == 'PubMed') and (((re.split("_", filename))[1] == 'GAT') or ((re.split("_", filename))[1] == 'HGAT')):
            #    df = df[df['auc_train'] != 0]
            df_dataset = pd.concat([df,df_dataset], ignore_index = True)
    
        df_epoch_max = df_dataset[df_dataset['auc_val'] == df_dataset.groupby(['model_name', 'num_layers', 'dimensions', 'learning_rate','trial'])['auc_val'].transform(max)].drop_duplicates(['model_name', 'num_layers', 'dimensions', 'learning_rate','trial'], keep = 'first')
        df_avg_auc_val_per_config = df_epoch_max.groupby(['model_name', 'num_layers', 'dimensions', 'learning_rate']).aggregate({'auc_val': ['mean', np.std], 'auc_test': ['mean', np.std]}).reset_index()
        df_avg_auc_val_per_config.columns = df_avg_auc_val_per_config.columns.map('_'.join).str.strip('_')
        df_best_config = df_avg_auc_val_per_config[df_avg_auc_val_per_config['auc_val_mean'] == df_avg_auc_val_per_config.groupby(['model_name'])['auc_val_mean'].transform(max)]
        df_best_config = df_best_config[df_best_config['auc_val_std'] == df_best_config.groupby(['model_name'])['auc_val_std'].transform(max)]
        df_best_config = df_best_config[df_best_config.duplicated(subset = ['model_name']) == False]
        df_best_config['auc_test_mean'] = df_best_config['auc_test_mean'].round(2).astype(str)
        df_best_config['auc_test_std'] = df_best_config['auc_test_std'].round(2).astype(str)
        df_best_config[str(dataset)] = df_best_config['auc_test_std']

        df_best_config = df_best_config.drop(columns = ['num_layers', 'dimensions', 'learning_rate', 'auc_val_mean', 'auc_test_mean', 'auc_test_std', 'auc_val_std'])
        df_best_config = df_best_config.reset_index(drop = True)
        df_main = df_main.merge(df_best_config, how = 'left')
    return df_main


def stability_of_best_config_heatmap():
    '''
    Represents the values from stability_of_best_config (see above for definition) as
    a heatmap.

    Parameters:
    N/A

    Returns:
    N/A
    '''
    
    df_auc = stability_of_best_config().set_index('model_name', drop = True).astype(float)


    plt.figure(figsize=(11, 8))
    sns.heatmap(df_auc, annot=True, cmap="Purples", fmt=".2f", linewidths=0.5
                ,linecolor='black', square=True)

    plt.axhline(3, color='black', linewidth=2)
    plt.axvline(3, color='black', linewidth=2)
    plt.axhline(6, color='black', linewidth=2)
    plt.axvline(6, color='black', linewidth=2)

    plt.title("Stability of auc for best config")
    plt.show()


def df_best_config():
    '''
    For each dataset/model, outputs the best performing hyper parameters

    Inputs:
    N/A

    Outputs:
    dataframe: desired table with each cell containing a list
    '''
    df_main = pd.DataFrame(index = ['GAT', 'HGAT', 'HyboNet', 'GC', 'HGC', 'HGNN', 'GCN', 'HGCN', 'LGCN'], columns = dataset_indices.keys())
    
    for dataset, dataset_idx in dataset_indices.items():
        for model, model_idx in model_indices.items():
            best_config = best_configuration(dataset,model)
            df_main.at[model, dataset] = configs[best_config]

    return df_main