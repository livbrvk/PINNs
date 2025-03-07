import pandas as pd
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GCN
import torch_geometric.loader as loader
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler

def get_cmapss_data(file_nr, train_percentage, data_dir='data_synthetic', chosen_sensor = None, scaling = 'standard'):
    """
    Function to get out training and testing set from the CMAPSS data.
    file_nr: 0-3, choosing which dataframe to use
    train_percentage: putting in a percentage for the train set, the rest will be allocated to testing (TODO: allocate for testing)
    data_dir: to specify the folder structure the data lies in, useful for RayTune or other software
    chosen_sensor: important for when looking at univariate data
    """
    files = os.listdir(data_dir)

    trains = [f for f in files if f.startswith("train")]

    df_train = pd.read_csv(os.path.join(data_dir,trains[file_nr]), sep = ' ', header = None)
    
    #istedenfor denne kan vi heller velge ut et tilfeldig antall mellom 0 og 100
    sensor_idx = (np.arange(df_train[0].min(), df_train[0].max()+1))
    np.random.seed(0)
    np.random.shuffle(sensor_idx)
    
    train_amt = int((train_percentage/100) * len(sensor_idx))
    train_idx = sensor_idx[:train_amt]
    test_idx = sensor_idx[train_amt:]
    
    if chosen_sensor!= None:
        temp_train = df_train[df_train[0].isin(train_idx)][[0,1,chosen_sensor]].copy()
        temp_test = df_train[df_train[0].isin(test_idx)][[0,1,chosen_sensor]].copy()
        
    else:
        temp_train = df_train[df_train[0].isin(train_idx)].copy()
        temp_test = df_train[df_train[0].isin(test_idx)].copy()
        
        
    cols = (temp_train.columns[(temp_train.columns >= 5)])
    
    if scaling == 'standard':
        
        train_mean = temp_train[cols].mean()
        train_std = temp_train[cols].std()

        temp_train.loc[:,cols] = ((temp_train[cols] - train_mean)/train_std)
        temp_test.loc[:,cols] = ((temp_test[cols] - train_mean)/train_std)
        
        
    elif isinstance(scaling, tuple):
        scaler = MinMaxScaler(feature_range = scaling)
        scaler.fit(temp_train.iloc[:,5:])
        temp_train.iloc[:,5:] = scaler.transform(temp_train.iloc[:,5:])
        temp_test.iloc[:,5:] = scaler.transform(temp_test.iloc[:,5:])
        
    
    temp_train = temp_train.drop(temp_train.columns[temp_train.isnull().all()].tolist(), axis = 1)
    temp_test = temp_test.drop(temp_test.columns[temp_test.isnull().all()].tolist(), axis = 1)
    
    
    return temp_train, temp_test



def get_edges(idx_list):
    num_nodes = np.sum(idx_list)
    
    edge_index = torch.tensor([list(range(num_nodes - 1)), list(range(1, num_nodes))], dtype=torch.long)
    
    return edge_index



def get_sliding_windows(df, chosen_sensor, win_size, outp_size, only_one = False):
    sliding_windows = df.groupby(0).apply(lambda x: np.lib.stride_tricks.sliding_window_view(x[chosen_sensor], window_shape = (win_size + outp_size)))
    
    x, y, idx, edges = [], [], [], []
    
    if only_one == True:
        for i in sliding_windows:
            x.append(i[:,:-outp_size])
            y.append(i[:,-1])
            edges.append(get_edges(len(i)))
            idx.append(len(i))
        
    else:
        for i in sliding_windows:
            x.append(i[:,:-outp_size])
            y.append(i[:,-outp_size:])
            edges.append(get_edges(len(i)))
            idx.append(len(i))
        
    return x, y, edges, idx
    

def get_loaded_data(df, win_size, outp_size, batch_size, chosen_sensor = 11, gcn = False, shuffle = True, only_one = False):
    
    sensor_nr, t_y, edges, idx = get_sliding_windows(df, 0, win_size, outp_size, only_one = only_one)
    
    sensor_nr = ([np.unique(sensor_nr[i], axis = 1) for i in range(len(sensor_nr))])

    t_x, t_y, edges, idx = get_sliding_windows(df, 1, win_size, outp_size, only_one = only_one)
    
    x, y, edges, idx = get_sliding_windows(df, chosen_sensor, win_size, outp_size, only_one = only_one)
    
    
    #Making the dataloader for the gcn case
    if gcn:
        dataset = ([Data(x = x[i], edge_index = get_edges(len(x[i])), y = y[i], t_x = t_x[i], t_y = t_y[i], sensor_nr = sensor_nr[i]) for i in range(len(x))])
        dataloader = loader.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)

    
    #Concatinating the rows for all other cases
    else:
        x = torch.from_numpy(np.concatenate(x, axis = 0)).float()
        y = torch.from_numpy(np.concatenate(y, axis = 0)).float()  
        
        t_x = torch.from_numpy(np.concatenate(t_x, axis = 0)).float()
        t_y = torch.from_numpy(np.concatenate(t_y, axis = 0)).float()
        
        sensor_nr = torch.from_numpy(np.concatenate(sensor_nr, axis = 0)).float()
        
        if x.ndim == 1:
            x = x.reshape(-1,1)
        if y.ndim == 1:
            y = y.reshape(-1,1)
        
        dataset = TensorDataset(x, y, t_x, t_y, sensor_nr)
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
        
    return dataloader, np.cumsum(idx)

