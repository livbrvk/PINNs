import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np


def recursive_pred(model, dataloader, output_size, mod_type = None):
    with torch.no_grad():
        pred_container = np.zeros((len(dataloader.dataset), output_size))  # Store all predictions
        start_idx = 0  # Track global position in dataset
        if mod_type == None:
            for batch in dataloader:
                x, y, tx, ty, sensor = batch
                batch_size = x.shape[0]
                

                temp_x = x.clone()
                for j in range(output_size):
                    pred = model(temp_x)  # Get prediction

                    # Shift temp_x and insert new prediction
                    temp_x[:, :-1] = temp_x[:, 1:]
                    temp_x[:, -1] = pred.reshape(-1)

                    # Store predictions in correct indices
                    pred_container[start_idx:start_idx + batch_size, j] = pred.reshape(-1)

                start_idx += batch_size  # Move index forward
                
        elif mod_type == 'transformer':
            for batch in dataloader:
                x, y, tx, ty, sensor = batch
                batch_size = x.shape[0]

                temp_x = x.clone()
                for j in range(output_size):
                    pred = model(temp_x,torch.zeros_like(y))  # Get prediction
               

                    # Shift temp_x and insert new prediction
                    temp_x[:, :-1] = temp_x[:, 1:]
                    temp_x[:, -1] = pred.reshape(-1)

                    # Store predictions in correct indices
                    pred_container[start_idx:start_idx + batch_size, j] = pred.reshape(-1)

                start_idx += batch_size  # Move index forward     
                
        else:
            amt = sum(data.x.shape[0] for data in dataloader.dataset)
            pred_container = np.zeros((amt,output_size))
            for batch in dataloader:
                x, y = batch.x, batch.y
                batch_size = np.concatenate(x).shape[0]

                temp_x = x.copy()

                #x_cont = np.concatenate(x).copy()
                for j in range(output_size):
                    pred = model(temp_x, batch.edge_index)  # Get prediction
                    #pred = np.concatenate(pred)
                    # Shift temp_x and insert new prediction
                    
                    idx = 0
                    
                    for arr in range(len(temp_x)):
                        temp_x[arr] = temp_x[arr].copy()
                        
                        temp_x[arr][:,:-1] = temp_x[arr][:,1:]
                        temp_x[arr][:, -1] = pred[idx:(idx+len(temp_x[arr]))].reshape(-1)
                      
                        
                        idx += len(temp_x[arr])
                        
                    
                    #temp_x[:, :-1] = temp_x[:, 1:]
                    #temp_x[:, -1] = pred.reshape(-1)

                    # Store predictions in correct indices
                    pred_container[start_idx:start_idx + batch_size, j] = pred.reshape(-1)

                start_idx += batch_size  # Move index forward

    return pred_container