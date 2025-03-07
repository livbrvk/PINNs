import ml_models
import data_pipeline2 as dp
import torch

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
#import gpytorch
import torch.nn as nn
import torch
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GCN
import numpy as np


def train_MLP_model(config, it_amt, data_dir = None, patience = 10):

    model = ml_models.MLP(input_size=config['input_size'], hidden_size=config['hidden_size'], num_layers=config['layer_amt'], output_size = config['output_size'])
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr = config['lr'])
    criterion = nn.MSELoss()

    start_epoch = 0
        
    file_nr = 0
    chosen_sensor = 11
    train_percentage = 80
    data_dir = 'C:/Users/lefti/OneDrive - KTH/phd_kth/1sem/lit_proj/data_tests/CMAPSSData/data'
    df_train, df_test = dp.get_cmapss_data(file_nr, chosen_sensor, train_percentage, data_dir)    

    trainloader, idx = dp.get_loaded_data(df_train, config['input_size'], config['output_size'], config['batch_size'])
    testloader, idx = dp.get_loaded_data(df_test, config['input_size'], config['output_size'], config['batch_size'])

    best_val_loss = 100
    for epoch in range(start_epoch, it_amt):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0

        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(testloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1
                
        print('RMSE, train ' + str((running_loss/epoch_steps)) + ', test ' + str((val_loss/val_steps)))
        
        val_loss = val_loss/val_steps
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss.copy()
            no_improve_epochs = 0  # Reset counter
        else:
            no_improve_epochs += 1
        
        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs).")
            break
       
    print("Finished Training")
    
    return model



def train_model(config, it_amt, model_fnc, data_dir = None, patience = 10, only_one = False):
    if only_one == True:
        model = model_fnc(input_size=config['input_size'], hidden_size=config['hidden_size'], num_layers=config['layer_amt'], output_size = 1)
    else:
        model = model_fnc(input_size=config['input_size'], hidden_size=config['hidden_size'], num_layers=config['layer_amt'], output_size = config['output_size']) 

    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr = config['lr'])
    criterion = nn.MSELoss()

    start_epoch = 0
        
    file_nr = 0
    chosen_sensor = 11
    train_percentage = 80
    
    df_train, df_test = dp.get_cmapss_data(file_nr, train_percentage, data_dir, chosen_sensor)    

    trainloader, idx = dp.get_loaded_data(df_train, config['input_size'], config['output_size'], config['batch_size'], only_one = only_one)
    testloader, idx = dp.get_loaded_data(df_test, config['input_size'], config['output_size'], config['batch_size'], only_one = only_one)

    best_val_loss = 100
    for epoch in range(start_epoch, it_amt):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0

        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, tx, ty, sensor = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(testloader, 0):
            with torch.no_grad():
                inputs, labels, tx, ty, sensor = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1
                
        print('RMSE, train ' + str((running_loss/epoch_steps)) + ', test ' + str((val_loss/val_steps)))
        
        val_loss = val_loss/val_steps
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss.copy()
            no_improve_epochs = 0  # Reset counter
        else:
            no_improve_epochs += 1
        
        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs).")
            break
            
       
    print("Finished Training")
    
    return model



def train_gcn(config, it_amt, data_dir = None, patience = 10, only_one = False):
    if only_one == True:
        model = ml_models.my_GCN(input_size=config['input_size'], hidden_size=config['hidden_size'], num_layers=config['layer_amt'], output_size = 1)
    else:
        model = ml_models.my_GCN(input_size=config['input_size'], hidden_size=config['hidden_size'], num_layers=config['layer_amt'], output_size = config['output_size'])
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr = config['lr'])
    criterion = nn.MSELoss()

    start_epoch = 0
        
    file_nr = 0
    chosen_sensor = 11
    train_percentage = 80
    df_train, df_test = dp.get_cmapss_data(file_nr, train_percentage, data_dir, chosen_sensor)

    trainloader, idx = dp.get_loaded_data(df_train, config['input_size'], config['output_size'], config['batch_size'], gcn = True, only_one = only_one)
    testloader, idx = dp.get_loaded_data(df_test, config['input_size'], config['output_size'], config['batch_size'], gcn = True, only_one = only_one)

    best_val_loss = 100
    for epoch in range(start_epoch, it_amt):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0

        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data.x, data.y
            #inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs, data.edge_index)
            labels = np.concatenate(labels)
            if labels.ndim==1:
                labels = labels.reshape(-1,1)

            loss = criterion(outputs.float(), torch.from_numpy(labels).float())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(testloader, 0):
            with torch.no_grad():
                inputs, labels = data.x, data.y
                #inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs, data.edge_index)
                
                labels = np.concatenate(labels)
                if labels.ndim==1:
                    labels = labels.reshape(-1,1)
                
                loss = criterion(outputs.float(), torch.from_numpy(labels).float())
                val_loss += loss.cpu().numpy()
                val_steps += 1
                
        print('RMSE, train ' + str((running_loss/epoch_steps)) + ', test ' + str((val_loss/val_steps)))
        
        val_loss = val_loss/val_steps
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss.copy()
            no_improve_epochs = 0  # Reset counter
        else:
            no_improve_epochs += 1
        
        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs).")
            break
       
    print("Finished Training")
    
    return model



def train_transformer(config, it_amt, data_dir = None, patience = 10, only_one = False):
    if only_one == True:
        model = ml_models.Transformer(input_size=config['input_size'], hidden_size=config['hidden_size'], num_layers=config['layer_amt'], output_size = 1)
    else:
        model = ml_models.Transformer(input_size=config['input_size'], hidden_size=config['hidden_size'], num_layers=config['layer_amt'], output_size = config['output_size'])
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr = config['lr'])
    criterion = nn.MSELoss()

    start_epoch = 0
        
    file_nr = 0
    chosen_sensor = 11
    train_percentage = 80

    df_train, df_test = dp.get_cmapss_data(file_nr, train_percentage, data_dir, chosen_sensor)    

    trainloader, idx = dp.get_loaded_data(df_train, config['input_size'], config['output_size'], config['batch_size'], only_one = True)
    testloader, idx = dp.get_loaded_data(df_test, config['input_size'], config['output_size'], config['batch_size'], only_one = True)
    
    best_val_loss = 100
    for epoch in range(start_epoch, it_amt):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0

        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, tx, ty, sensor = data
            #inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs, labels)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(testloader, 0):
            with torch.no_grad():
                inputs, labels, tx, ty, sensor = data
                #inputs, labels = inputs.to(device), labels.to(device)
                fake_labs = torch.zeros(labels.shape)

                outputs = model(inputs, fake_labs)
                
                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1
                
        print('RMSE, train ' + str((running_loss/epoch_steps)) + ', test ' + str((val_loss/val_steps)))
        
        val_loss = val_loss/val_steps
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss.copy()
            no_improve_epochs = 0  # Reset counter
        else:
            no_improve_epochs += 1
        
        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs).")
            break
       
    print("Finished Training")
    
    return model