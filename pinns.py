import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import ml_models
import data_pipeline2 as dp
import training_models as tm

import os
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GCN




def finite_difference_derivative(x, t):
    dxdt = torch.zeros_like(x)
    dxdt[:,1:-1] = (x[:,2:] - x[:,:-2]) / (t[:,2:] - t[:,:-2])  # Central difference
    dxdt[:,0] = (x[:,1] - x[:,0]) / (t[:,1] - t[:,0])  # Forward difference at the start
    dxdt[:,-1] = (x[:,-1] - x[:,-2]) / (t[:,-1] - t[:,-2])  # Backward difference at the end
    return dxdt

class decay_pinn(nn.Module):
    def __init__(self, base_model):
        super(decay_pinn, self).__init__()
        
        self.base_model = base_model
        self.decay = nn.Parameter(torch.tensor(0.0, requires_grad = True))
        
    def forward(self, x):
        x = self.base_model(x)
        
        return x

def train_pinn_1(config, it_amt, model, data_dir = None, patience = 10, only_one = False):
    
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
            
            #ty = ty.clone().detach().requires_grad_(True)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)            
            loss = criterion(outputs, labels)
            
            # Compute physics loss (parameters mu and k in the ODE)
            decay = model.decay
            
            #x_phys = model(t_phys)
            

            xdot = finite_difference_derivative(outputs, ty)
            #print(xdot)
            
            ode_residual = xdot + decay
            #print(torch.mean(ode_residual**2))
            
            loss += config['lambda_'] * torch.mean(ode_residual**2)
            
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



def train_pinn_2(config, it_amt, model, data_dir = None, patience = 10, only_one = False):
    
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
            
            #ty = ty.clone().detach().requires_grad_(True)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)            
            loss = criterion(outputs, labels)
            
            # Compute physics loss (parameters mu and k in the ODE)
            decay = model.decay
            
            #x_phys = model(t_phys)
            

            xdot = finite_difference_derivative(outputs, ty)
            #print(xdot)
            
            ode_residual = xdot + decay*outputs
            #print(torch.mean(ode_residual**2))
            
            loss += config['lambda_'] * torch.mean(ode_residual**2)
            
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



def train_pinn_3(config, it_amt, model, data_dir = None, patience = 10, only_one = False):
    
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
            
            #ty = ty.clone().detach().requires_grad_(True)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)            
            loss = criterion(outputs, labels)
            
            # Compute physics loss (parameters mu and k in the ODE)
            decay = model.decay
            
            #x_phys = model(t_phys)
            

            xdot = finite_difference_derivative(outputs, ty)
            #print(xdot)
            
            ode_residual = xdot + decay*ty**(decay - 1)
            #print(torch.mean(ode_residual**2))
            
            loss += config['lambda_'] * torch.mean(ode_residual**2)
            
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