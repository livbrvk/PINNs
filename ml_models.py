import torch.nn as nn
import torch
from torch_geometric.nn.models import GCN as torch_GCN
import numpy as np


class Baseline(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_first = True):
        super(Baseline, self).__init__()
        
        self.fc = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        
        x = self.fc(x)
        
        return x



class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_first = True):
        super(MLP, self).__init__()
        
        layers = []
        
        #First layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        #Hidden layer
        for i in range(1, num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        self.conv_layers = nn.Sequential(*layers)
        
        #Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.conv_layers(x)
        
        x = self.fc(x)
        
        return x
    

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_first = True):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers, batch_first = batch_first)
        
        self.linear = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x, _ = self.lstm(x)
        
        x = self.linear(x)
        return x
    
    
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_first = True):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers, batch_first = batch_first)
        
        self.linear = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x, _ = self.gru(x)
        
        x = self.linear(x)
        return x
    
    
class GRU_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, gru_layers, lstm_layers, output_size, batch_first = True):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size,
            hidden_size=hidden_size,
            num_layers=gru_layers, batch_first = batch_first)
        
        self.lstm = nn.LSTM(input_size = hidden_size, hidden_size = hidden_size, num_layers = lstm_layers, batch_first = batch_first)
        
        self.linear = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x, _ = self.gru(x)
        x, _ = self.lstm(x)
        
        x = self.linear(x)
        return x
    
    
    
class my_GCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,  output_size, batch_first = True):
        super().__init__()
        self.gcn = torch_GCN(in_channels = input_size, hidden_channels = hidden_size, num_layers = num_layers, out_channels = output_size)
        
        #self.linear = nn.Linear(hidden_size, output_size) GNN has a linear output already
    def forward(self, x, edge_index):
        x = torch.from_numpy(np.concatenate(x)).float() #Making it compatible with mini-batching
        x = self.gcn(x, edge_index)
                
        #x = self.linear(x)
        return x
    
    
class GCN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, gcn_layers, lstm_layers,  output_size, batch_first = True):
        super().__init__()
        self.gcn = GCN(in_channels = input_size, hidden_channels = hidden_size, num_layers = gcn_layers, out_channels = None)
        
        self.lstm = nn.LSTM(input_size = hidden_size, hidden_size = hidden_size, num_layers = lstm_layers, batch_first = batch_first)
        
        self.linear = nn.Linear(hidden_size, output_size) 
    def forward(self, x, edge_index):
        x = torch.from_numpy(np.concatenate(x)).float() #Making it compatible with mini-batching
        x = self.gcn(x, edge_index)
        
        x,_ = self.lstm(x)
        
        x = self.linear(x)
                
        return x
    
    
class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_first = True, kernel_size=3, stride=1, padding=1):
        super(CNN, self).__init__()
        
        # List to hold convolutional layers
        layers = []
        
        # First convolutional layer
        layers.append(nn.Conv1d(in_channels=1, out_channels=hidden_size, kernel_size=kernel_size, stride=stride, padding=padding))
        layers.append(nn.ReLU())
        
        # Additional convolutional layers (keeping the number of filters constant)
        for i in range(1, num_layers):
            layers.append(nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, stride=stride, padding=padding))
            layers.append(nn.ReLU())
        
        # Combine all layers into a Sequential container
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate the size of the output after convolutions
        conv_output_size = input_size
        for _ in range(num_layers):
            conv_output_size = (conv_output_size - kernel_size + 2 * padding) // stride + 1
        
        # Define a fully connected layer
        self.fc = nn.Linear(hidden_size * conv_output_size, output_size)
    
    def forward(self, x):
        # Apply convolutional layers
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, num_features)
        # Fully connected layer for prediction
        x = self.fc(x)
        
        return x
    
    
class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cnn_layers, lstm_layers, output_size, batch_first = True, kernel_size=3, stride=1, padding=1):
        super(CNN_LSTM, self).__init__()
        
        # List to hold convolutional layers
        layers = []
        
        # First convolutional layer
        layers.append(nn.Conv1d(in_channels=1, out_channels=hidden_size, kernel_size=kernel_size, stride=stride, padding=padding))
        layers.append(nn.ReLU())
        
        # Additional convolutional layers (keeping the number of filters constant)
        for i in range(1, cnn_layers):
            layers.append(nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, stride=stride, padding=padding))
            layers.append(nn.ReLU())
        
        # Combine all layers into a Sequential container
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate the size of the output after convolutions
        conv_output_size = input_size
        for _ in range(cnn_layers):
            conv_output_size = (conv_output_size - kernel_size + 2 * padding) // stride + 1
            
        #Adding LSTM layers
        self.lstm = nn.LSTM(input_size = hidden_size*conv_output_size, hidden_size = hidden_size, num_layers = lstm_layers)
        
        # Define a fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Apply convolutional layers
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, num_features)
        
        x,_ = self.lstm(x)
        
        # Fully connected layer for prediction
        x = self.fc(x)
        
        return x
    

    
# Positional Encoding function
def getPositionEncoding(seq_len, d, n=10000):
    P = torch.zeros((seq_len, d))
    for k in torch.arange(seq_len):  # Use torch.arange instead of np.arange
        for i in torch.arange(int(d / 2)):  # Use torch.arange instead of np.arange
            denominator = torch.pow(n, 2 * i / d)
            P[k, 2 * i] = torch.sin(k / denominator)
            P[k, 2 * i + 1] = torch.cos(k / denominator)
    P = P.unsqueeze(0)  # Add batch dimension (1, seq_len, d)
    return P


class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers,  num_heads = 2,batch_first = True, dropout = 0.1):
        super(Transformer, self).__init__()
        
        #Changing of dimensions
        self.embed_input = nn.Linear(input_size, hidden_size)
        self.embed_target = nn.Linear(output_size, hidden_size)
                
        #Positional encoding
        #self.pos_enc = getPositionEncoding(hidden_size, output_size)
        
        self.transformer = nn.Transformer(d_model=hidden_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size * 2,
            batch_first=True, dropout = dropout)
        
        self.embed_output = nn.Linear(hidden_size, output_size)

        
    def forward(self, src, tgt = None):
        #Embedding and adding positional encoding
        #src = src.unsqueeze(-1)
        src = self.embed_input(src)
        
        src = src + getPositionEncoding(src.shape[0], src.shape[1])
        #Embedding and adding positional encoding
        tgt = self.embed_target(tgt)
        tgt = tgt + getPositionEncoding(tgt.shape[0], tgt.shape[1])

        output = self.transformer(src, tgt)
        return self.embed_output(output).squeeze(0)

    
    
    
class Transformer2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_heads = 4,batch_first = True):
        super(Transformer2, self).__init__()
        
        #Changing of dimensions
        self.embed_input = nn.Linear(input_size, hidden_size)
        self.embed_target = nn.Linear(output_size, hidden_size)
                
        #Positional encoding
        #self.pos_enc = getPositionEncoding(hidden_size, output_size)
        
        self.transformer = nn.Transformer(d_model=hidden_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size * 2,
            batch_first=True)
        
        self.embed_output = nn.Linear(hidden_size, output_size)

        
    def forward(self, src, tgt = None):
        #Embedding and adding positional encoding
        src = self.embed_input(src)
        src = src #+ getPositionEncoding(src.shape[0], src.shape[1])
        #Embedding and adding positional encoding
        tgt = self.embed_target(tgt)
        tgt = tgt #+ getPositionEncoding(tgt.shape[0], tgt.shape[1])

        output = self.transformer(src, tgt)
        return self.embed_output(output)