import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np


hidden_size = 48
num_feature = 64

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
                
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        
        self.Linear1 = nn.Linear(self.input_size,self.input_size*2)
        self.relu = nn.ReLU()
        self.batch = nn.BatchNorm1d(1000)
        
        self.rnn = nn.GRU(self.input_size*2, self.hidden_size, batch_first=True,num_layers=4,dropout=0.4)
        
        self.Linear2 = nn.Linear(self.hidden_size, 32)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(32,3)
        
    # create function to init state
    def init_hidden(self, batch_size):
        return torch.zeros(4, batch_size, self.hidden_size)
        
    
    def forward(self, x):  
        x=self.Linear1(x)
        x=self.relu(x)
        x=self.batch(x)
        
        batch_size = x.size(0)
        
        
        h = self.init_hidden(batch_size).to(device)
        
        out, h = self.rnn(x, h)    
        
        out = self.Linear2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        
        return out
        
    
model = Model(input_size=num_feature, hidden_size=hidden_size, output_size=3)