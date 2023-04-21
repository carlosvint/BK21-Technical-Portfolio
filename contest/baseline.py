import random
import pandas as pd
import numpy as np
import os
import glob
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm
import zipfile

import warnings
import argparse

import torchcde
warnings.filterwarnings(action='ignore')

from ctc.ctc_encoder import Encoder

parser = argparse.ArgumentParser()
parser.add_argument('-model', help="lstm, cde", required=True)
opt = parser.parse_args()


class CustomDataset(Dataset):
    def __init__(self, input_paths, target_paths, infer_mode):
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.infer_mode = infer_mode
        
        self.data_list = []
        self.label_list = []
        print('Data Pre-processing..')
        for input_path, target_path in tqdm(zip(self.input_paths, self.target_paths)):
            input_df = pd.read_csv(input_path)
            target_df = pd.read_csv(target_path)
            
            input_df = input_df.drop(columns=['시간'])
            input_df = input_df.fillna(0)
            
            input_length = int(len(input_df)/1440)
            target_length = int(len(target_df))
            
            for idx in range(target_length):
                time_series = input_df[1440*idx:1440*(idx+1)].values
                self.data_list.append(torch.Tensor(time_series))
            for label in target_df["rate"]:
                self.label_list.append(label)
        print('Done.')
              
    def __getitem__(self, index):
        data = self.data_list[index]
        label = self.label_list[index]
        if self.infer_mode == False:
            return data, label
        else:
            return data
        
    def __len__(self):
        return len(self.data_list)


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.lstm = nn.LSTM(input_size=37, hidden_size=256, batch_first=True, bidirectional=False)
        self.classifier = nn.Sequential(
            nn.Linear(256, 1),
        )
        
    def forward(self, x, device):
        hidden, _ = self.lstm(x)
        output = self.classifier(hidden[:,-1,:])
        return output

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.padding = 1
        self.kernel_size = 3
        self.stride = 2
        self.dilation = 1

        self.src_embed = nn.Sequential(nn.Conv1d(in_channels=37,
                                                 out_channels=64,
                                                 kernel_size=self.kernel_size,
                                                 stride=self.stride,
                                                 padding=self.padding,
                                                 dilation=self.dilation,
                                                 bias=False),
                                       nn.BatchNorm1d(num_features=64),
                                       nn.ReLU(inplace=True),

                                       nn.Conv1d(in_channels=64,
                                                 out_channels=128,
                                                 kernel_size=self.kernel_size,
                                                 stride=self.stride,
                                                 padding=self.padding,
                                                 dilation=self.dilation,
                                                 bias=False),
                                       nn.BatchNorm1d(num_features=128),
                                       nn.ReLU(inplace=True),
                                       )

        self.lstm = nn.LSTM(input_size=128, hidden_size=256, batch_first=True, bidirectional=False)
        self.classifier = nn.Sequential(
            nn.Linear(256, 1),
        )
        
    def forward(self, x, device):
        x = x.transpose(-1,-2)
        embed_out = self.src_embed(x)
        embed_out = embed_out.transpose(-1, -2) 
        hidden, _ = self.lstm(embed_out)
        output = self.classifier(hidden[:,-1,:])
        return output

class TransformerModel(nn.Module):
    def __init__(self, d_model=256, d_ff=1024, n_head=8, n_layers=6, label_vocab_size=1, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.encoder = Encoder(d_model=d_model,
                                d_ff=d_ff,
                                n_head=n_head,
                                num_encoder_layers=n_layers,
                                dropout=dropout)
        self.final_proj = nn.Linear(d_model, label_vocab_size)

    def forward(self, signal, device):
        """
        :param signal: a tensor shape of [batch, length, 1]
        :param signal_lengths:  a tensor shape of [batch,]
        :return:
        """
        signal_lengths = signal.squeeze(2).sum(1)
        enc_output, enc_output_lengths = self.encoder(
            signal, signal_lengths[:,0])  # (N,L,C), [32, 256, 256]
        out = self.final_proj(enc_output)  # (N,L,C), [32, 256, 6]
        return out, enc_output_lengths

class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()
        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z

class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels=38, hidden_channels=32, output_channels=1, interpolation="cubic"):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation = interpolation

    def forward(self, x, device):
        #print(x.shape)
        t = torch.linspace(0., x.shape[1], x.shape[1]).to(device)

        X = torch.cat([x, t.unsqueeze(1).repeat(x.shape[0], 1, 1)], dim=2)
        #print(X.shape)
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X)
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)

        ######################
        # Actually solve the CDE.
        ######################
        #print(z0.shape)
        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              t=X.interval, atol=1e-2, rtol=1e-1)

        ######################
        # Both the initial value and the terminal value are returned from cdeint; extract just the terminal value,
        # and then apply a linear map.
        ######################
        z_T = z_T[:, 1]
        pred_y = self.readout(z_T)
        return pred_y


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train(model, optimizer, train_loader, val_loader, scheduler, device, CFG):
    model.to(device)
    criterion = nn.L1Loss().to(device)
    
    best_loss = 9999
    best_model = None
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        for X, Y in tqdm(iter(train_loader)):
            X = X.to(device)
            Y = Y.to(device)
            
            optimizer.zero_grad()
            
            output = model(X, device)
            loss = criterion(output, Y)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        val_loss = validation(model, val_loader, criterion, device)
        
        print(f'Train Loss : [{np.mean(train_loss):.5f}] Valid Loss : [{val_loss:.5f}]')
        
        if scheduler is not None:
            scheduler.step()
            
        if best_loss > val_loss:
            best_loss = val_loss
            best_model = model
    return best_model

def validation(model, val_loader, criterion, device):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for X, Y in tqdm(iter(val_loader)):
            X = X.float().to(device)
            Y = Y.float().to(device)
            
            model_pred = model(X, device)
            loss = criterion(model_pred, Y)
            
            val_loss.append(loss.item())
            
    return np.mean(val_loss)

def inference_per_case(model, test_loader, test_path, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for X in iter(test_loader):
            X = X.float().to(device)
            
            model_pred = model(X, device)
            
            model_pred = model_pred.cpu().numpy().reshape(-1).tolist()
            
            pred_list += model_pred
    
    submit_df = pd.read_csv(test_path)
    submit_df['rate'] = pred_list
    submit_df.to_csv(test_path, index=False)

def main():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    CFG = {
        'EPOCHS':10,
        'LEARNING_RATE':1e-3,
        'BATCH_SIZE':16,
        'SEED':41
    }

    seed_everything(CFG['SEED']) # Seed 고정

    all_input_list = sorted(glob.glob('/nas/datahub/dacon_chunggyungche/train_input/*.csv'))
    all_target_list = sorted(glob.glob('/nas/datahub/dacon_chunggyungche/train_target/*.csv'))

    train_input_list = all_input_list[:50]
    train_target_list = all_target_list[:50]

    val_input_list = all_input_list[50:]
    val_target_list = all_target_list[50:]

    train_dataset = CustomDataset(train_input_list, train_target_list, False)
    train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=6)

    val_dataset = CustomDataset(val_input_list, val_target_list, False)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=6)

    if opt.model == 'lstm':
        model = BaseModel()
    if opt.model == 'cde':
        model = NeuralCDE()
    if opt.model == 'cnn':
        model = CNNModel()
    if opt.model == 'transformer':
        model = TransformerModel()
    model.eval()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
    scheduler = None

    best_model = train(model, optimizer, train_loader, val_loader, scheduler, device, CFG)

    test_input_list = sorted(glob.glob('/nas/datahub/dacon_chunggyungche/test_input/*.csv'))
    test_target_list = sorted(glob.glob('/nas/home/carlos/time_series/contest/test_target/*.csv'))

    for test_input_path, test_target_path in zip(test_input_list, test_target_list):
        test_dataset = CustomDataset([test_input_path], [test_target_path], True)
        test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
        inference_per_case(best_model, test_loader, test_target_path, device)\


    os.chdir("./test_target/")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    submission = zipfile.ZipFile("../submission" + timestr + ".zip", 'w')
    for path in test_target_list:
        path = path.split('/')[-1]
        submission.write(path)
    submission.close()

if __name__ == "__main__":
    main()
