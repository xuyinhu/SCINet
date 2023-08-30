import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

class PVDataset(Dataset):
    def __init__(self,df,flag,input_len=288,output_len=144,months=3,scale=True,inverse=False,timeenc=1,freq='h'):
        assert flag in ['train','valid','test'], 'only train/test dataset is supported.'
        type_map = {'train':0,'valid':1,'test':2}
        self.type_id = type_map[flag]
        self.input_len = input_len
        self.output_len = output_len
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.inverse = inverse
        self.months=months

        self._read_data(df)
        # import pdb;pdb.set_trace()

    def _read_data(self,df):   
        self.scaler = StandardScaler()
        df_raw = df.copy()
        df = df_raw.iloc[:,1:]
        border1s = [0,self.months*30*288-7*288-self.input_len,self.months*30*288-self.input_len]
        border2s = [self.months*30*288-7*288,self.months*30*288,self.months*30*288+7*288]
        border1,border2 = border1s[self.type_id],border2s[self.type_id]
        if self.scale:
            train_data = df[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df.values)
        else:
            data = df.values

        # df_stamp = df_raw[border1:border2]['ts'].copy()
        # df_stamp['ts'] = df_stamp['ts'].apply(pd.to_datetime)
        df_stamp = pd.DataFrame({'ts':df_raw[border1:border2]['ts'].apply(pd.to_datetime)})
        data_stamp = time_features(df_stamp,timeenc=self.timeenc,freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # merge xuyh
        self.data_x = np.concatenate([self.data_x,self.data_stamp],axis=1)
        self.data_y = np.concatenate([self.data_y,self.data_stamp],axis=1)
        
    
    def __getitem__(self,index):
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.output_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return seq_x,seq_y

    def __len__(self):
        return len(self.data_x) - self.input_len - self.output_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
