import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.SCINet import SCINet
from datetime import datetime,timedelta
from utils.tools import StandardScaler,adjust_learning_rate
from utils.data_loader import PVDataset
from utils.timefeatures import time_features
import time
from datetime import datetime

import os
os.environ['CUDA_VISIBLE_DEVICES']='2'

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)


class PVSCINet():
    def __init__(self,input_len=288,output_len=144,hid_size=1,lr=0.001,dropout=0.1):
        self.input_len = input_len
        self.output_len = output_len
        self.hid_size = hid_size
        self.dropout = dropout
        self.lr = lr
        self.timeenc = 1
        self.freq = 'h'
        self.model,self.mean,self.std = self._build_model()
        assert input_len==288, 'if not 288, timedelta=1 in below predict() should modify'
        
        
    def _build_model(self):
        model = SCINet(input_len=self.input_len,output_len=self.output_len,input_dim=7,hid_size=self.hid_size,
                num_stacks=1,num_levels=3,concat_len=0, groups=1, kernel=3, dropout=self.dropout,
                single_step_output_One=0, positionalE=False,modified=True,RIN=True)
        mean = np.zeros((1,3))
        std = np.ones((1,3))
        return model,mean,std

    def load(self,load_file):
        check = torch.load(load_file)
        self.model.load_state_dict(check['state_dict'])
        self.mean = check['mean']
        self.std = check['std']

    def train(self,df,epochs=30,batch_size=64):
        '''
        :param df: dataframe, |--ts--|--pv--|--radiation--|--temperature--|
        '''
        self.model = self.model.cuda()
        self.model.train()
        train_set = PVDataset(df=df,flag='train',months=3,input_len=self.input_len,output_len=self.output_len)
        scaler = train_set.scaler
        train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,drop_last=True)
        val_set = PVDataset(df=df,flag='valid')
        val_loader = DataLoader(val_set,batch_size=batch_size,shuffle=False)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.MSELoss()
        writer = SummaryWriter('./runs/train_scinet{}_{}_h{}_lr{}_dp{}'.format(self.input_len,
            self.output_len,self.hid_size,self.lr,self.dropout))
        for epoch in range(epochs):
            self.model.train()
            iter = 0
            train_loss = []
            # if epoch in [10,20]:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] *= 0.1
            epoch_time = time.time()
            for x,y in train_loader:
                optimizer.zero_grad()
                x = x.float().cuda()
                y = y.float().cuda()
                outputs = self.model(x)
                loss = criterion(outputs,y)
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                if (iter+1)%100==0:
                    print('epoch {0} iters {1}/{2} | loss: {3:.7f}'.format(epoch+1,iter+1,len(train_loader),loss.item()))
                iter += 1
            print('epoch {0} cost time {1}'.format(epoch+1,time.time()-epoch_time))
            train_l = np.average(train_loss)
            
            print('--------start to validate-----------')
            val_l = self.valid(val_loader, criterion)
            print("Epoch: {} | Train Loss: {:.7f} valid Loss: {:.7f}".format(
                epoch + 1, train_l, val_l))
            
            writer.add_scalar('valid_loss',val_l,global_step=epoch)
            writer.add_scalar('train_loss',train_l,global_step=epoch)
            adjust_learning_rate(optimizer, epoch+1, lr=self.lr)
        # save model
        save_check = {'state_dict':self.model.state_dict(),'mean':scaler.mean,'std':scaler.std}
        torch.save(save_check,'./check/scinet_{}_{}_h{}_lr{}_dp{}_ep{}.pkl'.format(self.input_len,
            self.output_len,self.hid_size,self.lr,self.dropout,epoch+1))


    def valid(self,val_loader,criterion):
        self.model.eval()
        valid_loss = []
        for x,y in val_loader:
            x = x.float().cuda()
            y = y.float().cuda()
            outputs = self.model(x)
            loss = criterion(outputs,y)
            valid_loss.append(loss.item())
        return np.average(valid_loss)
        

    def eval(self,df):
        
        pass
        
    def predict(self,df):
        '''
        input:df,dataframe
        ---
        |--ts--|--pv--|--radiation--|--temperature--|
        '''
        self.model.eval()
        self.model = self.model.to(torch.device('cpu'))
        scaler = StandardScaler(self.mean,self.std)
        data = df.copy()
        data_stamp = torch.from_numpy(time_features(pd.DataFrame({'ts':data['ts']}),timeenc=self.timeenc,freq=self.freq))
        data = np.array([data.iloc[:,1:].values])
        data = scaler.transform(torch.tensor(data,dtype=torch.float32))
        # print(data)
        # print(data_stamp)
        data = torch.concat([data[0],data_stamp],axis=1).type(torch.float32).unsqueeze(0)
        # import pdb;pdb.set_trace()
        output = scaler.inverse_transform(self.model(data)[:,:,:3])
        output = output[0,:,0].detach().cpu().numpy()
        time_index = pd.to_datetime(df['ts'])+timedelta(days=1)
        pred_df = pd.DataFrame({'ts':time_index[:len(output)],'pv_prediction':output}).reset_index(drop=True)
        return pred_df

if __name__ == '__main__':
    # '''
    model = PVSCINet(input_len=288,output_len=144,hid_size=4,lr=0.007,dropout=0.1)
    df_raw = pd.read_csv('/public/home/xuyh02/projects/pv_forecast/data/pv_data.csv')
    df = pd.DataFrame({'ts': df_raw['0'], 'pv': df_raw['6'],'radiation':df_raw['4'],'temperature':df_raw['5']})
    df = df[:30*3*288+7*288]
    model.train(df,epochs=100)
    # '''
    
    '''
    model = PVSCINet(input_len=288,output_len=144,hid_size=1,lr=0.001,dropout=0.1)
    model.load('/public/home/xuyh02/projects/pv_forecast/check/scinet.pkl')
    x = torch.randn(32, 288, 7)
    y = model.model(x)
    print(y.shape)
    '''
    
