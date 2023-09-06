'''
SCINet for PV Forecasting
'''
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.PatchTST import PatchTST
from datetime import datetime,timedelta
from utils.tools import StandardScaler,adjust_learning_rate,EarlyStopping
from utils.data_loader import PVDataset
from utils.timefeatures import time_features
import time
from datetime import datetime
import argparse

import os
os.environ['CUDA_VISIBLE_DEVICES']='3'

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=42, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, default='test', help='model id')
parser.add_argument('--model', type=str, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='pv', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


# DLinear
#parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=3, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=3, help='decoder input size')
parser.add_argument('--c_out', type=int, default=3, help='output size')
parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()


class PVPatchTST():
    def __init__(self,configs=args,input_len=288,output_len=144,enc_in=3,e_layers=3,n_heads=4,
                 d_model=16,d_ff=128,dropout=0.3,fc_dropout=0.3,head_dropout=0,patch_len=16,
                 stride=8,lr=0.0001,batch_size=64):
        
        self.input_len = input_len
        self.output_len = output_len
        self.lr = lr
        self.timeenc = 1
        self.freq = 'h'
        self.batch_size = batch_size

        configs.enc_in = enc_in
        configs.seq_len = input_len
        configs.pred_len = output_len
        
        configs.e_layers = e_layers
        configs.n_heads = n_heads
        configs.d_model = d_model
        configs.d_ff = d_ff
        configs.dropout = dropout
        configs.fc_dropout = fc_dropout
        configs.head_dropout = head_dropout
    
        configs.patch_len = patch_len
        configs.stride = stride
        self.configs = configs

        self.model,self.mean,self.std = self._build_model(configs)
        assert input_len==288, 'if not 288, timedelta=1 in below predict() should modify'
        
        
    def _build_model(self,configs):
        model = PatchTST(configs)
        mean = np.zeros((1,3))
        std = np.ones((1,3))
        return model,mean,std

    def load(self,load_file):
        check = torch.load(load_file)
        self.model.load_state_dict(check['state_dict'])
        self.mean = check['mean']
        self.std = check['std']

    def train(self,df,epochs=30):
        '''
        :param df: dataframe, |--ts--|--pv--|--radiation--|--temperature--|
        '''
        self.model = self.model.cuda()
        self.model.train()
        train_set = PVDataset(df=df,flag='train',months=3,input_len=self.input_len,output_len=self.output_len)
        scaler = train_set.scaler
        train_loader = DataLoader(train_set,batch_size=self.batch_size,shuffle=True,drop_last=True)
        val_set = PVDataset(df=df,flag='valid',months=3,input_len=self.input_len,output_len=self.output_len)
        val_loader = DataLoader(val_set,batch_size=self.batch_size,shuffle=False)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.MSELoss()
        writer = SummaryWriter('./runs/train_patchtst{}_{}_h{}_dm{}_dff{}_dp{}_fcdp{}_ptl{}_s{}_lr{}_bt{}'.format(self.input_len,
            self.output_len,self.configs.n_heads,self.configs.d_model,self.configs.d_ff,self.configs.dropout,
            self.configs.fc_dropout,self.configs.patch_len,self.configs.stride,self.lr,self.batch_size))
        early_stopping = EarlyStopping(patience=100, verbose=True)
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
            adjust_learning_rate(optimizer, epoch+1, lr=self.lr,lradj=3)

            # Early Stopping
            save_check = {'state_dict':self.model.state_dict(),'mean':scaler.mean,'std':scaler.std}
            save_path = './check/patchtst{}_{}_h{}_dm{}_dff{}_dp{}_fcdp{}_ptl{}_s{}_lr{}_bt{}_best.pkl'.format(self.input_len,
            self.output_len,self.configs.n_heads,self.configs.d_model,self.configs.d_ff,self.configs.dropout,
            self.configs.fc_dropout,self.configs.patch_len,self.configs.stride,self.lr,self.batch_size)
            early_stopping(val_l, self.model, save_check, save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break   # 暂时不break
        # save model
        save_check = {'state_dict':self.model.state_dict(),'mean':scaler.mean,'std':scaler.std}
        torch.save(save_check,'./check/patchtst{}_{}_h{}_dm{}_dff{}_dp{}_fcdp{}_ptl{}_s{}_lr{}_bt{}_ep{}.pkl'.format(self.input_len,
            self.output_len,self.configs.n_heads,self.configs.d_model,self.configs.d_ff,self.configs.dropout,
            self.configs.fc_dropout,self.configs.patch_len,self.configs.stride,self.lr,self.batch_size,epoch+1))


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
    model = PVPatchTST(input_len=288,output_len=144,enc_in=3,e_layers=3,n_heads=4,
                 d_model=16,d_ff=128,dropout=0.3,fc_dropout=0.3,head_dropout=0,patch_len=16,
                 stride=8,lr=0.0001,batch_size=64)
    df_raw = pd.read_csv('/public/home/xuyh02/projects/pv_forecast/data/pv_data.csv')
    df = pd.DataFrame({'ts': df_raw['0'], 'pv': df_raw['6'],'radiation':df_raw['4'],'temperature':df_raw['5']})
    df = df[:30*3*288+7*288]
    model.train(df,epochs=100)
    # '''
    
    # # '''
    # # import pdb;pdb.set_trace()
    # model = PVPatchTST(input_len=288,output_len=144,enc_in=3,e_layers=3,n_heads=4,
    #              d_model=16,d_ff=128,dropout=0.3,fc_dropout=0.3,head_dropout=0,patch_len=16,
    #              stride=8,lr=0.0001,batch_size=64)
    # # model.load('/public/home/xuyh02/projects/pv_forecast/check/scinet.pkl')
    # x = torch.randn(32, 288, 3)
    # y = model.model(x)
    # print(y.shape)
    # # '''
    
