import os
import pandas as pd
import numpy as np
import random
import glob
from pymatreader import read_mat
from torch.utils.data import Dataset
import pywt
import random
import torch
from scipy.signal import savgol_filter

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

labels = {
        '11': 'frontside_kickturn',
        '12': 'backside_kickturn',
        '13': 'pumping',
        '21': 'frontside_kickturn',
        '22': 'backside_kickturn',
        '23': 'pumping'
    }

# 双極電出法を参考にした特徴量作成
def plus_ultra(df:pd.DataFrame):
    bananas = []
    bananas.append(['Fpz ','Fp1 ','AF7 ','F7  ','FT7 ','T7  ','TP7 ','P7  ','PO7 ','O1  ','Oz  '])
    bananas.append(['Fpz ','AF3 ','F5  ','FC5 ','C5  ','CP5 ','P5  ','PO3 ','Oz  '])
    bananas.append(['Fpz ','F3  ','FC3 ','C3  ','CP3 ','P3  ','PO3 ','Oz  '])
    bananas.append(['Fpz ','F1  ','FC1 ','C1  ','CP1 ','P1  ','Oz  '])

    bananas.append(['Fpz ','Fp2 ','AF8 ','F8  ','FT8 ','T8  ','TP8 ','P8  ','PO8 ','O2  ','Oz  '])
    bananas.append(['Fpz ','AF4 ','F6  ','FC6 ','C6  ','CP6 ','P6  ','PO4 ','Oz  '])
    bananas.append(['Fpz ','F4  ','FC4 ','C4  ','CP4 ','P4  ','PO4 ','Oz  '])
    bananas.append(['Fpz ','F2  ','FC2 ','C2  ','CP2 ','P2  ','Oz  '])

    bananas.append(['Fpz ','AFz ','Fz  ','FCz ','Cz  ','CPz ','Pz  ','PO7 ','Oz  '])
    
    a = ['Fp1 ', 'Fp2 ', 'F3  ', 'F4  ', 'C3  ', 'C4  ', 'P3  ', 'P4  ', 'O1  ',
       'O2  ', 'F7  ', 'F8  ', 'T7  ', 'T8  ', 'P7  ', 'P8  ', 'Fz  ', 'Cz  ',
       'Pz  ', 'Oz  ', 'FC1 ', 'FC2 ', 'CP1 ', 'CP2 ', 'FC5 ', 'FC6 ', 'CP5 ',
       'CP6 ', 'FT9 ', 'FT10', 'FCz ', 'AFz ', 'F1  ', 'F2  ', 'C1  ', 'C2  ',
       'P1  ', 'P2  ', 'AF3 ', 'AF4 ', 'FC3 ', 'FC4 ', 'CP3 ', 'CP4 ', 'PO3 ',
       'PO4 ', 'F5  ', 'F6  ', 'C5  ', 'C6  ', 'P5  ', 'P6  ', 'AF7 ', 'AF8 ',
       'FT7 ', 'FT8 ', 'TP7 ', 'TP8 ', 'PO7 ', 'PO8 ', 'Fpz ', 'CPz ', 'POz ',
       'Iz  ', 'F9  ', 'F10 ', 'P9  ', 'P10 ', 'PO9 ', 'PO10', 'O9  ', 'O10 ']

    for banana in bananas:
        df = df.copy()  
        for i in range(1,len(banana)):
            name = banana[i-1] + '-' + banana[i]
            df.loc[:, name] = df.loc[:, banana[i-1]] - df.loc[:, banana[i]]
    
    return df

# 学習データのmat→csv
def make_data(src_dir, dst_dir, subject_id):
    print(subject_id)
    os.makedirs(os.path.join(dst_dir, 'train', subject_id), exist_ok=True)

    counts = {'frontside_kickturn':0, 'backside_kickturn':0, 'pumping':0}
    for fname in os.listdir(os.path.join(src_dir, 'train', subject_id)):
        data = read_mat(os.path.join(src_dir, 'train', subject_id, fname))
        event = pd.DataFrame(data['event'])[['init_time', 'type']] 
        ts = pd.DataFrame(np.concatenate([np.array([data['times']]), data['data']]).T, columns=['Time']+list(data['ch_labels']))
        for i, d in event.iterrows():
            it = d['init_time']+0.2
            et = d['init_time']+0.7
            event_type = str(int(d['type']))
            ts_seg = ts[(ts['Time']>=it*1e3)&(ts['Time']<=et*1e3)]

            if not os.path.exists(os.path.join(dst_dir, 'train', subject_id, labels[event_type])):
                os.makedirs(os.path.join(dst_dir, 'train', subject_id, labels[event_type]), exist_ok=True)
            del ts_seg['Time']
            ts_seg = plus_ultra(ts_seg)
            ts_seg.to_csv(os.path.join(dst_dir, 'train', subject_id, labels[event_type], '{:03d}.csv'.format(counts[labels[event_type]])), index=False, header=False)

            counts[labels[event_type]]+=1
# 学習データのmat→csv
def make_data_test(src_dir, dst_dir):
    for fname in os.listdir(os.path.join(src_dir, 'test')):
        data = read_mat(os.path.join(src_dir, 'test', fname))
        
        if 'data' in data and 'ch_labels' in data:
            print(f"Processing file: {fname}")
            samples, channels, sequence_length = data['data'].shape
            
            subject_id = os.path.splitext(fname)[0]
            subject_dir = os.path.join(dst_dir, 'test', subject_id)
            os.makedirs(subject_dir, exist_ok=True)
            
            for sample_idx in range(samples):
                ts = data['data'][sample_idx].T  # shape (72, 250)
                ts_df = pd.DataFrame(ts, columns=list(data['ch_labels']))
                ts_df = plus_ultra(ts_df)
                
                output_file = os.path.join(subject_dir, f"{sample_idx:03d}.csv")
                ts_df.to_csv(output_file, header=None,index=None)
                
        else:
            print(f"Missing 'data' or 'ch_labels' in {fname}.")

# 学習データの前処理
class SeqDataset(Dataset):
    def __init__(self, root, seq_length, is_train, subject_id, transform=None):
        self.transform = transform
        self.seqs = []
        self.seq_labels = []
        self.class_names = os.listdir(root)
        self.class_names.sort()
        self.numof_classes = len(self.class_names)
        self.seq_length = seq_length
        self.is_train = is_train
        self.subject_id = subject_id

        for (i,x) in enumerate(self.class_names):
            temp = glob.glob(os.path.join(root, x, '*'))
            temp.sort()
            self.seq_labels.extend([i]*len(temp))
            for t in temp:
                df = pd.read_csv(t, header=None)
                tensor = preprocess(df)
                self.seqs.append(tensor)

    def __getitem__(self, index):
        seq = self.seqs[index]
        if self.transform is not None:
            seq = self.transform(seq, is_train=self.is_train, seq_length=self.seq_length, subject_id=self.subject_id)
        return {'seq':seq, 'label':self.seq_labels[index]}


    def __len__(self):
        return len(self.seqs)

# テストデータの前処理
class TestSeqDataset(Dataset):
    def __init__(self, root, seq_length, subject_id,transform=None):
        self.transform = transform
        self.seqs = []
        self.seq_length = seq_length
        self.subject_id = subject_id
        
        temp = glob.glob(os.path.join(root,'*'))
        temp.sort()
        for t in temp:
            df = pd.read_csv(t, header=None)
            tensor = preprocess(df)
            self.seqs.append(tensor)

    def __getitem__(self, index):
        seq = self.seqs[index]
        if self.transform is not None:
            seq = self.transform(seq, is_train=False, seq_length=self.seq_length, subject_id=self.subject_id)
        return {'seq': seq}  

    def __len__(self):
        return len(self.seqs)

# 転置と正規化実行
def preprocess(df: pd.DataFrame)->np.ndarray:

    mat = df.T.values
    mat = standardization(mat, axis=1)

    return mat

# 正規化
def standardization(a, axis=None, ddof=0):
    a_mean = a.mean(axis=axis, keepdims=True)
    a_std = a.std(axis=axis, keepdims=True, ddof=ddof)
    a_std[np.where(a_std==0)] = 1

    return (a - a_mean) / a_std

def savitzky_golay_filter(ts, window_length=5, polyorder=2):
    return savgol_filter(ts, window_length=window_length, polyorder=polyorder, axis=1)

def time_jittering(signal, jitter_factor=0.1):
    """
    時系列データに時間的ジッターを追加する関数
    signal: 時系列データ (numpy array) - 形状 (channels, time_length)
    jitter_factor: シフトの割合（全体の時間長に対する割合でシフトを決定）
    """
    # データの時間長
    time_length = signal.shape[1]
    
    # シフトの大きさ（全体の時間の jitter_factor の割合でランダムシフト）
    shift_amount = int(np.random.uniform(-jitter_factor, jitter_factor) * time_length)
    
    if shift_amount > 0:
        # 右にシフト: 左側にゼロパディング
        jittered_signal = np.pad(signal[:, :-shift_amount], ((0, 0), (shift_amount, 0)), mode='constant')
    elif shift_amount < 0:
        # 左にシフト: 右側にゼロパディング
        jittered_signal = np.pad(signal[:, -shift_amount:], ((0, 0), (0, -shift_amount)), mode='constant')
    else:
        # シフトなし
        jittered_signal = signal.copy()
    
    return jittered_signal.astype(np.float32)

def add_noise(data, noise_level=0.01):
    noise = np.random.normal(0, noise_level, data.shape)
    data_noisy = data + noise
    return data_noisy.astype(np.float32)


def transform(array, is_train, seq_length, subject_id):
    if is_train:
        _, n = array.shape
        s = random.randint(0, n - seq_length)
        ts = array[:, s:s + seq_length].astype(np.float32)

        if subject_id == 'subject0':
            ts = time_jittering(ts, jitter_factor=0.1)
            ts = savitzky_golay_filter(ts)
            
        elif subject_id == 'subject1':
            ts = time_jittering(ts, jitter_factor=0.1)
        
        elif subject_id == 'subject2':
            ts = ts
            # ts = savitzky_golay_filter(ts)

        elif subject_id == 'subject3':
            # ts = time_jittering(ts, jitter_factor=0.1)
            ts = savitzky_golay_filter(ts)

        elif subject_id == 'subject4':
            ts = time_jittering(ts, jitter_factor=0.1)
            ts = add_noise(ts).astype(np.float32)
    #         ts = time_jittering(ts, jitter_factor=0.1)

        ts = ts.astype(np.float32)
        return ts
    
    else:
        ts = array[:,:seq_length].astype(np.float32)
        if subject_id == 'subject0':
            ts = savitzky_golay_filter(ts)
        # if subject_id == 'subject1':
        #     ts = wavelet_transform(ts, wavelet='sym4')
        # if subject_id == 'subject2':
        if subject_id == 'subject3':
            ts = savitzky_golay_filter(ts)
        # if subject_id == 'subject4':

    return ts

