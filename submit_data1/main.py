import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from preprocess import make_data, make_data_test, TestSeqDataset, transform
from train import train_eval
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def output_pred(test_loader, models, subject_id, device, fold):
    prediction = []
    model = models[subject_id][fold]  
    train_dir = os.path.join('test_modeling', 'train', subject_id)
    class_names = os.listdir(train_dir)

    class_names.sort()
    model.eval()

    # 推論
    with torch.no_grad():
        for sample_batched in test_loader:
            data = sample_batched['seq'].to(device)
            output = model(data)
            probabilities = F.softmax(output, dim=1)
            prediction.append(probabilities.cpu().numpy())
            
    prediction = np.concatenate(prediction, axis=0)
    return prediction

# ハイパーパラメータ
log_interval = 5000
num_epoches = 200
seq_length = 250
num_folds = 5
batch_size = 64
num_channels = 140  
num_classes = 3    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# path
src_dir = '.'
dst_dir = 'test_modeling'
subject_ids = ['subject0', 'subject1', 'subject2', 'subject3', 'subject4']


# -------------------------------------main文----------------------------------------
# 学習データ作成
for subject_id in subject_ids:
    make_data(src_dir=src_dir, dst_dir=dst_dir, subject_id=subject_id)

# テストデータ作成
make_data_test(src_dir=src_dir, dst_dir=dst_dir)

predictions  = []
for subject_id in subject_ids:
    train_dir = os.path.join('test_modeling', 'train', subject_id)

    # 学習と前処理
    models = train_eval(train_dir, subject_id, log_interval, num_epoches, seq_length, transform, num_channels, num_classes)
    all_predictions = [] 

    for fold in range(num_folds):
        model = models[subject_id][fold]
        test_dir = os.path.join('test_modeling', 'test', subject_id)

        test_dataset = TestSeqDataset(root=test_dir, seq_length=250, subject_id=subject_id,transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 推論
        prediction = output_pred(test_loader, models, subject_id, device, fold)
        all_predictions.append(prediction)
    avg_probabilities = np.mean(all_predictions, axis=0)
    final_predictions = np.argmax(avg_probabilities, axis=1)
    class_names = os.listdir(train_dir)
    class_names.sort()
    pre={}
    for j, index in enumerate(final_predictions):
        pre['{}_{:03d}'.format(subject_id, j)] = class_names[index] 
        result = pd.Series(pre)

    predictions.append(pd.DataFrame(result)) 

final_submit = pd.concat(predictions, axis=0)
final_submit.to_csv('submit_data1.csv', header=False)