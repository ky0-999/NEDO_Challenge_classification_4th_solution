import pandas as pd
import numpy as np
import sys
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedKFold
from preprocess import SeqDataset
import torch
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ネットワークモデルの定義
class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, dropout_rate=0.5):
        super(DenseBlock, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.dropout_rate = dropout_rate  

        for i in range(num_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))

    def _make_layer(self, in_channels, growth_rate):
        conv = torch.nn.Conv1d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1)
        bn = torch.nn.BatchNorm1d(growth_rate)
        relu = torch.nn.SELU()
        dropout = torch.nn.Dropout(self.dropout_rate)  

        return torch.nn.Sequential(conv, bn, relu, dropout)

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat((x, out), 1) 
        return x

class Net1DBN(torch.nn.Module):
    def __init__(self, num_channels, num_classes, growth_rate=32, num_layers=4, dropout_rate=0.0):
        super(Net1DBN, self).__init__()

        self.conv1 = torch.nn.Conv1d(num_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.relu = torch.nn.SELU()
        self.dropout = torch.nn.Dropout(dropout_rate) 


        self.dense1 = DenseBlock(64, growth_rate, num_layers, dropout_rate)
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size=2, stride=2)

        self.dense2 = DenseBlock(64 + growth_rate * num_layers, growth_rate, num_layers, dropout_rate)  
        self.maxpool2 = torch.nn.MaxPool1d(kernel_size=2, stride=2)

        self.dense3 = DenseBlock(64 + growth_rate * num_layers * 2, growth_rate, num_layers, dropout_rate)  
        self.gap = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(64 + growth_rate * num_layers * 3, num_classes)  

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x) 

        x = self.dense1(x)
        x = self.maxpool1(x)

        x = self.dense2(x)
        x = self.maxpool2(x)

        x = self.dense3(x)
        x = self.gap(x)
        x = x.squeeze(2)

        x = self.fc(x)

        return x


def train(log_interval, model, device, train_loader, optimizer, epoch, iteration):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    for sample_batched in train_loader:
        # print(sample_batched) 
        data, target = sample_batched['seq'].to(device), sample_batched['label'].to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.max(1, keepdim=True)[1]
        correct = pred.eq(target.view_as(pred)).sum().item()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        iteration += 1
        if iteration % log_interval == 0:
            sys.stdout.write('\repoch:{0:>3} iteration:{1:>6} train_loss: {2:.6f} train_accracy: {3:5.2f}%'.format(
                            epoch, iteration, loss.item(), 100.*correct/float(len(sample_batched['label']))))
            sys.stdout.flush()
    return iteration


def val(model, device, test_loader):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sample_batched in test_loader:
            data, target = sample_batched['seq'].to(device), sample_batched['label'].to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= float(len(test_loader.dataset))
    correct /= float(len(test_loader.dataset))
    return test_loss, 100. * correct


def evaluate(model, device, test_loader):
    preds = []
    trues = []
    model.eval()
    with torch.no_grad():
        for sample_batched in test_loader:
            data, target = sample_batched['seq'].to(device), sample_batched['label'].to(device)
            output = model(data)
            pred = [test_loader.dataset.class_names[i] for i in list(output.max(1)[1].cpu().detach().numpy())]
            preds += pred
            true = [test_loader.dataset.class_names[i] for i in list(target.cpu().detach().numpy())]
            trues += true
    labels = test_loader.dataset.class_names
    cm = confusion_matrix(trues, preds, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()
    cr = classification_report(trues, preds, target_names=labels)
    print(cr)
    correct = 0
    for pred, true in zip(preds, trues):
        if pred == true:
            correct += 1
    df = pd.DataFrame({'pred': preds, 'true': trues})

    return correct/len(trues), df

def train_eval(root_dir, subject_id, log_interval, num_epoches, seq_length, transform=None, num_channels=140, num_classes=3):
    model = Net1DBN(num_channels=num_channels, num_classes=num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_loader = torch.utils.data.DataLoader(SeqDataset(root_dir, seq_length=seq_length, is_train=True, subject_id=subject_id, transform=transform), batch_size=64, shuffle=True) # type: ignore

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []  
    fold_loss = []
    models = {} 
    if subject_id not in models:
        models[subject_id] = {}
    for fold, (train_index, val_index) in enumerate(kf.split(np.zeros(len(train_loader.dataset)), train_loader.dataset.seq_labels)):
        print(f'Fold {fold + 1}/{kf.n_splits}')
        model = Net1DBN(num_channels=num_channels, num_classes=num_classes)
        model.to(device)
        train_subset = torch.utils.data.Subset(train_loader.dataset, train_index)
        val_subset = torch.utils.data.Subset(train_loader.dataset, val_index)
        train_subset_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
        val_subset_loader = torch.utils.data.DataLoader(val_subset, batch_size=64, shuffle=False)
        best_val_loss = float('inf')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    
        iteration = 0
        for epoch in range(num_epoches):
            iteration = train(log_interval, model, device, train_subset_loader, optimizer, epoch, iteration)
            val_loss, accuracy = val(model, device, val_subset_loader)
            scheduler.step(val_loss)
            if epoch % 10 == 0:
                print('\n  Validation: Accuracy: {0:.2f}%  test_loss: {1:.6f}'.format(accuracy, val_loss))

            # 検証データに対するクロスエントロピー損失が最良の場合、モデルを保存する
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_accuracy = accuracy  
                best_model_state = model.state_dict() 
                print(f'Model updated    accuracy: {accuracy: .2f}  validation loss: {best_val_loss:.6f} at epoch {epoch + 1}') 
        model.load_state_dict(best_model_state)
        models[subject_id][fold] = model
        fold_accuracies.append(best_accuracy)
        fold_loss.append(best_val_loss)

    # 各foldの精度の平均を計算
    mean_accuracy = np.mean(fold_accuracies)
    mean_loss = np.mean(fold_loss)
    print(f'Mean accuracy for subject {subject_id}: {mean_accuracy:.2f}%')
    print(f'Mean loss for subject {subject_id}: {mean_loss:.6f}%')

    return models