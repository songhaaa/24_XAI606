import os
import setproctitle
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np

# GPU 설정
setproctitle.setproctitle('Songha: XAI606')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
        
class MLPModel(nn.Module):
    def __init__(self, input_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def main():
    train = pd.read_csv('./dataset/train.csv', index_col='id')
    test = pd.read_csv('./dataset/test.csv', index_col='id')
    submission = pd.read_csv('./dataset/sample_submission.csv', index_col='id')

    all_data = pd.concat([train, test])
    all_data = all_data.drop('target', axis=1)  # 타깃값 제거

    encoder = OneHotEncoder()
    all_data_encoded = encoder.fit_transform(all_data).toarray()

    num_train = len(train)

    X_train = all_data_encoded[:num_train]
    X_test = all_data_encoded[num_train:]

    y = train['target'].values

    # 훈련 데이터, 검증 데이터 분리 (10%를 검증 데이터로 사용)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y,
                                                          test_size=0.1,
                                                          stratify=y,
                                                          random_state=10)

    train_dataset = CustomDataset(X_train, y_train)
    valid_dataset = CustomDataset(X_valid, y_valid)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1000, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPModel(X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in tqdm(range(100), desc='[Training]'):  # 300 에폭
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                y_valid_preds = []
                for X_batch, _ in valid_loader:
                    X_batch = X_batch.to(device)
                    outputs = model(X_batch).squeeze()
                    y_valid_preds.append(outputs.cpu().numpy())

            y_valid_preds = np.concatenate(y_valid_preds)
            roc_auc = roc_auc_score(y_valid, y_valid_preds)
            accuracy = ((y_valid_preds > 0.5) == y_valid).mean()

            print(f'Epoch {epoch + 1}, Validation set [ Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f} ]')

    model.eval()
    with torch.no_grad():
        y_test_preds = model(torch.tensor(X_test, dtype=torch.float32).to(device)).squeeze().cpu().numpy()

    submission['target'] = y_test_preds
    submission.to_csv('./result/submission4.csv', index=True)

if __name__ == '__main__':
    main()
