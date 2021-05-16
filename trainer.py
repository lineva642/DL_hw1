from datetime import datetime
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, lr, optimizer = None, criterion = None):
        self.model = model
        self.criterion = CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        
        self.experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter("runs/" + self.experiment_name)
        # self.writer = SummaryWriter(str('runs\' self.experiment_name))
    #метод, обучающий сеть    
    def fit(self, train_dataloader, n_epochs):
        #переводим модель в режим тренировки
        self.model.train()
        for epoch in range(n_epochs):
            print('epoch: ', epoch)
            epoch_loss = 0
            for i, (x_batch, y_batch) in enumerate(train_dataloader):
                if (epoch == 0) and (i == 0):
                    self.writer.add_graph(self.model, x_batch)
                #обнуляем градиенты    
                self.optimizer.zero_grad()
                #прямой проход по сети
                output = self.model(x_batch)
                # print(x_batch, y_batch)
                loss = self.criterion(output, y_batch.long())
                #дифференцируем loss
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            print(epoch_loss/len(train_dataloader))
            self.writer.add_scalar('training loss on epoch end', epoch_loss)
    #делаем предсказания                
    def predict(self, test_dataloader):
        all_outputs = torch.tensor([], dtype=torch.long)
        # не считаем градиенты
        self.model.eval()
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(test_dataloader):
                output_batch = self.model(x_batch)
                _, predicted = torch.max(output_batch.data, 1)
                all_outputs = torch.cat((all_outputs, predicted), 0)
        return all_outputs        
    #то же самое для предсказанных вероятностей
    def predict_proba(self, test_dataloader):
        all_outputs = torch.tensor([], dtype=torch.float32)
        # не считаем градиенты
        self.model.eval()
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(test_dataloader):
                output_batch = self.model(x_batch)
                # _, predicted = torch.max(output_batch.data, 1)
                all_outputs = torch.cat((all_outputs, output_batch), 0)
        return all_outputs
    #получим ответ в виде тензора
    def predict_proba_tensor(self, T):
        self.model.eval()
        with torch.no_grad():
            output = self.model(T)
        return output
    
def get_data_from_datasets(train_dataset, test_dataset):
    X_train = train_dataset.X.astype(np.float32)
    X_test = test_dataset.X.astype(np.float32)
    y_train = train_dataset.y.astype(np.int)
    y_test = test_dataset.y.astype(np.int)
    return X_train, X_test, y_train, y_test
def predict_proba_on_mesh_tensor(clf, xx, yy):
    q = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    # print(q)
    # print(type(q))
    Z = clf.predict_proba_tensor(q)[:, 1]
    Z = Z.reshape(xx.shape)
    # print(Z)
    return Z
    