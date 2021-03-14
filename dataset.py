from sklearn import datasets
import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import StandardScaler

class Data(Dataset):
    def __init__(self, n_samples, shuffle, noise, random_state=0, factor = .0):
        #инициализируем данные
        self.X, self.y = datasets.make_moons(n_samples = n_samples,
                                               shuffle = shuffle,
                                               noise = noise,
                                               random_state = random_state)
        #шкалируем данные
        sc = StandardScaler()
        self.X = sc.fit_transform(self.X)
        self.X, self.y = self.X.astype(np.float32), self.y.astype(np.int)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(np.array(self.y[idx]))

