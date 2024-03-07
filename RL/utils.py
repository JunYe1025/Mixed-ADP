import numpy as np
import torch
import pickle

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        # print('x',x,'type_x',type(x),'size_x',x.size)
        x = np.array(x)
        # print('x', x, 'type_x',type(x),'size_x',x.size)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)

        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.running_ms, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.running_ms = pickle.load(f)


class Shared_RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = torch.tensor(0).share_memory_()
        self.mean = torch.zeros(shape).share_memory_()
        self.S = torch.zeros(shape).share_memory_()
        self.std = torch.sqrt(self.S).share_memory_()

    def update(self, x):
        x = torch.tensor(x)
        self.n += 1
        if self.n == 1:
            self.mean[:] = x
            self.std[:] = torch.zeros_like(x)
        else:
            old_mean = self.mean.clone()
            self.mean[:] = old_mean + (x - old_mean) / self.n
            self.S[:] = self.S + (x - old_mean) * (x - self.mean)
            self.std[:] = torch.sqrt(self.S / self.n)

class Shared_Normalization:
    def __init__(self, shape):
        self.running_ms = Shared_RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        x = torch.tensor(x, dtype=torch.float32)
        if update:
            self.running_ms.update(x)
        x_normalized = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        return x_normalized.numpy()  # Convert back to numpy if necessary

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.running_ms, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.running_ms = pickle.load(f)