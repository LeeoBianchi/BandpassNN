import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import random_split


class VectorScalarDataset(Dataset):
    def __init__(self, X, y):
        """
        X: Tensor of shape (N, p) — inputs
        y: Tensor of shape (N,)   — scalar targets
        """
        self.X = X.float()
        self.y = y.float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ScalarNN(nn.Module):
    def __init__(
            self, 
            input_width:int=3, 
            hidden_width:int=8, 
            n_hidden:int=3, 
            hidden_func=torch.relu
            ):
        super(ScalarNN, self).__init__()
        assert n_hidden >= 1, "NN must have at least one hidden layer"
        # input layer (3 -> N)
        self.in_l = nn.Linear(input_width, hidden_width)
        # hidden layers (N -> N)
        for l in range(n_hidden):
            setattr(self, f"hl_{l+1}", nn.Linear(hidden_width, hidden_width))
        # output layer (N -> 1 value)
        self.out_l = nn.Linear(hidden_width, 1)
        self.hidden_func = hidden_func
        self.hidden_width = hidden_width
        self.n_hidden = n_hidden
        self.input_width = input_width

    @classmethod
    def from_dict(cls, dict):
        return cls(dict['input_width'], 
                   dict['hidden_width'],
                   dict['n_hidden'], 
                   dict['hidden_func'])

    def forward(self, x):
        x = self.hidden_func(self.in_l(x))       # hidden layer 1 + ReLU activation, maybe try gelu
        for l in range(self.n_hidden):
            x = self.hidden_func(getattr(self, f"hl_{l+1}")(x))
        x = self.out_l(x).squeeze(-1)       # output layer (reduce to 1 value)
        return x

    def reset_parameters(self):
        self.in_l.reset_parameters()
        for l in range(self.n_hidden):
            getattr(self, f"hl_{l+1}").reset_parameters()
        self.out_l.reset_parameters()

    @property
    def flops(self):
        count = 2*self.input_width*self.hidden_width \
                + 2*self.n_hidden*self.hidden_width**2 \
                + 2*self.hidden_width
        return count

class BPNN:
    """
    Description:
    ------------
        Wrapper class for a Feed Forward neural network designed for interpolating the 
        smooth bandpass integral.

    Attributes:
    ------------
        - dimensions (tuple[int]): A list of positive integers, which specifies the
            number of nodes in each of the networks layers. The first integer in the array
            defines the number of nodes in the input layer, the second integer defines number
            of nodes in the first hidden layer and so on until the last number, which
            specifies the number of nodes in the output layer.
        - hidden_func (Callable): The activation function for the hidden layers
        - loss_criterion (torch.nn.Module): Cost function criterion 
        - optimizer (torch.optim.Optimizer): Optimizer to be used for training the network
        - learning_rate (float): learning rate used for training the network 
    """

    def __init__(
        self,
        input_width:int=3,
        hidden_width:int=8,
        n_hidden:int=3,
        hidden_func:callable=torch.relu,
        normalize_y:bool=True
        ):
        self.model = ScalarNN(input_width=input_width,
                                 hidden_width=hidden_width,
                                 n_hidden=n_hidden,
                                 hidden_func=hidden_func)
        self.avg_train_loss_s = None
        self.avg_valid_loss_s = None
        self.y_mean = None
        self.y_std = None
        self.normalize_y = normalize_y

    @property
    def input_width(self):
        return self.model.input_width
    
    @property
    def hidden_width(self):
        return self.model.hidden_width
    
    @property
    def n_hidden(self):
        return self.model.n_hidden
    
    @property
    def hidden_func(self):
        return self.model.hidden_func

    def train(
        self,
        dataset:VectorScalarDataset,
        epochs:int=50,
        batch_size:int=64,
        learning_rate:float=1e-4,
        validation_rel_size:float=0.2,
        loss_class:nn.Module=nn.HuberLoss,
        optim_class:optim.Optimizer=optim.Adam,
        resume:bool=False,
        print_log:bool=True
        ):
        """
        fits the model on a dataset training the underlying NN. The dataset is authomatically split
        into training and validation datasets.
        """
        if self.normalize_y:
            self.y_mean = dataset.y.mean()
            self.y_std = dataset.y.std()
            dataset.y  = (dataset.y - self.y_mean) / self.y_std
        if not resume:
            self.model.reset_parameters()
        len_tot = len(dataset)
        val_size   = int(validation_rel_size * len_tot)
        train_size = len_tot - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        self.loss_criterion = loss_class()
        self.optimizer = optim_class(self.model.parameters(), learning_rate)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader  = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        avg_train_loss_s = np.zeros(epochs)
        avg_valid_loss_s = np.zeros(epochs)
        for epoch in range(epochs):
            # --- Training ---
            self.model.train()
            total_train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch #.to(device)
                y_batch = y_batch #.to(device)

                self.optimizer.zero_grad()
                preds = self.model(X_batch)
                loss  = self.loss_criterion(preds, y_batch)
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item() * len(y_batch)

            # --- Validation ---
            self.model.eval()
            total_val_loss = 0.0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch #.to(device)
                    y_batch = y_batch #.to(device)

                    preds = self.model(X_batch)
                    loss  = self.loss_criterion(preds, y_batch)
                    total_val_loss += loss.item() * len(y_batch)

            avg_train_loss = total_train_loss / len(train_loader.dataset)
            avg_val_loss   = total_val_loss   / len(val_loader.dataset)
            avg_train_loss_s[epoch] = avg_train_loss
            avg_valid_loss_s[epoch] = avg_val_loss
            if print_log:
                print(f"Epoch {epoch+1}: train={avg_train_loss:.4f}  val={avg_val_loss:.4f}")
        if resume:
            self.avg_train_loss_s = np.concatenate((self.avg_train_loss_s, avg_train_loss_s))
            self.avg_valid_loss_s = np.concatenate((self.avg_valid_loss_s, avg_valid_loss_s))
        else:
            self.avg_train_loss_s = avg_train_loss_s
            self.avg_valid_loss_s = avg_valid_loss_s


    def plot_training(self, out_path=None):
        if (self.avg_train_loss_s is None) or (self.avg_valid_loss_s is None):
            raise RuntimeError("Model has not been trained yet.")
        else:
            num_epochs = len(self.avg_train_loss_s)
            plt.figure(figsize=(10,6))
            plt.title("Training")
            plt.plot(np.arange(1, num_epochs+1), self.avg_train_loss_s, label="Train loss")
            plt.plot(np.arange(1, num_epochs+1), self.avg_valid_loss_s, label="Validation loss")
            plt.yscale('log')
            plt.xlabel("Epoch")
            plt.legend()
            if out_path is not None:
                plt.savefig(out_path)
            plt.close

    def save(self, path):
        torch.save({
        'ScalarNN_dict':{
            'model_state_dict': self.model.state_dict(),
            'input_width': self.input_width,
            'hidden_width': self.hidden_width,
            'n_hidden': self.n_hidden,
            'hidden_func': self.hidden_func
            },
        'BPNN_dict':{
            'y_mean': self.y_mean,
            'y_std':  self.y_std,
            'avg_train_loss_s': self.avg_train_loss_s,
            'avg_valid_loss_s': self.avg_valid_loss_s
            }
        }, path)
    
    def load(self, path):
        """
        Ovewrites underlying NN model by loading it from a
        """
        model_setup_dict = torch.load(path, weights_only=False)
        #NN model
        scalar_NN_dict = model_setup_dict['ScalarNN_dict']
        model = ScalarNN.from_dict(scalar_NN_dict)  # architecture first
        model.load_state_dict(scalar_NN_dict['model_state_dict'])
        #wrapper
        BPNN_dict = model_setup_dict['BPNN_dict']
        self.y_mean           = BPNN_dict['y_mean']
        self.y_std            = BPNN_dict['y_std']
        self.avg_train_loss_s = BPNN_dict['avg_train_loss_s']
        self.avg_valid_loss_s = BPNN_dict['avg_valid_loss_s']
        #TODO: maybe implement a for cycle with get/setattr
    
    def eval(self, X:torch.Tensor):
        """
        Evaluates the model inferring the integral value.
        """
        self.model.eval()
        with torch.no_grad():
            if self.normalize_y:
                assert self.y_std is not None and self.y_mean is not None, \
                    "Tried evaluation without the normalization parameters being set."        
                return self.model(X) * self.y_std + self.y_mean
            else:
                return self.model(X)

