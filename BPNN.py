import torch
import random
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
    
    def state_dict(self, **kwargs):
        sd = super().state_dict(**kwargs)
        sd['hidden_width']  = self.hidden_width
        sd['train_losses'] = self.train_losses
        sd['val_losses']   = self.val_losses
        return sd

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

class BPNN(nn.Module):
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
        normalize_y:bool=True,
        seed:int|None=None
        ):
        self._set_seed(seed)
        #NN
        super(BPNN, self).__init__()
        assert n_hidden >= 1, "NN must have at least one hidden layer"
        # input layer (3 -> N)
        self.in_l = nn.Linear(input_width, hidden_width)
        # hidden layers (N -> N)
        for l in range(n_hidden):
            setattr(self, f"hl_{l+1}", nn.Linear(hidden_width, hidden_width))
        # output layer (N -> 1 value)
        self.out_l = nn.Linear(hidden_width, 1)

        #params
        self.hidden_func    = hidden_func
        self.hidden_width   = hidden_width
        self.input_width    = input_width
        self.n_hidden       = n_hidden
        self.normalize_y    = normalize_y
        self.train_losses   = None
        self.valid_losses   = None
        self.y_mean         = None
        self.y_std          = None

    @property
    def flops(self):
        count = 2*self.input_width*self.hidden_width \
                + 2*self.n_hidden*self.hidden_width**2 \
                + 2*self.hidden_width
        return count

    def forward(self, x):
        x = self.hidden_func(self.in_l(x))  # hidden layer 1 + ReLU activation, maybe try gelu
        for l in range(self.n_hidden):
            x = self.hidden_func(getattr(self, f"hl_{l+1}")(x))
        x = self.out_l(x).squeeze(-1)       # output layer (reduce to 1 value)
        return x

    def _set_seed(self, seed):
        self.seed = seed
        if seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

    def reset_parameters(self):
        self.in_l.reset_parameters()
        for l in range(self.n_hidden):
            getattr(self, f"hl_{l+1}").reset_parameters()
        self.out_l.reset_parameters()

    def state_dict(self, **kwargs):
        sd = super().state_dict(**kwargs)
        sd['input_width']   = self.input_width
        sd['hidden_width']  = self.hidden_width
        sd['hidden_func']   = self.hidden_func
        sd['n_hidden']      = self.n_hidden
        sd['train_losses']  = self.train_losses
        sd['valid_losses']  = self.valid_losses
        sd['normalize_y']   = self.normalize_y
        sd['y_mean']        = self.y_mean
        sd['y_std']         = self.y_std
        sd['seed']          = self.seed
        return sd

    def save(self, path:str):
        torch.save(self.state_dict(), path)
    
    def load_state_dict(self, state_dict:dict, strict:bool=True):
        self.input_width    = state_dict.pop('input_width', self.input_width)
        self.hidden_width   = state_dict.pop('hidden_width',self.hidden_width)
        self.hidden_func    = state_dict.pop('hidden_func', self.hidden_func)
        self.n_hidden       = state_dict.pop('n_hidden',    self.n_hidden)
        self.train_losses   = state_dict.pop('train_losses', [])
        self.valid_losses   = state_dict.pop('valid_losses', [])
        self.normalize_y    = state_dict.pop('normalize_y', True)
        self.y_mean         = state_dict.pop('y_mean',      None)
        self.y_std          = state_dict.pop('y_std',       None)
        self._set_seed(state_dict.pop('seed', None))
        super().load_state_dict(state_dict, strict=strict)

    def load(self, path:str):
        self.load_state_dict(torch.load(path, weights_only=False), strict=True)

    def fit(
        self,
        dataset:VectorScalarDataset,
        epochs:int=50,
        batch_size:int=64,
        learning_rate:float=1e-4,
        validation_rel_size:float=0.2,
        loss_class:nn.Module=nn.HuberLoss,
        optim_class:optim.Optimizer=optim.Adam,
        resume:bool=False,
        print_log:bool=True,
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
            self.reset_parameters()
        len_tot = len(dataset)
        val_size   = int(validation_rel_size * len_tot)
        train_size = len_tot - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        self.loss_criterion = loss_class()
        self.optimizer = optim_class(self.parameters(), learning_rate)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader  = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        avg_train_loss_s = np.zeros(epochs)
        avg_valid_loss_s = np.zeros(epochs)
        try:
            for epoch in range(epochs):
                # --- Training ---
                self.train()
                total_train_loss = 0.0

                for X_batch, y_batch in train_loader:
                    X_batch = X_batch #.to(device)
                    y_batch = y_batch #.to(device)

                    self.optimizer.zero_grad()
                    preds = self(X_batch)
                    loss  = self.loss_criterion(preds, y_batch)
                    loss.backward()
                    self.optimizer.step()

                    total_train_loss += loss.item() * len(y_batch)

                # --- Validation ---
                self.eval()
                total_val_loss = 0.0

                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch #.to(device)
                        y_batch = y_batch #.to(device)

                        preds = self(X_batch)
                        loss  = self.loss_criterion(preds, y_batch)
                        total_val_loss += loss.item() * len(y_batch)

                avg_train_loss = total_train_loss / len(train_loader.dataset)
                avg_val_loss   = total_val_loss   / len(val_loader.dataset)
                avg_train_loss_s[epoch] = avg_train_loss
                avg_valid_loss_s[epoch] = avg_val_loss
                if print_log:
                    print(f"Epoch {epoch+1}: train={avg_train_loss:.4f}  val={avg_val_loss:.4f}")
        except KeyboardInterrupt:
            print(f"\nTraining interrupted — freezing model at epoch {epoch+1}.")
            avg_train_loss_s = avg_train_loss_s[:epoch+1]
            avg_valid_loss_s = avg_valid_loss_s[:epoch+1]
        finally:
            if resume:
                self.train_losses = np.concatenate((self.train_losses, avg_train_loss_s))
                self.valid_losses = np.concatenate((self.valid_losses, avg_valid_loss_s))
            else:
                self.train_losses = avg_train_loss_s
                self.valid_losses = avg_valid_loss_s

    def plot_training(self, out_path=None):
        if (self.train_losses is None) or (self.valid_losses is None):
            raise RuntimeError("Model has not been trained yet.")
        else:
            num_epochs = len(self.train_losses)
            plt.figure(figsize=(10,6))
            plt.title("Training")
            plt.plot(np.arange(1, num_epochs+1), self.train_losses, label="Train loss")
            plt.plot(np.arange(1, num_epochs+1), self.valid_losses, label="Validation loss")
            plt.yscale('log')
            plt.xlabel("Epoch")
            plt.legend()
            if out_path is not None:
                plt.savefig(out_path)
            plt.close

    def predict(self, X:torch.Tensor):
        """
        Evaluates the model inferring the integral value.
        """
        self.eval()
        with torch.no_grad():
            if self.normalize_y:
                assert self.y_std is not None and self.y_mean is not None, \
                    "Tried evaluation without the normalization parameters being set."        
                return self(X) * self.y_std + self.y_mean
            else:
                return self(X)
   
    def __repr__(self):
        lines = []
        lines.append('MODEL:')
        lines.append(super().__repr__())
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lines.append(f"Trainable parameters: {total}")
        lines.append(f"Total FLOPs: {self.flops}")
        return '\n'.join(lines)

