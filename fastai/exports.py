from utils import Callback
from torch import nn



class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)

class PrintCallback(Callback): 
    def before_epoch(self):
        print('before epoch')

    def before_fit(self):
        print('Before Fit Print')
    
    def after_fit(self):
        print('After Fit Print')