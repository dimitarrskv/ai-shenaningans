from torch import nn
import fastcore.all as fc
import torch.nn.functional as F


class GeneralReLU(nn.Module):
    def __init__(self, leak=None, sub=None, maxv=None) -> None:
        super().__init__()
        fc.store_attr()

    def forward(self, x):
        x = F.leaky_relu(x, self.leak) if self.leak is not None else F.relu(x)
        if self.sub is not None: x -= self.sub 
        if self.maxv is not None: x.clamp_max(self.maxv)
        return x
    
def init_weights(m, leaky=0.):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(m.weight, a=leaky)