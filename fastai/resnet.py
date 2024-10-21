from torch import nn
from utils import conv
from activations import GeneralReLU
from functools import partial
import fastcore.all as fc

act_gr = partial(GeneralReLU, leak=0.1, sub=0.4)

def _conv_bock(ni, nf, stride, act=act_gr, norm=None, ks=3):
    return nn.Sequential(
        conv(ni, nf, stride=1, act=act, norm=norm, ks=ks),
        conv(nf, nf, stride=stride, act=None, norm=norm, ks=ks)
    )

class ResBlock(nn.Module):
    def __init__(self, ni, nf, stride=1, ks=3, act=act_gr, norm=None):
        super().__init__()
        self.convs = _conv_bock(ni, nf, stride, act=act, norm=norm, ks=ks)
        self.idconv = fc.noop if ni == nf else conv(ni, nf, stride=1, act=None, ks=1)
        self.pool = fc.noop if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)
        self.act = act()

    def forward(self, x):
        return self.act(self.convs(x) + self.idconv(self.pool(x)))
    