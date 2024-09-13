from operator import attrgetter
from collections.abc import Mapping
import torch
import fastcore.all as fc
from copy import copy
from torcheval.metrics import MulticlassAccuracy, Mean
from fastprogress import progress_bar,master_bar
import numpy as np
import random

class with_cbs:
    def __init__(self, nm): self.nm = nm
    def __call__(self, f):
        def _f(o, *args, **kwargs):
            try:
                o.callback(f'before_{self.nm}')
                f(o, *args, **kwargs)
                o.callback(f'after_{self.nm}')
            except globals()[f'Cancel{self.nm.title()}Exception']: pass
            finally: o.callback(f'cleanup_{self.nm}')
        return _f
    
def run_cbs(cbs, method_nm, learn=None):
    for cb in sorted(cbs, key=attrgetter('order')):
        method = getattr(cb, method_nm, None)
        if method is not None: method(learn)

class CancelFitException(Exception): pass
class CancelBatchException(Exception): pass
class CancelEpochException(Exception): pass

class Callback():
    order = 0
    _fwd = 'model', 'opt', 'batch', 'epoch'

    def __getattr__(self, name):
        if name in self._fwd: return getattr(self.learn, name)
        raise AttributeError(name)

    def __setattr__(self, name, val):
        if name in self._fwd: warn(f'Setting {name} in callback. Did you mean to set `self.learn.{name}`?')
        super().__setattr__(name, val)

    @property
    def training(self): return self.model.training

# data loaders section

def inplace(f):
    def _f(b):
        f(b)
        return b
    return _f

def to_device(x, device='cpu'):
    if isinstance(x, torch.Tensor): return x.to(device)
    if isinstance(x, Mapping): return {k:v.to(device) for k,v in x.items()}
    return type(x)(to_device(o, device) for o in x)

def set_seed(seed, deterministic=False):
    torch.use_deterministic_algorithms(deterministic)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class MetricsCB(Callback):
    def __init__(self, *ms, **metrics):
        super().__init__()
        for o in ms: metrics[type(o).__name__] = o
        self.metrics = metrics
        self.all_metrics = copy(metrics)
        self.all_metrics['loss'] = self.loss = Mean()

    def _log(self, log): print(log)
    def before_fit(self, learn):
        learn.metrics = self
    def before_epoch(self, learn): [o.reset() for o in self.all_metrics.values()]

    def after_epoch(self, learn):
        log = { k: f'{v.compute():.3f}' for k,v in self.all_metrics.items() }
        log['epoch'] = learn.epoch
        log['train'] = 'train' if learn.model.training else 'eval'
        self._log(log)

    def after_batch(self, learn):
        x,y,*_ = to_device(learn.batch, device='cpu')
        for m in self.metrics.values(): m.update(to_device(learn.preds, device='cpu'), y)
        self.loss.update(to_device(learn.loss, device='cpu'), weight=len(x))


class DeviceCB(Callback):
    def __init__(self, device='cpu'): fc.store_attr()
    
    def before_fit(self, learn):
        if hasattr(learn.model, 'to'): learn.model.to(self.device)

    def before_batch(self, learn):
        learn.batch = to_device(learn.batch, device=self.device)

class ProgressCB(Callback):
    order = MetricsCB.order+1
    def __init__(self, plot=False): self.plot = plot

    def before_fit(self, learn):
        learn.epochs = self.mbar = master_bar(learn.epochs)
        self.first = True
        if hasattr(learn, 'metrics'): learn.metrics._log = self._log
        self.losses = []
        self.val_losses = []

    def _log(self, log):
        if self.first:
            self.mbar.write(list(log), table=True)
            self.first = False
        self.mbar.write(list(log.values()), table=True)

    def before_epoch(self, learn): learn.dl = progress_bar(learn.dl, leave=False, parent=self.mbar) 

    def after_epoch(self, learn):
        if not learn.training:
            if self.plot and hasattr(learn, 'metrics'):
                self.val_losses.append(learn.loss.item())
                self.mbar.update_graph([
                    [fc.L.range(self.losses), self.losses],
                    [fc.L.range(learn.epoch+1).map(lambda x: (x+1)*len(learn.dls.train)), self.val_losses]
                ])

    def after_batch(self, learn): 
        learn.dl.comment = f'{learn.loss:.3f}'
        if self.plot and hasattr(learn, 'metrics') and learn.training:
            self.losses.append(learn.loss.item())
            if self.val_losses:
                self.mbar.update_graph([
                    [fc.L.range(self.losses), self.losses],
                    [fc.L.range(learn.epoch).map(lambda x: (x+1)*len(learn.dls.train)), self.val_losses]
                ])

def append_stats(hook, mod, inp, outp):
    if not hasattr(hook, 'stats'): hook.stats = ([], [], [])
    acts = outp.detach()
    hook.stats[0].append(acts.mean())
    hook.stats[1].append(acts.std())
    hook.stats[2].append(acts.abs().histc(40, 0, 10))

def get_hist(h):
    return torch.stack(h.stats[2]).t().float().log1p()

def get_min(h):
    h1 = torch.stack(h.stats[2]).t().float()
    return h1[0]/h1.sum(0)