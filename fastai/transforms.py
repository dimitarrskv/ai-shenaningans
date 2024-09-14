from utils import inplace
import torchvision.transforms.functional as TF

@inplace
def transformi(b):
    x = 'image'
    b[x] = [TF.to_tensor(o) for o in b[x]]