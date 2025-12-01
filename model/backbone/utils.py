import torch 
import torch.nn as nn
import torch.nn.functional as F 
 
class reluS(nn.Module): 
    def __init__(self, target):
        super().__init__()
        assert target < 1 and target > 0, f"relu problem! expected <1, >0 but got {target}"
        self.target = target

    def forward(self, X):
        return torch.clamp(X, self.target) 

class rgem(nn.Module):
    def __init__(self, init_p=5.0, size=5):
        super().__init__()
        self.size = size
        self.p = nn.Parameter(torch.tensor(init_p, dtype=torch.float32))
        self.lppool = nn.LPPool2d(2, size, stride=1) # y = (\sum(x^p))^(1/p) 
        self.pad = nn.ReflectionPad2d((size-1)//2)

    def forward(self, X): 
        size = (self.size**2) ** (1./self.p) 
        x = 0.5*self.lppool(self.pad(X/size)) + 0.5*X
        return x 

class sgem(nn.Module):
    def __init__(self, ps=10., infinity = True):
        super(sgem, self).__init__()
        self.ps = nn.Parameter(torch.tensor(10, dtype=torch.float32))
        self.infinity = infinity

    def forward(self, x):
        x = torch.stack(x,0)
        if self.infinity:
            x = F.normalize(x, p=2, dim=-1) # 3 C
            x = torch.max(x, 0)[0] 
        else:
            gamma = x.min()
            x = (x - gamma).pow(self.ps).mean(0).pow(1./self.ps) + gamma

        return x
    
class gemp(nn.Module):
    def __init__(self, p=4.6, eps = 1e-8):
        super(gemp, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x: torch.tensor):
        x = x.clamp(self.eps).pow(self.p)
        x = torch.nn.functional.adaptive_avg_pool1d(x, 1).pow(1. / (self.p) )
        return x
    