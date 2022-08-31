import numpy as np
import torch
from scipy.stats import spearmanr,chi2_contingency
import pandas as pd
import random


def layer_norm(x,n_dim=1):
    return torch.nn.functional.layer_norm(x,x.shape[-n_dim:])

def chi2_test(a,b):
    df=pd.get_dummies(a).groupby(b).sum()
    return chi2_contingency(df)[:3]

def swish(x):
    return x*torch.sigmoid(x)

class Dfc(torch.nn.Module):
    def __init__(self,n,dims):
        super(Dfc,self).__init__()
        self.fc=torch.nn.Linear(n,dims)
    def forward(self,x):
        #x:b
        x=torch.nn.functional.one_hot(x.type(torch.int64),self.fc.in_features).type(torch.float32)
        return self.fc(x)

class Cfc(torch.nn.Module):
    def __init__(self,dims):
        super(Cfc,self).__init__()
        self.fc=torch.nn.Linear(1,dims)
    def forward(self,x):
        #x:b
        x=torch.unsqueeze(x,-1)
        return self.fc(x)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True