import numpy as np
from scipy.stats import skewnorm
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from source.base import gmm, h
import os 
np.random.seed(101)
torch.manual_seed(101)
saveroot = '1d_gmm'

def log_prob(mu, scale, x):
    cov = scale * scale
    dim = 1
    a = -0.5 * (1 * np.log(2 * np.pi) + torch.log(1e-8+cov))
    b =  -0.5*(x - mu)**2 / cov
    return a + b
    
def log_p_3mode(x):
    
    mu_1=torch.tensor(-1.0)
    scale_1 = torch.tensor(0.5)
    mu_2 = torch.tensor(0.8)
    scale_2 = torch.tensor(0.5)
    mu_3 = torch.tensor(3.0)
    scale_3 = torch.tensor(0.8)
    
    a = 0.4 * torch.exp(log_prob(mu_1, scale_1, x))
    b = 0.3 * torch.exp(log_prob(mu_2, scale_2, x))
    c = 0.3 * torch.exp(log_prob(mu_3, scale_3, x))
                          
    prob = a + b + c
    
    return torch.log(prob+1e-8).view(-1)


def train(number_iter=1000, lr=0.01, 
          method='Rkl', num_groups=5, 
          num_sample_rep=10, target_logp=log_p_3mode, normalizing_constant=2.0):
    
    q = gmm(num_groups, dim=1)

    opt = torch.optim.Adam(q.parameters(), lr=0.01)
    for i in range(number_iter):
        opt.zero_grad()
        loss = 0.0
        for k in range(num_groups):
            z = q.comp[k].rsample(num_sample_rep)
        
            log_ratio = target_logp(z) - q.log_prob(z) + np.log(normalizing_constant)
            r = log_ratio.exp()
            loss += -q.weights()[k]*(torch.mean(h(r, method)))
        loss.backward()
        opt.step()
   
   
    xx = torch.linspace(-7, 10, 1000)
    filepath = os.path.join(saveroot, f"1dgmm-{method}-{num_groups}.png")
    plt.plot(xx, target_logp(xx).exp(),
       'r-', lw=5, alpha=0.6, label=r'$p(x)$')
    approximated_pdf = q.log_prob(xx.view(-1, 1)).detach().exp()
    plt.plot(xx, approximated_pdf,
       'b-', lw=2, alpha=1.0,  label=r'$q(x;\theta)$')
    plt.legend(loc='best', frameon=False, fontsize=17)
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0.1)
    plt.close() 
    
    
if __name__ == '__main__':

    if not(os.path.isdir(saveroot)):
        os.mkdir(saveroot)
        
        
    train(number_iter=2001, num_groups=1, method='Rkl')
    train(number_iter=2001, num_groups=2, method='Rkl')
    train(number_iter=2001, num_groups=3, method='Rkl')
    train(number_iter=2001, num_groups=4, method='Rkl')
    
    train(number_iter=2001, num_groups=1, method='Fkl')
    train(number_iter=2001, num_groups=2, method='Fkl')
    train(number_iter=2001, num_groups=3, method='Fkl')
    train(number_iter=2001, num_groups=4, method='Fkl')
    
    train(number_iter=2001, num_groups=1, method='Chi')
    train(number_iter=2001, num_groups=2, method='Chi')
    train(number_iter=2001, num_groups=3, method='Chi')
    train(number_iter=2001, num_groups=4, method='Chi')
    
    train(number_iter=2001, num_groups=1, method='Hellinger')
    train(number_iter=2001, num_groups=2, method='Hellinger')
    train(number_iter=2001, num_groups=3, method='Hellinger')
    train(number_iter=2001, num_groups=4, method='Hellinger')