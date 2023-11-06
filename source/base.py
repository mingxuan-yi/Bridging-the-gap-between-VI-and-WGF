import torch
import torch.nn as nn

import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class q_theta_full(torch.nn.Module):
    def __init__(self, mu, scale):
        super(q_theta_full, self).__init__()
        self.mu = mu
        self.dim = len(mu)
       
        self.scale = scale

    def rsample(self, size):
        samples = self.mu + torch.randn([size, self.dim]) @ self.scale.T
        
        return samples

    def sample(self, size):
        samples = self.rsample(size)
        return samples.detach()
    
    def cov(self):
        return self.scale @ self.scale.T
    
    def log_prob(self, x):
        
        a = -0.5 * (self.dim * np.log(2 * np.pi) + torch.log(torch.det(self.cov().detach())))
        b =  -0.5*torch.sum((x - self.mu.detach()) @ torch.inverse(self.cov().detach()) * (x - self.mu.detach()), axis=1)
        return a + b
    

class gmm(torch.nn.Module):
    def __init__(self, num_comps=10, dim=2):
        super(gmm, self).__init__()
        
        self.mus = torch.nn.Parameter(torch.randn([num_comps, dim]))
        self.dim = dim
       
        self.scales =  torch.nn.Parameter(torch.eye(dim).repeat(num_comps, 1, 1))
        self.comp = [q_theta_full(self.mus[i], self.scales[i]) for i in range(num_comps)]
        self.weights_ = torch.nn.Parameter(torch.randn(num_comps))
    
    def log_prob(self, x):
        
        total_prob = 0.0
        for i in range(len(self.mus)):
            total_prob += self.weights()[i].detach()*torch.exp(self.comp[i].log_prob(x))
        return torch.log(total_prob)
    
    def weights(self):
        return torch.nn.functional.softmax(self.weights_, dim=0)


class vi_layer(nn.Module):
    def __init__(self, input_dim, output_dim, detach):
        super(vi_layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.weight_mean = nn.Parameter(torch.zeros(self.input_dim, self.output_dim))
        self.bias_mean = nn.Parameter(torch.ones(self.output_dim))
                            
        self.weight_rho = nn.Parameter(torch.zeros(self.input_dim, self.output_dim))
        self.bias_rho = nn.Parameter(torch.ones(self.output_dim))
        
        self.detach = detach
        
    def std(self, v):
        return F.softplus(v)
    
    def log_prob_q(self, z, mu, rho):
        
        if self.detach:
            output = -0.5 * torch.log(2 * torch.tensor(np.pi)) - \
                torch.log(self.std(rho.detach())) - (z - mu.detach())**2 / (2*self.std(rho.detach())**2)
        else:
            output = -0.5 * torch.log(2 * torch.tensor(np.pi)) - \
                torch.log(self.std(rho)) - (z - mu)**2 / (2*self.std(rho)**2)
        return torch.sum(output)  
    
    def forward(self, x):
        weight = self.weight_mean + self.std(self.weight_rho) * torch.randn_like(self.weight_mean)
        bias = self.bias_mean + self.std(self.bias_rho) * torch.randn_like(self.bias_mean)                               
        log_prob_w = self.log_prob_q(weight, self.weight_mean, self.weight_rho)
        log_prob_b = self.log_prob_q(bias, self.bias_mean, self.bias_rho)

        return torch.mm(x, weight) + bias, log_prob_w + log_prob_b
                                     
class vi_Model(nn.Module):
    def __init__(self, input_dim, output_dim, detach=False):
        super(vi_Model, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.layer1 = vi_layer(input_dim, output_dim, detach)
        

    def forward(self, x):
        
        x = x.view(-1, self.input_dim)
        y_, log_p1 = self.layer1(x)
        y = torch.sigmoid(y_)
        
        return y, log_p1
                                 
    
def contour_plot(ax, logprob, xlim, ylim, color, num_grid=100, Z_log=False, levels=10):
    
    xlist = np.linspace(xlim[0], xlim[1], num_grid)
    ylist = np.linspace(ylim[0], ylim[1], num_grid)
    X, Y = np.meshgrid(xlist, ylist)
    XY = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1).astype("float32")
    Z = logprob(torch.tensor(XY)).detach().view(num_grid, num_grid)
    if not Z_log:
        Z = torch.exp(Z)
    
    return ax.contour(X, Y, Z, levels=levels, cmap='viridis')

def log_prob(mu, scale, x):
    cov = scale@scale.T
    dim = x.shape[1]
    a = -0.5 * (dim * np.log(2 * np.pi) + torch.log(1e-8+torch.det(cov)))
    b =  -0.5*torch.sum((x - mu) @ torch.inverse(cov) * (x - mu), axis=1)
    return a + b

def log_prob_unnormalized(mu, scale, x):
    cov = scale@scale.T
    dim = x.shape[1]
    b =  -0.5*torch.sum((x - mu) @ torch.inverse(cov) * (x - mu), axis=1)
    return b

# define f divergence as f(r)
def f(r, method='Rkl'):
    if method=='Rkl':
        return -torch.log(r+1e-8)
    elif method=='Fkl':
        return r*torch.log(r+1e-8)
    elif method=='Chi':
        return (r-1)**2
    elif method=='Hellinger':
        return (torch.sqrt(r+1e-8)-1)**2
    

# define cost function h(logr)
def h(r, method='Rkl'):
    if method=='Rkl':
        return torch.log(r+1e-8) -1 
    elif method=='Fkl':
        return r
    elif method=='Chi':
        return r**2 - 1
    elif method=='Hellinger':
        return (torch.sqrt(r+1e-8)-1)