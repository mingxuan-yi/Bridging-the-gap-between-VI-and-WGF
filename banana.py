import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rosenbrock import rosenbrock
import os
from source.base import gmm, contour_plot, h

sns.set_style("white")
np.random.seed(2)
torch.manual_seed(2)
saveroot = 'banana'


mu = torch.Tensor([1.0])
a = torch.Tensor([1.0])
b = torch.ones([2, 1])

p_rosen_dist = rosenbrock.RosenbrockDistribution(mu, a, b)


def plot_contour(log_p, xlim=[-1, 2.7], ylim=[-2.5, 8], num_grid=100, method='orginal'):

    xlist = np.linspace(xlim[0], xlim[1], num_grid)
    ylist = np.linspace(ylim[0], ylim[1], num_grid)
    X, Y = np.meshgrid(xlist, ylist)
    XY = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1).astype("float32")
    Z = (log_p(torch.tensor(XY))).exp().reshape([100, 100])

    plt.figure(figsize=(4,4))
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.ylim(-4, 8)
    plt.contour(X, Y, Z, levels=10, cmap='viridis')
    filepath = os.path.join(saveroot, f"banana-{method}.png")
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    
def log_p(x):
    return -p_rosen_dist.nl_pdf(x)



def train(num_iter=3001, num_groups=5, num_sample_rep=30, method='Rkl', lr=0.01):

    q = gmm(num_groups)

    opt = optim.Adam(q.parameters(), lr=0.01)
    for i in range(3001):
        opt.zero_grad()
        loss = 0.0
        for k in range(num_groups):
            z = q.comp[k].rsample(num_sample_rep)
        
            log_ratio = -p_rosen_dist.nl_pdf(z) - q.log_prob(z)
            r = log_ratio.exp()
            loss += -q.weights()[k]*(torch.mean(h(r, method)))
        loss.backward()
        opt.step()
    plot_contour(q.log_prob, method=method)

if __name__ == '__main__':
    if not(os.path.isdir(saveroot)):
        os.mkdir(saveroot)
    plot_contour(log_p)
    train(method='Rkl')
    train(method='Fkl')
    train(method='Chi')
    train(method='Hellinger')