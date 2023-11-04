import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rosenbrock import rosenbrock
import os
from source.base import gmm, h
import argparse
sns.set_style("white")
np.random.seed(2)
torch.manual_seed(2)

def get_args():
    parser = argparse.ArgumentParser()
   
    
    parser.add_argument('--saveroot', type=str, default='.')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_components', type=int, default=5)
    parser.add_argument('--num_sample', type=int, default=30)
    parser.add_argument('--num_iter', type=int, default=3001)


    return parser.parse_args()


args = get_args()
saveroot = args.saveroot


mu = torch.Tensor([1.0])
a = torch.Tensor([1.0])
b = torch.ones([2, 1])

p_rosen_dist = rosenbrock.RosenbrockDistribution(mu, a, b)

def plot_contour(log_p, ax, xlim=[-1, 2.7], ylim=[-2.8, 8], num_grid=100, name=r'Target $p(x)$'):

    xlist = np.linspace(xlim[0], xlim[1], num_grid)
    ylist = np.linspace(ylim[0], ylim[1], num_grid)
    X, Y = np.meshgrid(xlist, ylist)
    XY = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1).astype("float32")
    Z = (log_p(torch.tensor(XY))).exp().reshape([100, 100])

    
    ax.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    ax.set_ylim(-4, 8)
    ax.contour(X, Y, Z, levels=10, cmap='viridis')
    ax.set_title(name, y=-0.15, size=15)

    
def log_p(x):
    return -p_rosen_dist.nl_pdf(x)

def train(num_iter=args.num_iter, num_groups=args.num_components, num_sample_rep=args.num_sample, method='Rkl', lr=0.01):
    print('Starting training GMM under', method)
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
    print('Completed !')
    return q

if __name__ == '__main__':
    if not(os.path.isdir(saveroot)):
        os.mkdir(saveroot)
    q_rkl = train(method='Rkl')
    q_fkl = train(method='Fkl')
    q_chi = train(method='Chi')
    q_hel = train(method='Hellinger')
    
    fig, axs = plt.subplots(1, 5, figsize=(15, 3.4))
    plot_contour(log_p, axs[0])
    plot_contour(q_rkl.log_prob, axs[1], name='Reverse KL')
    plot_contour(q_fkl.log_prob, axs[2], name='Forward KL')
    plot_contour(q_chi.log_prob, axs[3], name=r'$\chi^2$')
    plot_contour(q_hel.log_prob, axs[4], name='Hellinger')
    fig.tight_layout()
    filepath = os.path.join(saveroot, "banana_contour.png")
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0.1)
    plt.close()