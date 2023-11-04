import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import math
from scipy import linalg
import matplotlib as mpl

def plt_potential_func(log_prob, plt, num_grid=300, xs=[-2, 5], ys=[-2, 4.5], cmap=None):
 
    xlist = torch.linspace(xs[0], xs[1], num_grid)
    ylist = torch.linspace(ys[0], ys[1], num_grid)
    xx, yy = torch.meshgrid(xlist, ylist)
    z = torch.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    z = torch.Tensor(z)
    u = log_prob(z).cpu().numpy()
    p = np.exp(u).reshape(num_grid, num_grid)

    plt.pcolormesh(xx, yy, p, cmap=cmap, shading='auto', vmax=1.5)

def get_ecp(center, cov, ax, s=0.4, color='red', alpha=1.0, style='dotted'):

    v_f, w_f = linalg.eigh(cov)
    v_f = s * np.sqrt(2.0) * np.sqrt(v_f)
    u_f = w_f[0] / linalg.norm(w_f[0])
    angle = np.arctan(u_f[1] / u_f[0])
    angle = 180.0 * angle / np.pi  # convert to degrees
    ell = mpl.patches.Ellipse(center, v_f[0], v_f[1], angle=180.0 + angle, color=color, fc='None',lw=3, 
                          linestyle=style)
    ell.set_alpha(alpha)
    ax.add_patch(ell)
    
    
def plot_evolution(mus_rep, mus_path, mus_ode, covs_rep, covs_path, covs_ode, ax1, var_scale, mid=30):
    
    # Plot the mean trajectories
    ax1.plot(mus_rep[:, 0], mus_rep[:, 1], linestyle = '-', label='BBVI-rep', alpha=1.0,color='orange', linewidth=3)
    ax1.plot(mus_path[:, 0], mus_path[:, 1], linestyle = '-', label='BBVI-path(stl)', alpha=0.7, color='red', linewidth=3)
    ax1.plot(mus_ode[:, 0], mus_ode[:, 1], linestyle = '-', label='ODE evolution', alpha=0.5, color='green', linewidth=3)
    
    
    # Plot the variance ellipsoids at q0
    ax1.scatter(x=mus_rep[0][0], y=mus_rep[0][1], s=70, alpha=0.5, color='orange', zorder=2)
    get_ecp(mus_rep[0], covs_rep[0], ax1, s=var_scale, color='orange', style= (0, (3, 7)), alpha=1.0)
    ax1.scatter(x=mus_path[0][0], y=mus_path[0][1], s=70, alpha=0.5, color='red', zorder=2)
    get_ecp(mus_path[0],  covs_path[0], ax1, s=var_scale, color='red', style=(0, (4, 7)), alpha=0.5)
    ax1.scatter(x=mus_ode[0][0], y=mus_ode[0][1], s=70, alpha=0.5, color='green', zorder=2)
    get_ecp(mus_ode[0], covs_ode[0], ax1, s=var_scale, color='green', style=(0, (5, 7)))


    # Plot the variance ellipsoids at q30
    step = mid
    ax1.scatter(x=mus_rep[step][0], y=mus_rep[step][1], s=70, alpha=0.5, color='orange', zorder=2)
    get_ecp(mus_rep[step], covs_rep[step], ax1, s=var_scale, color='orange', style= (0, (3, 6)), alpha=1.0)
    ax1.scatter(x=mus_path[step][0], y=mus_path[step][1], s=70, alpha=0.5, color='red', zorder=2)
    get_ecp(mus_path[step], covs_path[step], ax1, s=var_scale, color='red', style=(0, (4, 6)), alpha=0.5)
    ax1.scatter(x=mus_ode[step][0], y=mus_ode[step][1], s=70, alpha=0.5, color='green', zorder=2)
    get_ecp(mus_ode[step], covs_ode[step], ax1, s=var_scale, color='green',  style= (0, (5, 6)))


    # Plot the variance ellipsoids at final q
    ax1.scatter(x=mus_rep[-1][0], y=mus_rep[-1][1], s=70, alpha=0.5, color='orange', zorder=2)
    get_ecp(mus_rep[-1], covs_rep[-1], ax1, s=var_scale, color='orange', style= (0, (3, 4)), alpha=1.0)
    ax1.scatter(x=mus_path[-1][0], y=mus_path[-1][1], s=70, alpha=0.5, color='red', zorder=2)
    get_ecp(mus_path[-1], covs_path[-1], ax1, s=var_scale, color='red', style=(0, (3, 7)), alpha=0.5)
    ax1.scatter(x=mus_ode[-1][0], y=mus_ode[-1][1], s=70, alpha=0.5, color='green', zorder=2)
    get_ecp(mus_ode[-1], covs_ode[-1],ax1, s=var_scale, color='green', style= (0, (5, 6)))