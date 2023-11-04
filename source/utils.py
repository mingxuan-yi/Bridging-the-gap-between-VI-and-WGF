from scipy.linalg import sqrtm
import torch
from source.base import log_prob_unnormalized, log_prob


def eva_kl(mu_q, scale_q, gaussian=True):
    
    z = torch.randn([100, 2])
    x = mu_q + z @ scale_q.T
    if gaussian:
         log_ratio = log_prob_unnormalized(mu_p, scale_p, x) - log_prob(mu_q, scale_q, x)
    return -log_ratio.mean(0)

def wasserstein2(mu1, cov1, mu2, cov2):
   
    s1 = sqrtm(cov1.numpy())
    s1cov2s1 = s1 @  cov2.numpy() @ s1
    d2 = torch.norm(mu1 - mu2)**2 + torch.trace(cov1 + cov2- 2 * sqrtm(s1cov2s1))
    return torch.sqrt(d2)