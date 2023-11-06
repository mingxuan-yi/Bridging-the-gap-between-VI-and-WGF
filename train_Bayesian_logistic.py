import torch
from torch import optim
import math
import numpy as np
from source.data_process import dataset
from source.base import vi_Model, h
import argparse
torch.manual_seed(4)
np.random.seed(2)

def get_args():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--dataset', type=str, default='heart', choices=['ionos', 'heart', 'wine', 'pima'])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--post_eval_samples', type=int, default=32)
    parser.add_argument('--elbo_samples', type=int, default=32)
    parser.add_argument('--num_epoch', type=int, default=2001)


    return parser.parse_args()


args = get_args()





loss_func = torch.nn.BCELoss(reduction='sum')
def negative_log_likelihood(y_pred, y_train):
    # return the minus log_likelihood
    return loss_func(y_pred, y_train)

def train(model, x_train, y_train, opt, num_samples_eblo, method='rep-rkl'):
    
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    
    opt.zero_grad()
    log_ratios = []
    for i in range(num_samples_eblo):
        y_pred, log_q = model.forward(x_train)
        log_ratio_per = -negative_log_likelihood(y_pred, y_train) - log_q
        log_ratios.append(log_ratio_per) 
        
    log_ratios = torch.stack(log_ratios)
    
    if method=='rep-rkl':
        loss = -log_ratios.mean() # averaging elbo
    elif method=='path-rkl':
        loss = -log_ratios.mean() # averaging elbo
        
    # adaptively adjust ratio to enhance numerical stability.
    else:
        max_c = torch.max(log_ratios).detach().clone()
        ratios = torch.exp(log_ratios - max_c)
        loss = -h(ratios, method).mean()

    loss = loss / num_samples_eblo
    loss.backward()
    opt.step()
    return loss

def get_accuracy(model, x_test, y_test):
    
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_pred, log_q = model.forward(x_test)
    label = (y_pred > 0.5).detach().numpy()
    acc = label==y_test
    return sum(acc)/len(y_test)

def eval_post_mean_std(x_test, y_test, model, samples=args.post_eval_samples):
    post_test_acc = []
    for i in range(samples):
        post_test_acc.append(get_accuracy(model, x_test, y_test))
    return  np.mean(post_test_acc), np.std(post_test_acc)

if __name__ == '__main__':
    
    if args.dataset == 'heart':
        path = 'data/heart.csv'
    elif args.dataset == 'ionos':
        path = 'data/ionosphere_data.csv'
    elif args.dataset == 'wine':
        path = 'data/winequality-red.csv'
    elif args.dataset == 'pima':
        path = 'data/pima-indians-diabetes.csv'
        
    x_train, y_train, x_test, y_test = dataset(args, path)
    input_dim = x_train.shape[1]
  
    print('Starting training: method = rep-rkl')
    model = vi_Model(input_dim, 1, detach=False)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    for i in range(args.num_epoch):
        train(model, x_train, y_train, opt, args.elbo_samples, 'rep-rkl')
    
    mean, std = eval_post_mean_std(x_test, y_test, model)
    print('Posterior accuracy mean:', np.round(mean, 3), 
          'std:', np.round(std, 3))
    print('==================================================')
    
    print('Starting training: method = path-rkl')
    model = vi_Model(input_dim, 1, detach=True)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    for i in range(args.num_epoch):
        train(model, x_train, y_train, opt, args.elbo_samples, 'path-rkl')
    
    mean, std = eval_post_mean_std(x_test, y_test, model)
    print('Posterior accuracy mean:', np.round(mean, 3), 
          'std:', np.round(std, 3))
    print('==================================================')
  
    print('Starting training: method = path-fkl')
    model = vi_Model(input_dim, 1, detach=True)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    for i in range(args.num_epoch):
        train(model, x_train, y_train, opt, args.elbo_samples, 'Fkl')
    
    mean, std = eval_post_mean_std(x_test, y_test, model)
    print('Posterior accuracy mean:', np.round(mean, 3), 
          'std:', np.round(std, 3))
    print('==================================================')
    
    print('Starting training: method = path-chi')
    model = vi_Model(input_dim, 1, detach=True)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    for i in range(args.num_epoch):
        train(model, x_train, y_train, opt, args.elbo_samples, 'Chi')
    
    mean, std = eval_post_mean_std(x_test, y_test, model)
    print('Posterior accuracy mean:', np.round(mean, 3), 
          'std:', np.round(std, 3))
    print('==================================================')
    
    print('Starting training: method = path-hellinger')
    model = vi_Model(input_dim, 1, detach=True)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    for i in range(args.num_epoch):
        train(model, x_train, y_train, opt, args.elbo_samples, 'Hellinger')
    
    mean, std = eval_post_mean_std(x_test, y_test, model)
    print('Posterior accuracy mean:', np.round(mean, 3), 
          'std:', np.round(std, 3))
    print('==================================================')
    print('Completed!')
    