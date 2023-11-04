import torch
from torch import optim
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from source.data_process import dataset
from source.base import vi_Model

torch.manual_seed(4)
np.random.seed(2)


path = 'data/heart.csv'
x_train, y_train, x_test, y_test = dataset(path)
input_dim = x_train.shape[1]


loss_func = nn.BCELoss(reduction='sum')
def get_loss(y_pred, y_train):
    
    return loss_func(y_pred, y_train)

def train(model, x_train, y_train, opt, num_samples_eblo):
    
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    
    opt.zero_grad()
    #y_pred, log_q = model.forward(x_train)
    
    
    # minus elbo = -log_likeli + logq
    loss = 0
    for i in range(num_samples_eblo):
        y_pred, log_q = model.forward(x_train)
        loss += get_loss(y_pred, y_train) + log_q
        
    loss = loss / num_samples_eblo
    #loss = get_loss(y_pred, y_train) 
    loss.backward()
    opt.step()
    return loss

def get_accuracy(model, x_test, y_test):
    
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_pred, log_q = model.forward(x_test)
    label = (y_pred > 0.5).detach().numpy()
    acc = label==y_test
    return sum(acc)/len(y_test)


if __name__ == '__main__':
    num_epochs = 2001

    np.set_printoptions(precision=3)

    model = vi_Model(input_dim, 1, True)
    opt = optim.Adam(model.parameters(), lr=0.01)

    for i in range(num_epochs):
        train(model, x_train, y_train, opt, 32)
            #if i % 500 == 0:
                #print(i, los
    
    
    post_test_acc = []
    for i in range(32):
        post_test_acc.append(get_accuracy(model, x_test, y_test))
    print('Posterior accuracy mean:', np.mean(post_test_acc), 
          'Posterior accuracy std:', np.std(post_test_acc))