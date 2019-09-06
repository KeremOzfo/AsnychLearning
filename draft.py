import torchvision
import torchvision.transforms as transforms
import torch
import time

import torch.nn as nn
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm

import nn_classes
import data_loader
import ps_functions
import SGD_custom

# select gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

# number of workers
N_w = 20
# number of training samples
# Cifar10  50,000
# Fashin MNIST 60,000
N_s = 50000

batch = 64

trainloaders, testloader = data_loader.CIFAR_data(batch, N_w, N_s)



w_index = 0
results = np.empty([1, 300])
res_ind = 0
nets = [nn_classes.ResNet18().to(device) for n in range(N_w)]

ps_model = nn_classes.ResNet18().to(device)
avg_model = nn_classes.ResNet18().to(device)
prev_model = nn_classes.ResNet18().to(device)
bcast_model = nn_classes.ResNet18().to(device)

lr = 1e-1
momentum = 0.9
weight_decay = 1e-4
alpha = 0.45

criterions = [nn.CrossEntropyLoss() for n in range(N_w)]
optimizers = [SGD_custom.define_optimizer(nets[n], lr, momentum, weight_decay) for n in range(N_w)]
avg_Optimizer = SGD_custom.define_optimizer(avg_model,lr,momentum, weight_decay)
bcast_Optimizer = SGD_custom.define_optimizer(bcast_model,lr,momentum, weight_decay)

# initilize all weights equally

[ps_functions.synch_weight(nets[i], ps_model) for i in range(N_w)]
ps_functions.synch_weight(ps_model, avg_model)
ps_functions.synch_weight(avg_model, bcast_model)

runs = int(30000)
for r in tqdm(range(runs)):
    # index of the worker doing local SGD
    w_index = w_index % N_w
    ts=time.time()
    for worker in range(N_w):
        for data in trainloaders[worker]:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizers[worker].zero_grad()
            preds = nets[worker](inputs)
            loss = criterions[worker](preds, labels)
            loss.backward()
            optimizers[worker].step()
            break
    tf1 =time.time()
    #corrected momentum
    ps_functions.model_Bcast(bcast_Optimizer,optimizers[w_index],r,alpha)
    tf2 = time.time()
    # assaign bcast_model to prev model

    # w_index sends its model to other workers
    # # other workers upon receiving the model take the average
    for n in range(N_w):
        if n != w_index:
            ps_functions.average_model(nets[n], nets[w_index])

    # averaging the momentums
    for n in range(N_w):
        if n != w_index:
            ps_functions.average_momentum(optimizers[n], bcast_Optimizer)
    tf3 = time.time()
    print('training of workers: ', tf1-ts,'modelCorrection: ', tf2-tf1, 'avgring: ', tf3-tf2)
    if (r % 100) == 0 and r != 0:
        ## reset of extraModel
        ps_functions.initialize_zero(avg_model)  # model
        ps_functions.momentum_zero(avg_Optimizer)

        ## take average
        for worker in range(N_w):
            ps_functions.weight_accumulate(nets[worker],avg_model,N_w) # model
            ps_functions.momentum_accumulate(avg_Optimizer,optimizers[worker],N_w)
        ##assign all worker models
        for worker in range(N_w):
            ps_functions.modelavg(nets[worker], avg_model) # model
            ps_functions.momentum_Avg(avg_Optimizer,optimizers[worker]) # momentum
    if r % 100 == 0:
        ps_functions.initialize_zero(ps_model)
        for n in range(N_w):
            ps_functions.weight_accumulate(nets[n], ps_model, N_w)
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = ps_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        results[0][res_ind] = 100 * correct / total
        res_ind += 1
    if r == 15000:
        for n in range(N_w):
            ps_functions.lr_change(0.025,optimizers[n])
    if r == 22500:
        for n in range(N_w):
            ps_functions.lr_change(0.0025, optimizers[n])

    # moving to next worker
    w_index += 1
f = open('newasynchResults' + '.txt', 'ab')
np.savetxt(f, (results), fmt='%.5f', encoding='latin1')
f.close()







