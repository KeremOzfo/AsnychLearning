import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np


def MNIST_data(mini_batch, N_w, N_s):

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=mini_batch, shuffle=False, num_workers=2)

    samp_per_worker = int(N_s / N_w)
    excess_samps = N_s - N_w * int(N_s / N_w)
    seq = [samp_per_worker for i in range(N_w)]

    # giving the extra samples to workers
    excess_w = np.random.choice(N_w, excess_samps, replace=False)

    for i in range(excess_w.shape[0]):
        seq[i] += 1

    trainsets = torch.utils.data.random_split(trainset, seq)

    trainloaders = [torch.utils.data.DataLoader(trainsets[i], batch_size=mini_batch, shuffle=True, num_workers=2)
                    for i in range(N_w)]

    return trainloaders, testloader


def FMNIST_data(mini_batch, N_w, N_s):

    transform_train = transforms.Compose([
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)

    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=mini_batch, shuffle=False, num_workers=2)

    samp_per_worker = int(N_s / N_w)
    excess_samps = N_s - N_w * int(N_s / N_w)
    seq = [samp_per_worker for i in range(N_w)]

    # giving the extra samples to workers
    excess_w = np.random.choice(N_w, excess_samps, replace=False)

    for i in range(excess_w.shape[0]):
        seq[i] += 1

    trainsets = torch.utils.data.random_split(trainset, seq)

    trainloaders = [torch.utils.data.DataLoader(trainsets[i], batch_size=mini_batch, shuffle=True, num_workers=2)
                    for i in range(N_w)]

    return trainloaders, testloader


def CIFAR_data(mini_batch, N_w, N_s):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    samp_per_worker = int(N_s / N_w)
    excess_samps = N_s - N_w * int(N_s / N_w)
    seq = [samp_per_worker for i in range(N_w)]

    # giving the extra samples to workers
    excess_w = np.random.choice(N_w, excess_samps, replace=False)

    for i in range(excess_w.shape[0]):
        seq[i] += 1

    trainsets = torch.utils.data.random_split(trainset, seq)

    trainloaders = [torch.utils.data.DataLoader(trainsets[i], batch_size=mini_batch, shuffle=True, num_workers=2)
                    for i in range(N_w)]

    return trainloaders, testloader