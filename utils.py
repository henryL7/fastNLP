import pickle
import torch
import numpy as np
from torch.autograd import Variable

# get a train set containing m samples and a test set containing n samples
def get_sets(m,n):
    samples=pickle.load(open("shuffle_tuples.pkl","rb"))
    if m+n > len(samples):
        print("asking for too many tuples\n")
        return
    train_samples = samples[:m]
    test_samples = samples[m:m+n]
    return train_samples,test_samples

# transform float type list to pytorch variable
def float_wrapper(x,requires_grad=True,using_cuda=True):
    if using_cuda==True:
        return Variable(torch.FloatTensor(x).cuda(),requires_grad=requires_grad)
    else:
        return Variable(torch.FloatTensor(x),requires_grad=requires_grad)

# transform long type list to pytorch variable
def long_wrapper(x,requires_grad=True,using_cuda=True):
    if using_cuda==True:
        return Variable(torch.LongTensor(x).cuda(),requires_grad=requires_grad)
    else:
        return Variable(torch.LongTensor(x),requires_grad=requires_grad)

# transform word to vector according to word_dict, zero padding to given size
def transfrom_words(x,padding_size,word_dict,word_size):
    #result=[ word_dict[x_i].tolist() for x_i in x]
    result = []
    padding=[0 for i in range(word_size)]
    for x_i in x:
        try:
            w = word_dict[x_i]
        except KeyError:
            w = np.random.randn(word_size)
        result.append(w.tolist())
    padding_num=padding_size-len(x)
    paddings=[padding for i in range(padding_num)]
    result+=paddings
    return result







  