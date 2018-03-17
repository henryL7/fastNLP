import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

"""
define the basic net structure
hyper-parameters 
using the same notation from the paper
word_size
lstm_hidden_size
d_a
r 
"""

SCALE_FACTOR=0.1
def penalization(A):
    I = Variable(torch.eye(A.size()[1]).cuda(),requires_grad=False)
    M = torch.matmul(A,torch.transpose(A,1,2)) - I
    M = M.view(M.size()[0],-1)
    return torch.sum(M**2,dim=1)

class SentenceNet(nn.Module):
    def __init__(self,word_size,lstm_hidden_size,d_a,r):
        super(SentenceNet,self).__init__()
        # bidirectional layers
        self.blstm = nn.LSTM(word_size,lstm_hidden_size,1,True,True,0.5,True)
        self.W_s1 = nn.Parameter(torch.randn(d_a,lstm_hidden_size*2)*SCALE_FACTOR,requires_grad=True)
        self.W_s2 = nn.Parameter(torch.randn(r,d_a)*SCALE_FACTOR,requires_grad=True)
        self.softmax = nn.Softmax(dim=2)
        self.tanh = nn.Tanh()
        
    def forward(self,inputs):
        H,_ = self.blstm(inputs)
        inter = self.tanh(torch.matmul(self.W_s1,torch.transpose(H,1,2)))
        A = self.softmax(torch.matmul(self.W_s2,inter))
        out = torch.matmul(A,H)
        penalty = penalization(A)
        return out,penalty

"""
 net for yelp sentiment analysis
 these parameters are kept fixed since they are for a certain task
"""

WORD_SIZE = 100
HIDDEN_SIZE = 300
D_A = 350
R = 20
MLP_HIDDEN = 2000 
CLASSES_NUM = 5 

class SentimentNet(nn.Module):
    def __init__(self):
        super(SentimentNet,self).__init__()
        self.embedding = SentenceNet(WORD_SIZE,HIDDEN_SIZE,D_A,R)
        self.L1 = nn.Linear(R*HIDDEN_SIZE*2,MLP_HIDDEN)
        self.L2 = nn.Linear(MLP_HIDDEN,CLASSES_NUM)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,inputs):
        out,penalty = self.embedding(inputs)
        out = out.view(out.size()[0],-1)
        out = self.L2(F.relu(self.L1(out)))
        out = self.softmax(out)
        return out,penalty





