import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import word2vec
import time
import model
import utils

"""
training procedure

If model_dict is given (a file address), it will continue training on the given model.
Otherwise, it would train a new model from scratch.
Worddict should be given by the user, which should be a pretrained word2vec model using gensim.
If using_cuda is true, the training would be conducted on GPU.
Learning_rate and momentum is for SGD optimizer.
coef is the coefficent between the cross-entropy loss and the penalization term.
interval is the frequncy of reporting.
word_size is the word-vector dimension.

the result will be saved with a form "model_dict_+current time", which could be used for further training

"""
def train(worddict,model_dict=None,using_cuda=True,learning_rate=0.06,\
    momentum=0.3,batch_size=32,epochs=5,coef=1.0,interval=10):
    
    if using_cuda == True:
        net = model.SentimentNet().cuda()
    else:
        net = model.SentimentNet()
    if model_dict != None:
        net.load_state_dict(torch.load(model_dict))
    word_dict = word2vec.Word2Vec.load(worddict)
    optimizer = optim.SGD(net.parameters(),lr=learning_rate,momentum=momentum)
    criterion = nn.CrossEntropyLoss(size_average=True)
    train_set,_ = utils.get_sets(500000,2000)
    batch_nums = int(len(train_set)/batch_size)

    #statistics

    loss_count=0
    prepare_time=0
    run_time=0
    count=0
    
    for epoch in range(epochs):
        for i in range(batch_nums):
            t1 = time.time()
            x_raw = []
            max_size = 0
            for j in range(i*batch_size,(i+1)*batch_size):
                x_raw.append(train_set[j][0])
                if len(train_set[j][0])>max_size:
                    max_size=len(train_set[j][0])
            y = [ train_set[j][1] for j in range(i*batch_size,(i+1)*batch_size) ]
            X = [ utils.transfrom_words(words,max_size,word_dict,model.WORD_SIZE) for words in x_raw ]
            t2 = time.time()
            y_pred,y_penl = net(utils.float_wrapper(x=X,using_cuda=using_cuda,requires_grad=True))
            loss = criterion(y_pred, utils.long_wrapper(x=y,using_cuda=using_cuda,requires_grad=False)-1)\
            +torch.sum(y_penl)/batch_size*coef
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(net.embedding.blstm.parameters(),0.5)
            optimizer.step()
            t3 = time.time()
            loss_count += torch.sum(y_penl).data[0]
            prepare_time += (t2-t1)
            run_time += (t3-t2)
            p, idx = torch.max(y_pred, dim=1)
            idx = idx.data
            count += torch.sum(torch.eq(idx.cpu(), torch.LongTensor(y)-1))
            if i%interval==0:
                print(i)      
                print("loss count:"+str(loss_count/(interval*batch_size)))
                print("acuracy:"+str(count/(interval*batch_size)))
                print("penalty:"+str(torch.sum(y_penl).data[0]/batch_size))
                print("prepare time:"+str(prepare_time))
                print("run time:"+str(run_time))
                prepare_time=0
                run_time=0
                loss_count=0
                count=0
        torch.save(net.state_dict(),"model_dict_"+str(time.time()))

if __name__ == "__main__":
    train(worddict="worddict",using_cuda=torch.cuda.is_available())


        







        





