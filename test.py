import utils
import model
from gensim.models import word2vec
import torch
import time

"""
model_dict: model for testing(a file address)
worddict should be given by the user, which should be a pretrained word2vec model using gensim.
If using_cuda is true, the training would be conducted on GPU.

test for classification accuracy
"""

def test(worddict,model_dict,using_cuda=True):
    net = model.SentimentNet().cuda()
    net.load_state_dict(torch.load(model_dict))
    net=net.eval()
    _,test_set=utils.get_sets(500000,50000)
    batch_size=32
    batch_nums=int(len(test_set)/batch_size)
    word_dict=word2vec.Word2Vec.load(worddict)

    count=0
    for i in range(batch_nums):
        t1=time.time()
        x_raw = []
        max_size=0
        for j in range(i*batch_size,(i+1)*batch_size):
            x_raw.append(test_set[j][0])
            if len(test_set[j][0])>max_size:
                max_size=len(test_set[j][0])
        y = [ test_set[j][1] for j in range(i*batch_size,(i+1)*batch_size) ]
        X=[ utils.transfrom_words(words,max_size,word_dict,model.WORD_SIZE) for words in x_raw ]
        y_pred,y_penl=net(utils.float_wrapper(X,requires_grad=False,using_cuda=using_cuda))
        p, idx = torch.max(y_pred, dim=1)
        idx = idx.data
        count += torch.sum(torch.eq(idx.cpu(), torch.LongTensor(y)-1))
        t2=time.time()
        if i%10==0:
            print("time:"+str(t2-t1))
            print("iters:"+str(i))
    print("accuracy:"+str(count/(batch_nums*batch_size)))

if __name__ == "__main__":
    test(worddict="worddict",model_dict="model_dict_1",using_cuda=torch.cuda.is_available())

