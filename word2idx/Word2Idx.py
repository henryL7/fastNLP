import collections
import pickle

class Word2Idx():
    """
    Build a word index according to word frequency.
    If "min_freq" is given, then only words with a frequncy not lesser than min_freq will be kept.
    If "max_num" is given, then at most the most frequent $max_num words will be kept.
    "words" should be a list [ w_1,w_2,...,w_i,...,w_n ] where each w_i is a string representing a word.
    
    num is the size of the lookup table.
    w2i is a lookup table assigning each word an index.
    Note that index $num will be returned for any unregistered words.
    i2w is a vector which serves as an invert mapping of w2i.
    e.g. i2w[w2i["word"]] == "word"
    """
    def __init__(self):
        self.__w2i = dict()
        self.__i2w = []
        self.num = 0

    def build(self,words,min_freq=0,max_num=None):
        """build a model from words"""
        counter = collections.Counter(words)
        word_set = set(words)
        if max_num is not None:
            most_common = counter.most_common(min(len(word_set),max_num))
        else:
            most_common = counter.most_common()
        self.__w2i = dict((w[0],i) for i,w in enumerate(most_common) if w[1] >= min_freq)
        self.__i2w = [ w[0] for w in most_common if w[1] >= min_freq ]
        self.num = len(self.__i2w)

    def w2i(self,word):
        """word to index"""
        if word in self.__w2i:
            return self.__w2i[word]
        return self.num

    def i2w(self,idx):
        """index to word"""
        if idx >= self.num:
            raise Exception("out of range\n")
        return self.__i2w[idx]

    def save(self,addr):
        """save the model to a file with address "addr" """
        pickle.dump(self,open(addr,"wb"))

    def load(self,addr):
        """load a model from a file with address "addr" """
        self = pickle.load(open(addr,"rb"))



