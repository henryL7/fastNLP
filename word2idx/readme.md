# Word2Idx
## Build a word index according to word frequency. 
### Parameters
#### "words" should be a list [ w_1,w_2,...,w_i,...,w_n ] where each w_i is a string representing a word.
#### If "min_freq" is given, then only words with a frequncy not lesser than min_freq will be kept.
#### If "max_num" is given, then at most the most frequent $max_num words will be kept.
### Member objects
#### num is the size of the lookup table.
#### __w2i is a lookup table assigning each word an index. Note that index $num will be returned for any unregistered words.
#### __i2w is a vector which serves as an invert mapping of w2i. e.g. i2w[w2i["word"]] == "word"
### Functions
#### build(): build a word index from words
#### w2i(): convert word to index
#### i2w(): convert index to word
#### save(): save the model to a file with address "addr"
#### load(): load a model from a file with address "addr"
### Usage
    import Word2Idx
    words = ["hello","world"]
    w = Word2Idx.Word2Idx()
    w.build(words)
    print(w.w2i("hello"))
    w.save("word2idx.pkl")
    w.load("word2idx.pkl")
    print(w.i2w(0))

