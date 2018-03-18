# A pytorch implementation for paper "A Structured Self-attentive Sentence Embedding"

## Implementation Details

### model.py

#### Implementation for the main sentence embedding model and a classification model for sentiment analysis task on yelp data set.

#### Same notations are used for hyperparameters in the sentence embedding model as in the paper. The hyperparameters for the classification model are kept fixed for simplicity, since they are more task-specific. You can change the source code for other classification tasks, or just plug the sentence embedding model in your own model for any downstream applications.

#### The sentence embedding model has a output with form (out,penalty). The first term is the computed sentence embedding vector (matrix). The second term is the penalization term introduced in this paper.

#### Some hyperparameters for the model achitecture are set lower than the paper's suggestion. It still has a reasonable performance since the classification task may not require too much information.

### train.py

#### Training procedure. The coefficient for the penalization term is set to 1 during the training, following the paper. The learning rate is set to 0.06 and the momentum is set to 0.2 during the first three epochs. Then, the learning rate is set to 0.02 and the momentum is set to 0.6 for the next five epochs. 500K review-star pairs are used for training.

### test.py

#### Test for the classification accuracy. The model trained eight epochs reachs a 0.688 accuracy on a 50K test set.

## Notes

### Dataset

#### The dataset can be download at https://www.yelp.com/dataset/challenge. Each comment is croped to a maximum length of 500. After that, the dataset is randomly shuffled.

### Pretraining 

#### The word2vec model in gensim is used to get the word vectors. You can get the gensim package here https://radimrehurek.com/gensim/. The word vectors are kept fixed during the training. Any unregistered words will be converted to a zero vector. Note that the whole dataset is used for the word2vec pretraining, so there are actually no unregietered words in test set. Therefore, it is unclear whether unregiesterd words will harm the classification accuracy. Since the zero padding is heavily used for the forming of each batch during the training and testing, my assumption is that the zero vector should be good for unregistered words, which needs to be testified later.  

### More information

#### The experiment is conducted on Ubuntu 16.04, python 3.5, pytorch 0.3.0. You may want to use GPU for an obvious acceleration. If the default setting is used, 2GB RAM should be sufficient for the training. ( I'm pretty sure about this since I only get a 2GB GPU on my laptop :) ).




    
