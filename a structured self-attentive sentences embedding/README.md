# A pytorch implementation for paper "A Structured Self-attentive Sentence Embedding"

## Implementation Details

### model.py

#### Implementation for the main sentence embedding model and a classification model for sentiment analysis task on yelp data set.

#### Same notations are used for hyperparameters in the sentence embedding model as in the paper. The hyperparameters for the classification model are kept fixed for simplicity, since they are more task-specific. You can change the source code for other classification tasks, or just plug the sentence embedding model for any downstream applications.

#### The sentence embedding model has a output with form (out,penalty). The first term is the computed sentence embedding vector (matrix). The second term is the penalization term introduced in this paper.

 
    
