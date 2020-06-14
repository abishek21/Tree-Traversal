# Tree-Traversal
This model uses Sequence-to-Sequence Attention Mechanism to Traverse the Tree

# To train the model
### Run python attention_train.py --traindata path/to/traindata --loadvocab yes --model model_name.h5
### While training for first time pass --loadvocab no, This will let you to create the vocab file and--loadvocab yes loads the vocab already built.

## After training the all the models(inorder/postorder/preorder)
### To Deploy the model in flask, run python app.py

## To do inference
### run python inference.py

#### I would recommend to read this blog post on attention model https://wanasit.github.io/attention-based-sequence-to-sequence-in-keras.html