## General problems of classifying -- overfitting
Too many features might cause the model to predict the result too accurate which won't generalize the future prediction very well.
This kind of problems exists in both linear regression and logistic regression classification.
### Solution:
* Reduce the number of features:
    * manually select which features to keep
    * use a model selection algorithms like PCA
* Regularization
    * keep all the features but reduce the magnitude of parameters θj.
    * Regularization works well when we have a lot of slightly useful features.
