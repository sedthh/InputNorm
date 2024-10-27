# InputNorm
InputNorm is a normalization layer capable of learning estimations of common [scikit-learn scalers](https://scikit-learn.org/dev/api/sklearn.preprocessing.html) (such as the Yeo-Johnson / Box-Cox [\[1\]](https://stats.stackexchange.com/questions/603496/interpreting-the-lambdas-of-yeo-johnson-transformation)[\[2\]](https://feaz-book.com/numeric-yeojohnson)[\[3\]](https://www.statisticshowto.com/probability-and-statistics/normal-distributions/box-cox-transformation/) based [PowerTransformer](https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.PowerTransformer.html)) in a fully differentiable and numerically stable way.  

To mimic the preprocessing steps of most data preprocessing pipelines where scaling is applied before missing value imputation, InputNorm also allows accepts NaNs in its input (it will pass the missing values in its output accordingly).

The normalization is applied feature-wise, however unlike BatchNorm, no running statistic is learned in the process (so the same results will be returned during both training and inference). Another difference between InputNorm and other normalization layers is that this layer is only meant to be aplied **once** right after the input.
        
**NOTE:**
- The layer is sensitive to learning rate settings at the moment.
- Skip InputNorm's parameters when applying weight decay!
- You can directly apply a dropout layer after this layer to mimic random tree-like network structures.
- Although the layer does most of the necessarry steps for data preprocessing, extreme outliers can still hinder training! Consider clipping your inputs as a preprocessing step!