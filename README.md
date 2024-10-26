    """
    A normalization layer for tabular input data capable of 
    estimating the transformation functions of common 
    scikit-learn scalers in a fully differentiable 
    and numerically stable way.
    
    The normalization is applied feature-wise, however
    unlike with BatchNorm, no running statistic is learned
    (results will be the same during both train and test time)
    and the layer should only be applied once, after the input.
        
    NOTE:
    - the layer is sensitive to learning rate
    - skip this layer when applying weight decay
    - you can directly apply a dropout after this layer to
    mimic random tree-like network structures
    - extreme outliers can still hinder training, consider
    clipping your inputs as a preprocessing step still
    """