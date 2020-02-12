# bdrgnlearn

### Classic machine learning algorithms implemented in plain Python / Numpy. 

# Performance
| Algorithm | Dataset | sklearn performance | bdrgnlearn performance |
| ------------- | ------------- | ------------- | ------------- | 
| [Random Forest Classifier](bdrgnlearn/ensemble.py) | Breast Cancer dataset (sklearn) | 0.95 accuracy | 0.95 accuracy |
| [Logistic Regression](bdrgnlearn/linear_model.py) | Breast Cancer dataset (sklearn) | 0.96 accuracy | 0.94 |
| [KNeighborsClassifier](bdrgnlearn/neighbors.py) | Breast Cancer dataset (sklearn) | 0.93 accuracy | 0.93 accuracy |
| [Linear Regression](bdrgnlearn/linear_model.py) | Boston House Prices dataset (sklearn) | 0.68 R2 | 0.66 R2 |
| [Random Forest Regressor](bdrgnlearn/ensemble.py) | Boston House Prices dataset (sklearn) | 0.78 R2 | 0.73 R2 |




# Contents

| Algorithm | Visualization | 
| ------------- | ------------- | 
| [Random Forests](bdrgnlearn/ensemble.py) | ![](demo_gifs/rf_demo.gif) |
| [Decision Trees](bdrgnlearn/tree.py) | ![](demo_gifs/decision_tree_demo.gif) | 
| [Linear models](bdrgnlearn/linear_model.py)  | ![](demo_gifs/linreg_sgd_demo.gif) | 
