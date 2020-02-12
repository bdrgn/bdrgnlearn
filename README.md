# bdrgnlearn

### Classic machine learning algorithms implemented in plain Python / Numpy. 

| [Decision Trees](bdrgnlearn/tree.py) | [Linear models](bdrgnlearn/linear_model.py) | 
| ------------- | ------------- | 
| ![](demo_gifs/decision_tree_demo.gif) |  ![](demo_gifs/linreg_sgd_demo.gif) | 

# Performance
| Algorithm | Dataset | sklearn performance | bdrgnlearn performance |
| ------------- | ------------- | ------------- | ------------- |
| [Random Forest Classifier](bdrgnlearn/ensemble.py) | Breast Cancer dataset (sklearn)|0.95 accuracy|0.95 accuracy|
| [Logistic Regression](bdrgnlearn/linear_model.py) | Breast Cancer dataset (sklearn)|0.96 accuracy|0.94 accuracy|
| [KNeighborsClassifier](bdrgnlearn/neighbors.py) | Breast Cancer dataset (sklearn) |0.93 accuracy|0.93 accuracy|
| [Linear Regression](bdrgnlearn/linear_model.py) | Boston House Prices dataset (sklearn) |0.68 R2|0.66 R2|
| [Random Forest Regressor](bdrgnlearn/ensemble.py) | Boston House Prices dataset (sklearn)|0.78 R2|0.73 R2|
| [KNeighborsRegressor](bdrgnlearn/ensemble.py) | Boston House Prices dataset (sklearn) |0.63 R2|0.63 R2|
