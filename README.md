# bdrgnlearn

### Classic machine learning algorithms implemented in plain Python / Numpy. 

| [Decision Trees](bdrgnlearn/tree.py) | [Linear models](bdrgnlearn/linear_model.py) | 
| ------------- | ------------- | 
| ![](demo_gifs/decision_tree_demo.gif) |  ![](demo_gifs/linreg_sgd_demo.gif) | 

# Classification performance
#### * accuracy measured on Breast Cancer dataset (sklearn)

| Algorithm | sklearn performance | bdrgnlearn performance |
| ------------- | ------------- | ------------- |
| [Random Forest Classifier](bdrgnlearn/ensemble.py) |0.95 |0.95 |
| [Logistic Regression](bdrgnlearn/linear_model.py) |0.96 |0.94 |
| [KNeighborsClassifier](bdrgnlearn/neighbors.py) |0.93 |0.93 |



| [Linear Regression](bdrgnlearn/linear_model.py) |0.68 R2|0.66 R2|
| [Random Forest Regressor](bdrgnlearn/ensemble.py) |0.78 R2|0.73 R2|
| [KNeighborsRegressor](bdrgnlearn/ensemble.py) |0.63 R2|0.63 R2|
