# bdrgnlearn

### Classic machine learning algorithms implemented in plain Python / Numpy. 

| [Decision Trees](bdrgnlearn/tree.py), [Random Forests](bdrgnlearn/ensemble.py) | [Linear models](bdrgnlearn/linear_model.py) | 
| ------------- | ------------- | 
| ![](demo_gifs/decision_tree_regressor_demo_2.gif) |  ![](demo_gifs/linear_regression_demo_2.gif) |

### Accuracy on Breast Cancer dataset
| Algorithm | sklearn | __bdrgnlearn__ |
| ------------- | ------------- | ------------- |
| [Random Forest Classifier](bdrgnlearn/ensemble.py) |0.95 | __0.95__ |
| [Logistic Regression](bdrgnlearn/linear_model.py) |0.96 |__0.94__ |
| [KNeighborsClassifier](bdrgnlearn/neighbors.py) |0.93 |__0.93__ |

### R2 on Boston House Prices dataset
| Algorithm | sklearn | __bdrgnlearn__ |
| ------------- | ------------- | ------------- |
| [Random Forest Regressor](bdrgnlearn/ensemble.py) |0.78|__0.73__ |
| [Linear Regression](bdrgnlearn/linear_model.py) |0.68|__0.66__ |
| [KNeighborsRegressor](bdrgnlearn/ensemble.py) |0.63 |__0.63__ |
