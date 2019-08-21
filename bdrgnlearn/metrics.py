import numpy as np

def confusion_matrix(y_true, y_pred):
    # Store all unique classes labels in an array
    classes = np.unique(y_true)

    # Save the total number of classes
    n_classes = len(classes)

    # Store the number of observations
    n_obs = len(y_true)

    # Create an array of true y values repeated times the number of classes squared
    y_true_2d = np.tile(y_true, n_classes ** 2).reshape(n_classes ** 2, n_obs)

    # Create an array of predicted y values repeated times the number of classes squared
    y_pred_2d = np.tile(y_pred, n_classes ** 2).reshape(n_classes ** 2, n_obs)

    # Create an array of classes to compare with true y values
    classes_2d_y_true = np.repeat(classes, n_obs * n_classes).reshape(n_classes ** 2, n_obs)

    # Create an array of classes to compare with predicted y values
    classes_2d_y_pred = np.array([np.repeat(classes, n_obs).reshape(n_classes, n_obs)] * 3) \
        .reshape(n_classes ** 2, n_obs)

    # Calculate confusion matrix values
    conf_matr = np.sum((y_true_2d == classes_2d_y_true) & (y_pred_2d == classes_2d_y_pred), axis=1)

    # Return the correctly shaped matrix
    return conf_matr.reshape(n_classes, n_classes)


def precision_score(y_true, y_pred):
    # Store all unique classes labels in an array
    classes = np.unique(y_true)

    # Save the total number of classes
    n_classes = len(classes)

    # Store the number of observations
    n_obs = len(y_true)

    # Create a binary array of true y values
    y_true_binary = np.tile(y_true, n_classes).reshape(n_classes, n_obs) == classes.reshape(n_classes, 1)

    # Create a binary array of predicted y values
    y_pred_binary = np.tile(y_pred, n_classes).reshape(n_classes, n_obs) == classes.reshape(n_classes, 1)

    # Calculate the true positives sum
    tp = np.sum(y_true_binary & y_pred_binary, axis=1)

    # Calculate true positives + false positives sum = positive predictions sum
    positive_predictions = np.sum(y_pred_binary, axis=1)

    # Calculate precision scores for each class
    precision_scores = tp / positive_predictions

    # Calculate a macro mean of the precision scores
    final_precision_score = np.mean(precision_scores)

    # Return the final averaged value
    return final_precision_score


def recall_score(y_true, y_pred):
    # Store all unique classes labels in an array
    classes = np.unique(y_true)

    # Save the total number of classes
    n_classes = len(classes)

    # Store the number of observations
    n_obs = len(y_true)

    # Create a binary array of true y values
    y_true_binary = np.tile(y_true, n_classes).reshape(n_classes, n_obs) == classes.reshape(n_classes, 1)

    # Create a binary array of predicted y values
    y_pred_binary = np.tile(y_pred, n_classes).reshape(n_classes, n_obs) == classes.reshape(n_classes, 1)

    # Calculate the true positives sum
    tp = np.sum(y_true_binary & y_pred_binary, axis=1)

    # Calculate true positives and false negatives sum
    ap = np.sum(y_true_binary, axis=1)

    # Calculate a recall scores for each class
    recall_scores = tp / ap

    # Calculate a final averaged recall score
    final_recall_score = np.mean(recall_scores)

    # Return the final value
    return final_recall_score

# Define accuracy score function
def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)
