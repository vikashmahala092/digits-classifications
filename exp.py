"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause



# Import datasets, classifiers and performance metrics
from utils import preprocess_data, split_data, train_model, read_digits, train_test_dev_split, predict_and_eval, tune_hparams, get_hyperparameter_combinations
from itertools import product


# 1. Get the dataset
X, y = read_digits()

# 3. Data splitting -- to create train, test and dev sets
X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=0.3, dev_size=0.2)

# 4. Data Preprocessing
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)
X_dev = preprocess_data(X_dev)


# Creating dictionary of all list combinations for hyperparameters
gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
C_ranges = [0.1, 1, 2, 5, 10]

list_of_dicts = get_hyperparameter_combinations(gamma_ranges, C_ranges)

# Hyper parameter tunning
optimal_gamma, optimal_C, best_model, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, list_of_dicts)

#print("Optimal parameters gamma: ", optimal_gamma, "C: ", optimal_C, "best_accuracy: ", best_accuracy)


# Varying the test_size and dev_size and getting the best hyperparameters for the combinations
test_size_list = [0.1,0.2,0.3]
dev_size_list = [0.1,0.2,0.3]
size_combinations = list(product(test_size_list, dev_size_list))
size_list_of_dicts = [{ 'test_size': combo[0], 'dev_size': combo[1]} for combo in size_combinations]

for comb in size_list_of_dicts:
    X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=comb['test_size'], dev_size=comb['dev_size'])
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)
    X_dev = preprocess_data(X_dev)
    optimal_gamma, optimal_C, best_model, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, list_of_dicts)
    train_acc = predict_and_eval(best_model, X_train, y_train)
    dev_acc = predict_and_eval(best_model, X_dev, y_dev)
    test_acc = predict_and_eval(best_model, X_test, y_test)
    train_size = 1 -(comb['test_size'] + comb['dev_size'])

    # added quiz - 1 changes
    print("Number of Samples in train dataset = ", len(X_train))
    print("Number of Samples in test dataset = ", len(X_test))
    print("Number of Samples in dev dataset = ", len(X_dev))

    print("Size of image in dataset = ", X_train[0].shape)


    print("test_size= " , comb['test_size'], "dev_size= ", comb['dev_size'], "train_size= ", train_size, "train_acc= ", train_acc, "dev_acc= ", dev_acc, "test_acc= ", test_acc, "optimal_gamma= ", optimal_gamma, "optimal_C= ", optimal_C)

# 6. Getting model Predictions and Evaluating on test set
#test_acc = predict_and_eval(best_model, X_test, y_test)
#print("Test accuracy:" , test_acc)

