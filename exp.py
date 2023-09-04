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
from utils import preprocess_data, split_data, train_model, read_digits, train_test_dev_split, predict_and_eval
gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
C_ranges = [0.1, 1, 2, 5, 10]

# 1. Get the dataset
X, y = read_digits()

# 3. Data splitting -- to create train, test and dev sets
X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=0.3)

# 4. Data Preprocessing
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)
X_dev = preprocess_data(X_dev)


# Hyper parameter tunning
best_acc_so_far = -1
best_model = None
for cur_gamma in gamma_ranges:
    for cur_C in C_ranges:
        #print("Running for gamma={} C={}".format(cur_gamma, cur_C))
        # 5. Model training
        cur_model = train_model(X_train, y_train, {'gamma': cur_gamma, 'C': cur_C}, model_type="svm")
        cur_accuracy = predict_and_eval(cur_model, X_dev, y_dev, 'Dev')
        if cur_accuracy > best_acc_so_far:
            print("New best accuracy: ", cur_accuracy)
            best_acc_so_far = cur_accuracy
            optimal_gamma = cur_gamma
            optimal_C = cur_C
            best_model = cur_model

print("Optimal parameters gamma: ", optimal_gamma, "C: ", optimal_C)


# 6. Getting model Predictions and Evaluating on test set
test_acc = predict_and_eval(best_model, X_test, y_test, 'Test')
print("Test accuracy:" , test_acc)