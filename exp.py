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

# 1. Get the dataset
X, y = read_digits()

# 3. Data splitting -- to create train, test and dev sets
X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=0.3)

# 4. Data Preprocessing
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)
X_dev = preprocess_data(X_dev)

# 5. Model training
model = train_model(X_train, y_train, {'gamma': 0.001}, model_type="svm")


# 6. Getting model Predictions and Evaluating on dev set
predict_and_eval(model, X_dev, y_dev, 'Dev')

# 7. Getting model Predictions and Evaluating on test set
predict_and_eval(model, X_test, y_test, 'Test')