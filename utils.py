from sklearn.model_selection import train_test_split
from sklearn import svm, datasets, metrics
from itertools import product
# We will put all utils here

def read_digits():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    return X, y

def preprocess_data(data):
    # flatten the images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

# Split data into 50% train and 50% test subsets
def split_data(x, y, test_size, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, shuffle=False, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

# train the model of choice with the model parameters
def train_model(x, y, model_params, model_type="svm"):
    if model_type == "svm":
        # Create a classifier: a support vector classifier
        clf = svm.SVC;
    model = clf(**model_params)
    # train the model
    model.fit(x, y)
    return model


# Split data into train, test and dev subsets
def train_test_dev_split(x, y, test_size, dev_size, random_state=1):
    X_train, X_temp, y_train, y_temp = train_test_split(
        x, y, test_size=test_size, shuffle=False, random_state=random_state
    )
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp, y_temp, test_size=dev_size, shuffle=False, random_state=random_state
    )

    return X_train, X_test, X_dev, y_train, y_test, y_dev


def predict_and_eval(model, X_test, y_test):
    # prediction
    predicted = model.predict(X_test)
    return metrics.accuracy_score(y_test, predicted)


def tune_hparams(X_train, y_train, X_dev, y_dev, list_of_all_param_combination):
    best_acc_so_far = -1
    best_model = None
    for comb in list_of_all_param_combination:
        #print("Running for gamma={} C={}".format(comb['gamma'], comb['C']))
        # 5. Model training
        cur_model = train_model(X_train, y_train, {'gamma': comb['gamma'], 'C': comb['C']}, model_type="svm")
        cur_accuracy = predict_and_eval(cur_model, X_dev, y_dev)
        if cur_accuracy > best_acc_so_far:
            #print("New best accuracy: ", cur_accuracy)
            best_acc_so_far = cur_accuracy
            optimal_gamma = comb['gamma']
            optimal_C = comb['C']
            best_model = cur_model
    return optimal_gamma, optimal_C, best_model, best_acc_so_far


def get_hyperparameter_combinations(gamma_ranges, C_ranges):
        combinations = list(product(gamma_ranges, C_ranges))
        return [{ 'gamma': combo[0], 'C': combo[1]} for combo in combinations]