from utils import get_hyperparameter_combinations, read_digits, train_test_dev_split

def test_hparam_combinations_count():
    # a test case to check that all possible combinations of parameters are indeed generated
    gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
    C_ranges = [0.1, 1, 2, 5, 10]
    h_params_combinations = get_hyperparameter_combinations(gamma_ranges, C_ranges)
    assert len(h_params_combinations) == len(gamma_ranges) * len(C_ranges)

def test_for_hparam_combinations_values():
    gamma_ranges = [0.001, 0.01]
    C_ranges = [1]
    h_params_combinations = get_hyperparameter_combinations(gamma_ranges, C_ranges)
    
    expected_param_combo_1 = {'gamma': 0.001, 'C': 1}
    expected_param_combo_2 = {'gamma': 0.01, 'C': 1}

    assert (expected_param_combo_1 in h_params_combinations) and (expected_param_combo_2 in h_params_combinations)


def  test_data_splitting():
        X, y = read_digits()

        X = X[:100,:,:]
        y = y[:100]

        test_size = .1
        dev_size = .6
        train_size = 1 - test_size - dev_size

        X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X,y, test_size=test_size, dev_size = dev_size)

        assert (len(X_train) == int(train_size * len(X)))
        assert (len(X_test) == int(test_size * len(X)))
        assert (len(X_dev) == int(dev_size * len(X)))


