from utils import get_hyperparameter_combinations

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


