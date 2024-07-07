from sklearn.metrics import mean_squared_error
import numpy as np

def extract_cv_results(cv_obj):
    train_score = abs(cv_obj.cv_results_['mean_train_score'][cv_obj.best_index_])
    valid_score = abs(cv_obj.cv_results_['mean_test_score'][cv_obj.best_index_])
    best_score = abs(cv_obj.best_score_)
    best_params = cv_obj.best_params_
    return train_score, valid_score, best_score, best_params