import xport
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from modules.read_data import read_data
from modules.split_input_output import split_input_output
from modules.split_train_test import split_train_test
from modules.num_cat_splitter import num_cat_splitter
from modules.cat_encoder import cat_encoder
from modules.concat_data import concat_data
from modules.fit_scaler import fit_scaler
from modules.transform_scaler import transform_scaler
from modules.preprocess_data import preprocess_data

def main():
    data = read_data('Hypertension_dataset.sav')

    X, y = split_input_output(data=data,
                            target_col='HTN')
    
    # Split the data
    # First, split the train & not train
    X_train, X_not_train, y_train, y_not_train = split_train_test(X, y, 0.2)

    # Then, split the valid & test
    X_valid, X_test, y_valid, y_test = split_train_test(X_not_train, y_not_train, 0.5)
    
    cat_columns = X_train.select_dtypes(exclude=[np.number]).columns
    num_columns = X_train.select_dtypes(include=[np.number]).columns
    X_train_num, X_train_cat = num_cat_splitter(X_train, num_columns, cat_columns)
    
    print(f'Data X train num  : ', X_train_num.shape)
    print(f'Data X train cat  : ', X_train_cat.shape)
    print(f'Data y train  : ', y_train.shape)

    X_train_cat_encoded = cat_encoder(X_train_cat)
    X_train_concat = concat_data(X_train_num, X_train_cat_encoded)
    # Fit the scaler
    scaler = fit_scaler(X_train_concat)

    # Transform the scaler
    X_train_clean = transform_scaler(X_train_concat, scaler)
    X_valid_clean = preprocess_data(X_valid, num_columns, cat_columns, scaler)
    X_test_clean = preprocess_data(X_test, num_columns, cat_columns, scaler)
    
if __name__ == "__main__":
    main()