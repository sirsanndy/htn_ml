import pandas as pd
def concat_data(num_data_train, cat_data_train):
    joined_data = pd.concat([num_data_train.reset_index(drop=True), cat_data_train.reset_index(drop=True)], axis=1)
    print(f'Numerical data shape : {num_data_train.shape}')
    print(f'Categorical data shape : {cat_data_train.shape}')
    print(f'Concat data shape : {joined_data.shape}')
    return joined_data