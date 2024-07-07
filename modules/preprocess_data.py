from modules.num_cat_splitter import num_cat_splitter
from modules.cat_encoder import cat_encoder
from modules.concat_data import concat_data
from modules.transform_scaler import transform_scaler

def preprocess_data(data_train, num_cols, cat_cols, scaler):
    data_num, data_cat = num_cat_splitter(data_train, num_cols, cat_cols)
    
    print("Train numerical shape after  : ", data_num.shape)
    print("Train categorical shape after  : ", data_cat.shape)

    # Concatenate encoded dataframes
    cat_concat_encoded = cat_encoder(data_cat)
    
    print("Train cat encoded concatenated shape : ", cat_concat_encoded.shape)
    
    data_concatenated = concat_data(data_num, cat_concat_encoded)
    print('')
    print('Original data shape:', data_concatenated.shape)

    # Transform the scaler
    clean_data = transform_scaler(data_concatenated, scaler)
    print('Mapped data shape :', data_concatenated.shape,'\n')
    return clean_data