import pandas as pd
def cat_encoder_fit_transform(encoder, cat_data_train, cat_columns):
    return pd.DataFrame(encoder.fit_transform(cat_data_train), columns=cat_columns)