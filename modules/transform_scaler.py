import pandas as pd
def transform_scaler(data_imputed, std):
    return pd.DataFrame(std.transform(data_imputed), columns=data_imputed.columns, index=data_imputed.index)