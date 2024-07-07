from sklearn.preprocessing import StandardScaler
def fit_scaler(data_imputed):
    std = StandardScaler()
    return std.fit(data_imputed)