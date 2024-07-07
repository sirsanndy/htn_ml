from sklearn.model_selection import train_test_split
def split_train_test(X, y, test_size_int):
  X_train, X_not_train, y_train, y_not_train = train_test_split(X, y, test_size=test_size_int, 
                                                                random_state=123, stratify=y)
  print(f'X train shape : {X_train.shape}')
  print(f'y train shape : {y_train.shape}')
  print(f'X test shape : {X_not_train.shape}')
  print(f'y test shape : {y_not_train.shape}')
  print()
  return X_train, X_not_train, y_train, y_not_train