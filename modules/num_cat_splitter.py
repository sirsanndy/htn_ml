def num_cat_splitter(data, num_columns, cat_columns):
  data_train_num = data[num_columns]
  print(f'Data num shape: {data_train_num.shape}')
  data_train_cat = data[cat_columns]
  print(f'Data cat shape: {data_train_cat.shape}')
  return data_train_num, data_train_cat