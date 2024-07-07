def split_input_output(data, target_col):
  output_data = data[target_col]
  input_data = data.drop(target_col, axis=1)
  print(f'X shape : {input_data.shape}')
  print(f'y shape : {output_data.shape}')
  return input_data, output_data