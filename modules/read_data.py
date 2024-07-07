import copy
import pandas as pd
def read_data(fname):
  df = pd.DataFrame(pd.read_spss(fname), index=None)
  df.head(n=10)
  print(f'Data Shape Raw : {df.shape}')
  data = copy.deepcopy(df)
  df = df[['sex','Age','physicalinactivity','walkforatleast10minutes','Height','Weight','Hisory_DM',
            'HTN','marital_new','BMI_cat','smoke','drink','kchat','fruit','vegetables','fat','salt','History_HTN']]
  print(f'Data Shape Final : {df.shape}')
  return df