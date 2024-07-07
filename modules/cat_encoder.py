from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder as ohe
from modules.cat_encoder_fit_transform import cat_encoder_fit_transform
import pandas as pd

def cat_encoder(data_cat):
    data_cat_bmi_enc = data_cat[['BMI_cat']]
    data_cat_marital_enc = data_cat[['marital_new']]

    # Transform
    data_cat_bmi_encoded = cat_encoder_fit_transform(ohe(sparse_output=False), data_cat_bmi_enc, ['Normal', 'Under weight', 'Obese', 'Over weight'])
    data_cat_marital_encoded = cat_encoder_fit_transform(ohe(sparse_output=False), data_cat_marital_enc, ['Single', 'Married', 'Divorced/Widowed'])
    # Encode loan_grade with LabelEncoder
    data_cat_dm_encoded = cat_encoder_fit_transform(LabelEncoder(), data_cat['Hisory_DM'], cat_columns=['Hisory_DM'])
    data_cat_physic_encoded = cat_encoder_fit_transform(LabelEncoder(), data_cat['physicalinactivity'], cat_columns=['physicalinactivity'])
    data_cat_sex_encoded = cat_encoder_fit_transform(LabelEncoder(), data_cat['sex'], cat_columns=['sex'])
    data_cat_walk_encoded = cat_encoder_fit_transform(LabelEncoder(), data_cat['walkforatleast10minutes'], cat_columns=['walkforatleast10minutes'])
    # Encode cb_person_default_on_file with LabelEncoder
    data_cat_smoke_encoded = cat_encoder_fit_transform(LabelEncoder(), data_cat['smoke'], cat_columns=['smoke'])
    data_cat_drink_encoded = cat_encoder_fit_transform(LabelEncoder(), data_cat['drink'], cat_columns=['drink'])
    data_cat_kchat_encoded = cat_encoder_fit_transform(LabelEncoder(), data_cat['kchat'], cat_columns=['kchat'])
    data_cat_vegetables_encoded = cat_encoder_fit_transform(LabelEncoder(), data_cat['fruit'], cat_columns=['fruit'])
    data_cat_fat_encoded = cat_encoder_fit_transform(LabelEncoder(), data_cat['fat'], cat_columns=['fat'])
    data_cat_salt_encoded = cat_encoder_fit_transform(LabelEncoder(), data_cat['salt'], cat_columns=['salt'])
    data_cat_hist_htn_encoded = cat_encoder_fit_transform(LabelEncoder(), data_cat['History_HTN'], cat_columns=['History_HTN'])


    # # Concatenate encoded dataframes
    data_cat_encoded = pd.concat([data_cat_dm_encoded,data_cat_physic_encoded,data_cat_sex_encoded,data_cat_walk_encoded,
    data_cat_smoke_encoded,data_cat_drink_encoded,data_cat_kchat_encoded,data_cat_vegetables_encoded,data_cat_fat_encoded,
    data_cat_salt_encoded,data_cat_hist_htn_encoded, data_cat_bmi_encoded, data_cat_marital_encoded], axis=1)
    
    return data_cat_encoded