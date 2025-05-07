"""
    * 原始示范程序：0.9363
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


# 读取数据
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

from numericOobject import obj_num_cols, encode_categorical_features, value_encoding_dict, box_cut, onehot_encoding_list
train_obj_col, train_num_col = obj_num_cols(train_data)
test_obj_col, test_num_col = obj_num_cols(test_data)
print("train_obj_col:", train_obj_col)
print("train_num_col:", train_num_col)
print("test_obj_col:", test_obj_col)
print("test_num_col:", test_num_col)

train_data = encode_categorical_features(train_data, value_encoding_dict, onehot_encoding_list)
test_data = encode_categorical_features(test_data, value_encoding_dict, onehot_encoding_list)

o, _ = obj_num_cols(train_data); assert len(o) == 0
o, _ = obj_num_cols(test_data); assert len(o) == 0
_n = 20; print('-'*_n, 'class features 处理完成', '-'*_n)

train_data = box_cut(train_data)
test_data = box_cut(test_data)


from nan import is_nan
is_nan(train_data, 'train_data')
is_nan(test_data, 'test_data')
_n = 20; print('-'*_n, 'Nan缺失值 处理完成', '-'*_n)

from importance import feature_importance
train_data_for_model, test_data_for_model = feature_importance(train_data, 'id', 'subscribe', test=test_data)

train_data, test_data = train_data_for_model['xgb'], test_data_for_model['xgb']

_n = 20; print('-'*_n, '低相关性特征已经剔除: ', '-'*_n)
# print(train_data.head())

# print(test_data.columns)
# # to_drop = ['education', 'housing']
# # to_drop = ['housing']
# # to_drop = ['education']
# for col_to_drop in to_drop:
#     for col in train_data.columns:
#         if col.startswith(col_to_drop):
#             train_data.drop(col, axis=1, inplace=True)
#             test_data.drop(col, axis=1, inplace=True)


from up_sample import up_sample, up_sample_ratio_vs_auc, up_sample_smote

# up_sample_ratio_vs_auc(train_data, 'subscribe', target_pos_ratio=0.8)


train_data = train_data.drop(['id'], axis=1)

data_shuffled = train_data.sample(frac=1, random_state=1)

train_size = 0.9
train_data = data_shuffled.iloc[:int(len(data_shuffled)*train_size)]
train_data = up_sample_smote(train_data, 'subscribe', target_pos_ratio=0.0)
val_data = data_shuffled.iloc[int(len(data_shuffled)*train_size):]
X_train = train_data.drop(columns=['subscribe'])
y_train = train_data['subscribe']
X_val = val_data.drop(columns=['subscribe'])
y_val = val_data['subscribe']
X_pred = test_data.drop(['id'], axis=1)



# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=1)



# from model import train_v41 as train
from model import train_v12 as train
from eval import eval
model = train(X_train, y_train, X_val, y_val)


eval_result = eval({
    'y_proba': model.predict_proba(X_val)[:, 1],
    'y_pred': model.predict(X_val)
}, y_val)


from res import output_res
output_res(model, X_pred, test_data['id'])