import numpy as np
import pandas as pd
def output_res(model, X_pred, ID):
    y_pred = model.predict(X_pred)
    r = ['no', 'yes']
    res_j = [r[y] for y in y_pred]
    res_i = ID
    res = []
    for k in range(len(res_i)):
        ans = [res_i[k], res_j[k]]
        res.append(ans)

    res_df = pd.DataFrame(res,  columns=['id', 'subscribe'])
    _n = 20; print('-'*_n, 'Answer generated: ', '-'*_n)
    print(res_df['subscribe'].value_counts())
    res_df.to_csv('res.csv', index=False)