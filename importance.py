import os
import pandas as pd
import matplotlib.pyplot as plt
from UTIL.colorful import *

def feature_importance(data: pd.DataFrame, col_a: str, col_b: str, test:pd.DataFrame):
    if not os.path.exists('./result'):
        os.makedirs('./result', exist_ok=True)
    
    importance_dict = {}
    

    # xgb importance matrix
    from xgboost import XGBClassifier
    tmp_model = XGBClassifier(eval_metric='auc', random_state=1, n_jobs=-1)
    tmp_model.fit(data.drop(columns=[col_b, col_a]), data[col_b])
    importance = tmp_model.feature_importances_
    importance_df = pd.DataFrame(importance, index=data.drop(columns=[col_b, col_a]).columns, columns=['importance'])
    importance_df = importance_df.sort_values(by='importance', ascending=True)
    # visualize the importance matrix
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance (xgb)')
    plt.barh(importance_df.index, importance_df['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('./result/feature_importance_xgb.png')
    plt.cla()
    # preserve top 10 importance features
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    top_importance_features = importance_df.index.tolist()
    importance_dict['xgb'] = top_importance_features


    # random forest importance matrix
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1)
    rf_model.fit(data.drop(columns=[col_b]), data[col_b])
    importance = rf_model.feature_importances_
    importance_df = pd.DataFrame(importance, index=data.drop(columns=[col_b]).columns, columns=['importance'])
    importance_df = importance_df.sort_values(by='importance', ascending=True)
    # visualize the importance matrix
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance (Random Forest)')
    plt.barh(importance_df.index, importance_df['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('./result/feature_importance_rf.png')
    plt.cla()
    rf_model = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1)
    rf_model.fit(data.drop(columns=[col_b, col_a]), data[col_b])
    importance = rf_model.feature_importances_
    importance_df = pd.DataFrame(importance, index=data.drop(columns=[col_b, col_a]).columns, columns=['importance'])
    importance_df = importance_df.sort_values(by='importance', ascending=True)
    # visualize the importance matrix
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance (Random Forest without id)')
    plt.barh(importance_df.index, importance_df['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('./result/feature_importance_rf.png')
    plt.cla()
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    top_importance_features = importance_df.index.tolist()
    importance_dict['rf'] = top_importance_features



    corr_mat = data.corr(method='pearson')
    # visualize the correlation matrix
    plt.figure(figsize=(12, 8))
    plt.title('Correlation Matrix')
    plt.imshow(corr_mat, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(corr_mat.columns)), corr_mat.columns, rotation=90)
    plt.yticks(range(len(corr_mat.index)), corr_mat.index)
    plt.tight_layout()
    plt.savefig('./result/correlation_matrix.png')
    plt.cla()

    corr_vector = corr_mat[col_b]
    # visualize the correlation vector
    plt.figure(figsize=(12, 8))
    plt.title(f'Correlation with {col_b}')
    plt.bar(corr_vector.index, corr_vector)
    plt.xlabel('Features')
    plt.ylabel('Correlation')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('./result/correlation_vector.png')
    plt.cla()

    threshold = abs(corr_mat[col_a][col_b])
    low_corr_features = [col for col in corr_mat.index if abs(corr_mat[col][col_b]) < threshold]
    print(f"Threshold for correlation with '{col_b}': {threshold}")
    print(f"Features with correlation less than {threshold}: {low_corr_features}")
    

    # auc_vs_top_n(importance_dict, data, col_a, col_b)


    return drop(importance_dict, data, test)

def drop(importance_dict, data: pd.DataFrame, test: pd.DataFrame, top_n = 10000):
    data_for_model = {}
    test_data_for_model = {}
    for model, importance_list in importance_dict.items():
        
        to_drop = []
        if len(importance_list) > top_n:
            to_drop = importance_list[top_n:]
        else:
            print红(f"Not enough features for model '{model}' to drop. There are only {len(importance_list)} features.")
        data_for_model[model] = data.drop(columns=to_drop)
        test_data_for_model[model] = test.drop(columns=to_drop)

        print(f"Dropped features for model '{model}': {to_drop}")
    
    return data_for_model, test_data_for_model

def auc_vs_top_n(importance_dict, data: pd.DataFrame, col_a, col_b: str):
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    X = data.drop(columns=[col_b, col_a])
    y = data[col_b]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=1)

    def test_model(model):
        auc_scores = []
        top_n_list = list(reversed(range(1, len(X.columns) + 1)))
        for top_n in top_n_list:
            to_drop = []
            importance_list = importance_dict['xgb']
            if len(importance_list) > top_n:
                to_drop += importance_list[top_n:]
            print(to_drop)
            X_train_subset = X_train.drop(columns=to_drop)
            X_val_subset = X_val.drop(columns=to_drop)

    
            model.fit(X_train_subset, y_train)
            y_pred = model.predict_proba(X_val_subset)[:, 1]

            auc_score = roc_auc_score(y_val, y_pred)
            auc_scores.append(auc_score)
            print(f"Top {top_n} features AUC score: {auc_score}")
        plt.figure(figsize=(12, 8))
        plt.title(f'AUC vs Top N Features {model_name}')
        # reverse the x-axis
        plt.gca().invert_xaxis()
        plt.plot(top_n_list, auc_scores, marker='o')
        plt.xlabel('Top N Features')
        plt.ylabel('AUC Score')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'./result/auc_vs_top_n_{model_name}.png')
        plt.cla()

    for model_name in importance_dict.keys():
        if model_name == 'xgb':
            model = XGBClassifier(eval_metric='auc', random_state=1, n_jobs=-1)
            test_model(model)
        if model_name == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1)
            test_model(model)