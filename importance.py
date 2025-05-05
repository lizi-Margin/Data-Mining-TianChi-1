import os
import pandas as pd
import matplotlib.pyplot as plt

def feature_importance(data: pd.DataFrame, col_a: str, col_b: str, test=None):
    if not os.path.exists('./result'):
        os.makedirs('./result', exist_ok=True)

    # xgb importance matrix
    from xgboost import XGBClassifier
    tmp_model = XGBClassifier(eval_metric='auc', random_state=1, n_jobs=-1)
    tmp_model.fit(data.drop(columns=[col_b]), data[col_b])
    importance = tmp_model.feature_importances_
    importance_df = pd.DataFrame(importance, index=data.drop(columns=[col_b]).columns, columns=['importance'])
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
    
    # data = data.drop(columns=low_corr_features)
    # if test is not None: test = test.drop(columns=low_corr_features)
    
    return data