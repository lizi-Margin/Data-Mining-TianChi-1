"""
    by shc 2025.5.3
"""
import os,shutil
import numpy as np
from UTIL.colorful import *

def eval(result: dict, y_test, logdir='result'):
    y_proba = result['y_proba']
    y_pred = result['y_pred']

    os.makedirs(logdir, exist_ok=True)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print_green("评估结果:")
    print(f"准确率: {accuracy}")
    print(f"精确率: {precision}")
    print(f"召回率: {recall}")
    print(f"F1分数: {f1}")
    print(f"AUC: {auc}")

    # ROC curve
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f"{logdir}/ROC_curve.png")
    plt.cla()


    # precision-recall curve
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower left")
    plt.savefig(f"{logdir}/Precision_Recall_curve.png")
    plt.cla()

    # confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print_yellow("混淆矩阵:")
    print(cm)

    # 绘制混淆矩阵
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f"{logdir}/Confusion_Matrix.png")
    plt.cla()

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'precision': precision,
        'recall': recall,
        'pic_roc': f'{logdir}/ROC_curve.png',
        'pic_pr': f'{logdir}/Precision_Recall_curve.png',
        'pic_cm': f'{logdir}/Confusion_Matrix.png'
    }
    # dump metrics
    import json
    with open(f"{logdir}/metrics.json", 'w') as f:
        dump_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                dump_metrics[key] = value.tolist()
            else:
                dump_metrics[key] = value
        json.dump(dump_metrics, f, indent=4)

    result.update(metrics)
    return result

    