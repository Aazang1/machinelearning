import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 中文标签
CLASS_NAMES_CN = ['山鸢尾', '杂色鸢尾', '维吉尼亚鸢尾']


def evaluate_model(y_true, y_pred, average='macro'):
    """评估模型性能"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES_CN, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    return accuracy, precision, recall, f1, report_df


def main():
    """主函数：SVM鸢尾花分类实验"""

    print("SVM鸢尾花分类实验")
    print("=" * 50)

    # 1. 加载数据
    print("\n1. 加载数据")
    iris = load_iris()
    X = iris.data
    y = iris.target

    feature_names_cn = ['花萼长度(cm)', '花萼宽度(cm)', '花瓣长度(cm)', '花瓣宽度(cm)']

    print(f"数据形状: {X.shape}")
    print(f"类别分布: {np.bincount(y)}")

    # 2. 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 五折交叉验证
    print("\n2. 五折交叉验证")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)

        accuracy, precision, recall, f1, _ = evaluate_model(y_test, y_pred)

        results.append({
            'fold': fold + 1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_support_vectors': len(svm.support_vectors_)
        })

        print(f"折{fold + 1}: 准确度={accuracy:.4f}, 支持向量数={len(svm.support_vectors_)}")

    # 4. 结果分析
    print("\n3. 平均性能")
    results_df = pd.DataFrame(results)
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names_cn = ['准确度', '精确率', '召回率', 'F1分数']

    for metric, name in zip(metrics, metric_names_cn):
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        print(f"{name}: {mean_val:.4f} (±{std_val:.4f})")

    # 5. 不同核函数比较
    print("\n4. 不同核函数比较")
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    kernel_names_cn = ['线性核', '多项式核', 'RBF核', 'Sigmoid核']

    # 使用第一折数据
    train_idx, test_idx = list(kf.split(X_scaled))[0]
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    for kernel, name in zip(kernels, kernel_names_cn):
        if kernel == 'poly':
            svm = SVC(kernel=kernel, degree=3, C=1.0, random_state=42)
        else:
            svm = SVC(kernel=kernel, C=1.0, random_state=42)

        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"{name}: 准确度={accuracy:.4f}, 支持向量数={len(svm.support_vectors_)}")

    # 6. 正则化参数C的影响
    print("\n5. 正则化参数C的影响")
    C_values = [0.01, 0.1, 1, 10, 100]

    for C in C_values:
        svm = SVC(C=C, kernel='rbf', gamma='scale', random_state=42)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"C={C}: 准确度={accuracy:.4f}, 支持向量数={len(svm.support_vectors_)}")

    # 7. 特征重要性（使用线性核）
    print("\n6. 特征重要性分析")
    svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
    svm_linear.fit(X_scaled, y)

    if hasattr(svm_linear, 'coef_'):
        coef_abs_mean = np.mean(np.abs(svm_linear.coef_), axis=0)

        feature_importance = pd.DataFrame({
            '特征': feature_names_cn,
            '权重绝对值': coef_abs_mean
        }).sort_values('权重绝对值', ascending=False)

        print("\n特征重要性:")
        print(feature_importance.to_string(index=False))

    print("\n" + "=" * 50)
    print("实验完成!")


if __name__ == "__main__":
    main()