import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')


def load_iris_from_local():
    """
    (1) 从本地读取iris数据集
    注意：需要先下载iris数据集到本地
    """
    try:
        # 假设iris数据集的本地路径，您需要根据实际文件路径修改
        # 可以从UCI机器学习仓库下载：https://archive.ics.uci.edu/ml/datasets/iris
        df = pd.read_csv('iris.data', header=None)
        print("本地数据集加载成功！")
        print(f"数据集形状: {df.shape}")
        print("前5行数据:")
        print(df.head())
        return df
    except FileNotFoundError:
        print("本地文件未找到，请检查文件路径")
        return None


def load_iris_from_sklearn():
    """
    (2) 从scikit-learn库中直接加载iris数据集
    """
    iris = load_iris()
    X = iris.data  # 特征数据
    y = iris.target  # 目标变量
    feature_names = iris.feature_names  # 特征名称
    target_names = iris.target_names  # 目标类别名称

    print("\nscikit-learn数据集加载成功！")
    print(f"特征数据形状: {X.shape}")
    print(f"目标变量形状: {y.shape}")
    print(f"特征名称: {feature_names}")
    print(f"目标类别: {target_names}")

    return X, y, feature_names, target_names


def five_fold_cross_validation(X, y):
    """
    (3) 实现五折交叉验证进行模型训练
    """
    # 创建随机森林分类器
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # 定义五折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 定义评估指标
    scoring = {
        'accuracy': 'accuracy',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'f1_macro': 'f1_macro'
    }

    # 执行交叉验证
    cv_results = cross_validate(
        rf_classifier, X, y,
        cv=kf,
        scoring=scoring,
        return_train_score=True
    )

    return cv_results, rf_classifier


def calculate_metrics_manually(X, y, classifier):
    """
    手动计算模型的各项评估指标
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 训练模型
    classifier.fit(X_train, y_train)

    # 预测
    y_pred = classifier.predict(X_test)

    # 计算各项指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print("\n" + "=" * 50)
    print("手动计算的评估指标:")
    print("=" * 50)
    print(f"准确度 (Accuracy): {accuracy:.4f}")
    print(f"精度 (Precision-macro): {precision:.4f}")
    print(f"召回率 (Recall-macro): {recall:.4f}")
    print(f"F1值 (F1-macro): {f1:.4f}")

    # 显示详细的分类报告
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred,
                                target_names=['setosa', 'versicolor', 'virginica']))

    return accuracy, precision, recall, f1


def main():
    """
    主函数：执行所有实验内容
    """
    print("=" * 60)
    print("实验二：数据准备与模型评估")
    print("=" * 60)

    # (1) 尝试从本地加载数据集
    print("\n1. 从本地加载iris数据集:")
    local_data = load_iris_from_local()

    # (2) 从scikit-learn加载数据集
    print("\n2. 从scikit-learn加载iris数据集:")
    X, y, feature_names, target_names = load_iris_from_sklearn()

    # (3) 五折交叉验证
    print("\n3. 五折交叉验证结果:")
    cv_results, classifier = five_fold_cross_validation(X, y)

    # 输出交叉验证结果
    print("\n五折交叉验证各项指标:")
    print(f"训练准确度: {np.mean(cv_results['train_accuracy']):.4f} (+/- {np.std(cv_results['train_accuracy']):.4f})")
    print(f"测试准确度: {np.mean(cv_results['test_accuracy']):.4f} (+/- {np.std(cv_results['test_accuracy']):.4f})")
    print(
        f"测试精度: {np.mean(cv_results['test_precision_macro']):.4f} (+/- {np.std(cv_results['test_precision_macro']):.4f})")
    print(
        f"测试召回率: {np.mean(cv_results['test_recall_macro']):.4f} (+/- {np.std(cv_results['test_recall_macro']):.4f})")
    print(f"测试F1值: {np.mean(cv_results['test_f1_macro']):.4f} (+/- {np.std(cv_results['test_f1_macro']):.4f})")

    # 显示每一折的详细结果
    print("\n各折详细结果:")
    for i in range(5):
        print(f"折{i + 1}: 准确度={cv_results['test_accuracy'][i]:.4f}, "
              f"精度={cv_results['test_precision_macro'][i]:.4f}, "
              f"召回率={cv_results['test_recall_macro'][i]:.4f}, "
              f"F1值={cv_results['test_f1_macro'][i]:.4f}")

    # (4) 手动计算评估指标
    print("\n4. 手动计算评估指标:")
    accuracy, precision, recall, f1 = calculate_metrics_manually(X, y, classifier)

    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()