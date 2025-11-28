import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate, KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
import warnings

warnings.filterwarnings('ignore')


def load_iris_dataset():
    """
    (2) 从scikit-learn库中直接加载iris数据集
    """
    iris = load_iris()
    X = iris.data  # 特征数据
    y = iris.target  # 目标变量
    feature_names = iris.feature_names  # 特征名称
    target_names = iris.target_names  # 目标类别名称

    print("Iris数据集加载成功！")
    print(f"特征数据形状: {X.shape}")
    print(f"目标变量形状: {y.shape}")
    print(f"特征名称: {feature_names}")
    print(f"目标类别: {target_names}")
    print(f"样本分布: {np.bincount(y)} - {target_names}")

    # 创建DataFrame用于展示
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['class'] = df['target'].map({i: name for i, name in enumerate(target_names)})

    print("\n数据集前5行:")
    print(df.head())
    print("\n数据集基本信息:")
    print(df.describe())

    return X, y, feature_names, target_names, df


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
        return_train_score=True,
        return_estimator=True
    )

    return cv_results, rf_classifier


def print_cross_validation_results(cv_results, target_names):
    """
    打印交叉验证的详细结果
    """
    print("\n" + "=" * 60)
    print("五折交叉验证结果")
    print("=" * 60)

    # 整体统计结果
    print("\n整体统计（均值 ± 标准差）:")
    print(f"训练准确度: {np.mean(cv_results['train_accuracy']):.4f} (±{np.std(cv_results['train_accuracy']):.4f})")
    print(f"测试准确度: {np.mean(cv_results['test_accuracy']):.4f} (±{np.std(cv_results['test_accuracy']):.4f})")
    print(
        f"测试精度:   {np.mean(cv_results['test_precision_macro']):.4f} (±{np.std(cv_results['test_precision_macro']):.4f})")
    print(
        f"测试召回率: {np.mean(cv_results['test_recall_macro']):.4f} (±{np.std(cv_results['test_recall_macro']):.4f})")
    print(f"测试F1值:   {np.mean(cv_results['test_f1_macro']):.4f} (±{np.std(cv_results['test_f1_macro']):.4f})")

    # 每一折的详细结果
    print("\n各折详细结果:")
    print("折号\t准确度\t\t精度\t\t召回率\t\tF1值")
    print("-" * 55)
    for i in range(5):
        print(f"{i + 1}\t"
              f"{cv_results['test_accuracy'][i]:.4f}\t\t"
              f"{cv_results['test_precision_macro'][i]:.4f}\t\t"
              f"{cv_results['test_recall_macro'][i]:.4f}\t\t"
              f"{cv_results['test_f1_macro'][i]:.4f}")


def calculate_detailed_metrics(X, y, target_names):
    """
    (4) 计算并输出模型的准确度、精度、召回率和F1值
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 创建并训练模型
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # 预测
    y_pred = rf_classifier.predict(X_test)

    # 计算各项指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # 计算每个类别的指标
    precision_per_class = precision_score(y_test, y_pred, average=None)
    recall_per_class = recall_score(y_test, y_pred, average=None)
    f1_per_class = f1_score(y_test, y_pred, average=None)

    print("\n" + "=" * 60)
    print("详细评估指标")
    print("=" * 60)

    # 整体指标
    print(f"\n整体指标:")
    print(f"准确度 (Accuracy): {accuracy:.4f}")
    print(f"精度 (Precision-macro): {precision:.4f}")
    print(f"召回率 (Recall-macro): {recall:.4f}")
    print(f"F1值 (F1-macro): {f1:.4f}")

    # 每个类别的指标
    print(f"\n各类别详细指标:")
    print("类别\t\t精度\t\t召回率\t\tF1值")
    print("-" * 45)
    for i, class_name in enumerate(target_names):
        print(f"{class_name:<12}\t{precision_per_class[i]:.4f}\t\t"
              f"{recall_per_class[i]:.4f}\t\t{f1_per_class[i]:.4f}")

    # 显示分类报告
    print(f"\n详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 显示混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f"混淆矩阵:")
    print(cm)

    return accuracy, precision, recall, f1, rf_classifier


def feature_importance_analysis(rf_classifier, feature_names):
    """
    特征重要性分析
    """
    print("\n" + "=" * 60)
    print("特征重要性分析")
    print("=" * 60)

    importance_df = pd.DataFrame({
        '特征': feature_names,
        '重要性': rf_classifier.feature_importances_
    }).sort_values('重要性', ascending=False)

    print(importance_df.to_string(index=False))

    # 可视化特征重要性
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['特征'], importance_df['重要性'])
    plt.xlabel('特征重要性')
    plt.title('Iris数据集特征重要性')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def model_interpretation_example(rf_classifier, X, y, feature_names, target_names):
    """
    模型解释示例
    """
    print("\n" + "=" * 60)
    print("模型预测示例")
    print("=" * 60)

    # 选择一个测试样本进行解释
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    rf_classifier.fit(X_train, y_train)

    # 随机选择一个测试样本
    sample_idx = 0
    sample = X_test[sample_idx].reshape(1, -1)
    true_label = y_test[sample_idx]
    prediction = rf_classifier.predict(sample)[0]
    probabilities = rf_classifier.predict_proba(sample)[0]

    print(f"\n样本特征值:")
    for i, feature in enumerate(feature_names):
        print(f"{feature}: {sample[0][i]:.2f}")

    print(f"\n真实类别: {target_names[true_label]}")
    print(f"预测类别: {target_names[prediction]}")
    print(f"\n预测概率:")
    for i, prob in enumerate(probabilities):
        print(f"{target_names[i]}: {prob:.4f}")


def main():
    """
    主函数：执行所有实验内容
    """
    print("=" * 60)
    print("实验一：Iris数据集模型评估")
    print("=" * 60)

    # (2) 从scikit-learn加载iris数据集
    print("\n1. 数据加载与探索")
    X, y, feature_names, target_names, df = load_iris_dataset()

    # (3) 五折交叉验证
    print("\n2. 五折交叉验证")
    cv_results, rf_classifier = five_fold_cross_validation(X, y)
    print_cross_validation_results(cv_results, target_names)

    # (4) 计算详细评估指标
    print("\n3. 详细评估指标计算")
    accuracy, precision, recall, f1, trained_model = calculate_detailed_metrics(X, y, target_names)

    # 特征重要性分析
    print("\n4. 特征重要性分析")
    feature_importance_analysis(trained_model, feature_names)

    # 模型解释示例
    print("\n5. 模型预测示例")
    model_interpretation_example(rf_classifier, X, y, feature_names, target_names)

    # 实验总结
    print("\n" + "=" * 60)
    print("实验总结")
    print("=" * 60)
    print(f"✅ 数据集: Iris (150个样本, 4个特征, 3个类别)")
    print(f"✅ 使用模型: 随机森林 (100棵树)")
    print(f"✅ 评估方法: 五折交叉验证 + 独立测试集")
    print(f"✅ 最终性能:")
    print(f"   - 准确度: {accuracy:.4f}")
    print(f"   - 精度:   {precision:.4f}")
    print(f"   - 召回率: {recall:.4f}")
    print(f"   - F1值:   {f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()