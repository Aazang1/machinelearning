import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, train_test_split
import math


class Node:
    """决策树节点类"""

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    """决策树分类器"""

    def __init__(self, max_depth=10, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None

    def _gini_index(self, y):
        """计算基尼指数"""
        if len(y) == 0:
            return 0
        class_counts = np.bincount(y)
        class_probs = class_counts / len(y)
        return 1 - np.sum(class_probs ** 2)

    def _best_split(self, X, y):
        """寻找最佳划分点"""
        n_samples, n_features = X.shape
        if n_samples <= 1:
            return None, None

        if self.n_features and self.n_features < n_features:
            features = np.random.choice(n_features, self.n_features, replace=False)
        else:
            features = range(n_features)

        best_gini = float('inf')
        best_feature, best_threshold = None, None

        for feature_index in features:
            feature_values = X[:, feature_index]
            possible_thresholds = np.unique(feature_values)

            for threshold in possible_thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = X[:, feature_index] > threshold

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_gini = self._gini_index(y[left_mask])
                right_gini = self._gini_index(y[right_mask])

                weighted_gini = (np.sum(left_mask) * left_gini + np.sum(right_mask) * right_gini) / n_samples

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """递归构建决策树"""
        n_samples, n_features = X.shape

        if (depth >= self.max_depth or
                n_samples < self.min_samples_split or
                len(np.unique(y)) == 1):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feature_index, threshold = self._best_split(X, y)

        if feature_index is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature_index=feature_index, threshold=threshold,
                    left=left_subtree, right=right_subtree)

    def _most_common_label(self, y):
        """返回出现最频繁的标签"""
        if len(y) == 0:
            return 0
        return np.bincount(y).argmax()

    def fit(self, X, y):
        """训练决策树"""
        if self.n_features is None:
            self.n_features = X.shape[1]
        else:
            self.n_features = min(X.shape[1], self.n_features)

        self.root = self._build_tree(X, y)

    def _traverse_tree(self, x, node):
        """遍历树进行预测"""
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def predict(self, X):
        """预测"""
        return np.array([self._traverse_tree(x, self.root) for x in X])


class RandomForest:
    """随机森林分类器"""

    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2, n_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def _bootstrap_sample(self, X, y):
        """生成自助采样样本"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        """训练随机森林"""
        self.trees = []
        for i in range(self.n_estimators):
            if (i + 1) % 20 == 0:
                print(f"训练第 {i + 1}/{self.n_estimators} 棵树...")

            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        """预测"""
        tree_preds = []
        for i, tree in enumerate(self.trees):
            pred = tree.predict(X)
            tree_preds.append(pred)

        tree_preds = np.array(tree_preds)
        tree_preds = tree_preds.T

        y_pred = []
        for sample_preds in tree_preds:
            most_common = Counter(sample_preds).most_common(1)[0][0]
            y_pred.append(most_common)

        return np.array(y_pred)


class EvaluationMetrics:
    """评估指标计算类"""

    @staticmethod
    def accuracy_score(y_true, y_pred):
        """计算准确率"""
        correct = np.sum(y_true == y_pred)
        return correct / len(y_true)

    @staticmethod
    def precision_score(y_true, y_pred, average='macro'):
        """计算精度"""
        classes = np.unique(np.concatenate([y_true, y_pred]))
        precisions = []

        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fp = np.sum((y_pred == cls) & (y_true != cls))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            precisions.append(precision)

        if average == 'macro':
            return np.mean(precisions)
        return precisions

    @staticmethod
    def recall_score(y_true, y_pred, average='macro'):
        """计算召回率"""
        classes = np.unique(np.concatenate([y_true, y_pred]))
        recalls = []

        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recalls.append(recall)

        if average == 'macro':
            return np.mean(recalls)
        return recalls

    @staticmethod
    def f1_score(y_true, y_pred, average='macro'):
        """计算F1值"""
        if average == 'macro':
            precision = EvaluationMetrics.precision_score(y_true, y_pred, average='macro')
            recall = EvaluationMetrics.recall_score(y_true, y_pred, average='macro')
        else:
            # 对于每个类别的F1计算
            precisions = EvaluationMetrics.precision_score(y_true, y_pred, average=None)
            recalls = EvaluationMetrics.recall_score(y_true, y_pred, average=None)
            f1_scores = []
            for i in range(len(precisions)):
                if precisions[i] + recalls[i] == 0:
                    f1_scores.append(0)
                else:
                    f1_scores.append(2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i]))

            if average is None:
                return f1_scores
            else:
                return np.mean(f1_scores)

        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def classification_report(y_true, y_pred, target_names=None):
        """生成分类报告"""
        classes = np.unique(np.concatenate([y_true, y_pred]))
        if target_names is None:
            target_names = [f'Class {cls}' for cls in classes]

        # 计算每个类别的指标
        precisions = EvaluationMetrics.precision_score(y_true, y_pred, average=None)
        recalls = EvaluationMetrics.recall_score(y_true, y_pred, average=None)
        f1_scores = EvaluationMetrics.f1_score(y_true, y_pred, average=None)

        report = []
        report.append("{:>15} {:>10} {:>10} {:>10} {:>10}".format(
            "类别", "精度", "召回率", "F1值", "支持数"))
        report.append("-" * 60)

        for i, cls in enumerate(classes):
            support = np.sum(y_true == cls)
            report.append("{:>15} {:>10.2f} {:>10.2f} {:>10.2f} {:>10}".format(
                target_names[cls], precisions[i], recalls[i], f1_scores[i], int(support)))

        # 宏平均
        macro_precision = EvaluationMetrics.precision_score(y_true, y_pred, average='macro')
        macro_recall = EvaluationMetrics.recall_score(y_true, y_pred, average='macro')
        macro_f1 = EvaluationMetrics.f1_score(y_true, y_pred, average='macro')
        accuracy = EvaluationMetrics.accuracy_score(y_true, y_pred)

        report.append("-" * 60)
        report.append("{:>15} {:>10.2f} {:>10.2f} {:>10.2f} {:>10}".format(
            "宏平均", macro_precision, macro_recall, macro_f1, len(y_true)))
        report.append("{:>15} {:>10.2f} {:>10} {:>10} {:>10}".format(
            "准确率", accuracy, "", "", len(y_true)))

        return "\n".join(report)


def five_fold_cross_validation(X, y, model, n_splits=5):
    """五折交叉验证"""
    print("开始五折交叉验证...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies, precisions, recalls, f1_scores = [], [], [], []

    fold = 1
    for train_idx, test_idx in kf.split(X):
        print(f"\n第 {fold} 折交叉验证...")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = EvaluationMetrics.accuracy_score(y_test, y_pred)
        prec = EvaluationMetrics.precision_score(y_test, y_pred)
        rec = EvaluationMetrics.recall_score(y_test, y_pred)
        f1 = EvaluationMetrics.f1_score(y_test, y_pred)

        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

        print(f"第 {fold} 折结果: 准确率={acc:.4f}, 精度={prec:.4f}, 召回率={rec:.4f}, F1={f1:.4f}")
        fold += 1

    return (np.mean(accuracies), np.mean(precisions),
            np.mean(recalls), np.mean(f1_scores),
            accuracies, precisions, recalls, f1_scores)


def main():
    # 加载数据
    iris = load_iris()
    X, y = iris.data, iris.target
    target_names = iris.target_names

    print("非调库随机森林实现 - Iris数据集分类")
    print("=" * 60)
    print(f"数据集信息: {X.shape[0]}个样本, {X.shape[1]}个特征")
    print(f"类别: {target_names}")
    print(f"类别分布: {np.bincount(y)}")

    # 创建随机森林模型
    rf = RandomForest(n_estimators=50, max_depth=5, min_samples_split=5, n_features=2)

    # 五折交叉验证
    print("\n" + "=" * 50)
    print("五折交叉验证")
    print("=" * 50)

    mean_accuracy, mean_precision, mean_recall, mean_f1, accs, precs, recs, f1s = five_fold_cross_validation(X, y, rf)

    print(f"\n五折交叉验证平均结果:")
    print(f"平均准确率: {mean_accuracy:.4f}")
    print(f"平均精度:   {mean_precision:.4f}")
    print(f"平均召回率: {mean_recall:.4f}")
    print(f"平均F1值:   {mean_f1:.4f}")

    print(f"\n各折详细结果:")
    for i in range(5):
        print(f"折{i + 1}: 准确率={accs[i]:.4f}, 精度={precs[i]:.4f}, 召回率={recs[i]:.4f}, F1值={f1s[i]:.4f}")

    # 独立测试集评估
    print("\n" + "=" * 50)
    print("独立测试集评估")
    print("=" * 50)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("训练最终模型...")
    rf_final = RandomForest(n_estimators=100, max_depth=8, min_samples_split=3, n_features=2)
    rf_final.fit(X_train, y_train)

    print("在测试集上进行预测...")
    y_pred = rf_final.predict(X_test)

    test_accuracy = EvaluationMetrics.accuracy_score(y_test, y_pred)
    test_precision = EvaluationMetrics.precision_score(y_test, y_pred)
    test_recall = EvaluationMetrics.recall_score(y_test, y_pred)
    test_f1 = EvaluationMetrics.f1_score(y_test, y_pred)

    print(f"\n独立测试集结果:")
    print(f"准确率: {test_accuracy:.4f}")
    print(f"精度:   {test_precision:.4f}")
    print(f"召回率: {test_recall:.4f}")
    print(f"F1值:   {test_f1:.4f}")

    # 显示详细分类报告
    print(f"\n详细分类报告:")
    report = EvaluationMetrics.classification_report(y_test, y_pred, target_names)
    print(report)

    # 结果总结
    print("\n" + "=" * 50)
    print("实验总结")
    print("=" * 50)
    print(f"✅ 非调库随机森林实现完成")
    print(f"✅ 五折交叉验证平均准确率: {mean_accuracy:.4f}")
    print(f"✅ 独立测试集准确率: {test_accuracy:.4f}")
    print(f"✅ 最终性能指标:")
    print(f"   - 精度: {test_precision:.4f}")
    print(f"   - 召回率: {test_recall:.4f}")
    print(f"   - F1值: {test_f1:.4f}")


if __name__ == "__main__":
    main()