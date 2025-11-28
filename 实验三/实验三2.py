import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import math


class Node:
    """决策树节点类"""

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class C45DecisionTree:
    """改进的C4.5决策树实现"""

    def __init__(self, max_depth=10, min_samples_split=2, min_impurity_decrease=0.01):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.tree = None

    def _entropy(self, y):
        """计算信息熵"""
        if len(y) == 0:
            return 0
        class_counts = np.bincount(y)
        probabilities = class_counts / len(y)
        entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
        return entropy

    def _information_gain_ratio(self, X, y, feature_idx):
        """计算信息增益比"""
        n_samples = len(y)

        # 获取特征值并排序
        feature_values = np.unique(X[:, feature_idx])

        best_gain_ratio = -1
        best_threshold = None

        # 尝试所有可能的分割点（相邻值的中间点）
        for i in range(len(feature_values) - 1):
            threshold = (feature_values[i] + feature_values[i + 1]) / 2

            # 分割数据
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask

            y_left = y[left_mask]
            y_right = y[right_mask]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            # 计算信息增益
            parent_entropy = self._entropy(y)
            left_entropy = self._entropy(y_left)
            right_entropy = self._entropy(y_right)

            n_left, n_right = len(y_left), len(y_right)
            weighted_entropy = (n_left / n_samples) * left_entropy + (n_right / n_samples) * right_entropy
            information_gain = parent_entropy - weighted_entropy

            # 计算分裂信息
            left_ratio = n_left / n_samples
            right_ratio = n_right / n_samples
            split_info = - (left_ratio * np.log2(left_ratio) + right_ratio * np.log2(right_ratio))

            # 计算信息增益比
            if split_info == 0:
                continue

            gain_ratio = information_gain / split_info

            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_threshold = threshold

        return best_gain_ratio, best_threshold

    def _find_best_split(self, X, y):
        """寻找最佳分裂特征和阈值"""
        n_features = X.shape[1]
        best_gain_ratio = -1
        best_feature = None
        best_threshold = None

        for feature_idx in range(n_features):
            gain_ratio, threshold = self._information_gain_ratio(X, y, feature_idx)

            if gain_ratio is not None and gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_feature = feature_idx
                best_threshold = threshold

        return best_feature, best_threshold, best_gain_ratio

    def _build_tree(self, X, y, depth=0):
        """递归构建决策树"""
        n_samples = len(y)

        # 终止条件1: 所有样本属于同一类别
        if len(np.unique(y)) == 1:
            return Node(value=y[0])

        # 终止条件2: 达到最大深度
        if depth >= self.max_depth:
            majority_class = np.argmax(np.bincount(y))
            return Node(value=majority_class)

        # 终止条件3: 样本数太少
        if n_samples < self.min_samples_split:
            majority_class = np.argmax(np.bincount(y))
            return Node(value=majority_class)

        # 寻找最佳分裂
        best_feature, best_threshold, best_gain_ratio = self._find_best_split(X, y)

        # 终止条件4: 没有合适的分裂或信息增益比太小
        if best_feature is None or best_gain_ratio < self.min_impurity_decrease:
            majority_class = np.argmax(np.bincount(y))
            return Node(value=majority_class)

        # 分割数据
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # 递归构建子树
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold,
                    left=left_subtree, right=right_subtree)

    def _predict_single(self, node, x):
        """单样本预测"""
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_single(node.left, x)
        else:
            return self._predict_single(node.right, x)

    def fit(self, X, y):
        """训练决策树"""
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        """批量预测"""
        return np.array([self._predict_single(self.tree, x) for x in X])


def main():
    # 加载数据
    iris = load_iris()
    X, y = iris.data, iris.target

    print("数据集信息:")
    print(f"特征形状: {X.shape}")
    print(f"标签分布: {np.unique(y, return_counts=True)}")
    print(f"特征名称: {iris.feature_names}")
    print(f"目标名称: {iris.target_names}")

    # 五折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"\n=== 第{fold + 1}折训练 ===")
        print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

        # 创建并训练C4.5决策树
        tree = C45DecisionTree(max_depth=5, min_samples_split=3, min_impurity_decrease=0.01)
        tree.fit(X_train, y_train)

        # 预测
        y_pred = tree.predict(X_test)

        # 计算指标
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        accuracies.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        print(f"第{fold + 1}折结果:")
        print(f"准确率: {acc:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")

        # 打印预测分布
        print(f"真实标签分布: {np.bincount(y_test)}")
        print(f"预测标签分布: {np.bincount(y_pred, minlength=3)}")

    # 输出平均结果
    print("\n" + "=" * 60)
    print("五折交叉验证平均结果:")
    print(f"平均准确率: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")
    print(f"平均精确率: {np.mean(precisions):.4f} (±{np.std(precisions):.4f})")
    print(f"平均召回率: {np.mean(recalls):.4f} (±{np.std(recalls):.4f})")
    print(f"平均F1分数: {np.mean(f1_scores):.4f} (±{np.std(f1_scores):.4f})")

    # 对比sklearn的决策树
    from sklearn.tree import DecisionTreeClassifier
    print("\n" + "=" * 60)
    print("对比sklearn的决策树性能:")

    sklearn_accuracies = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        sklearn_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=3, random_state=42)
        sklearn_tree.fit(X_train, y_train)
        y_pred_sklearn = sklearn_tree.predict(X_test)
        sklearn_acc = accuracy_score(y_test, y_pred_sklearn)
        sklearn_accuracies.append(sklearn_acc)

    print(f"Sklearn决策树平均准确率: {np.mean(sklearn_accuracies):.4f} (±{np.std(sklearn_accuracies):.4f})")


if __name__ == "__main__":
    main()