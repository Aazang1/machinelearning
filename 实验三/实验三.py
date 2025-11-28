import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class C45DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 pre_pruning=True, post_pruning=True, ccp_alpha=0.01):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.pre_pruning = pre_pruning
        self.post_pruning = post_pruning
        self.ccp_alpha = ccp_alpha
        self.tree = None
        self.feature_names = None

    def _entropy(self, y):
        """计算信息熵"""
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def _information_gain_ratio(self, X, y, feature_idx):
        """计算信息增益比"""
        # 原始熵
        total_entropy = self._entropy(y)

        # 按特征值划分
        feature_values = X[:, feature_idx]
        unique_values = np.unique(feature_values)

        # 连续特征处理（二分法）
        if len(unique_values) > 10:  # 假设连续特征
            # 找到最佳分割点
            sorted_values = np.sort(unique_values)
            thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2

            best_gain_ratio = -1
            best_threshold = None

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = feature_values > threshold

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # 计算信息增益
                left_entropy = self._entropy(y[left_mask])
                right_entropy = self._entropy(y[right_mask])
                weighted_entropy = (np.sum(left_mask) * left_entropy +
                                    np.sum(right_mask) * right_entropy) / len(y)
                information_gain = total_entropy - weighted_entropy

                # 计算分裂信息
                split_info = -((np.sum(left_mask) / len(y)) * np.log2(np.sum(left_mask) / len(y)) +
                               (np.sum(right_mask) / len(y)) * np.log2(np.sum(right_mask) / len(y)))

                if split_info == 0:
                    continue

                gain_ratio = information_gain / split_info

                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_threshold = threshold

            return best_gain_ratio, best_threshold, True
        else:
            # 离散特征
            weighted_entropy = 0
            split_info = 0

            for value in unique_values:
                mask = feature_values == value
                if np.sum(mask) == 0:
                    continue
                subset_entropy = self._entropy(y[mask])
                weight = np.sum(mask) / len(y)
                weighted_entropy += weight * subset_entropy
                split_info -= weight * np.log2(weight) if weight > 0 else 0

            information_gain = total_entropy - weighted_entropy
            gain_ratio = information_gain / split_info if split_info > 0 else 0

            return gain_ratio, None, False

    def _find_best_split(self, X, y):
        """找到最佳分割特征"""
        best_gain_ratio = -1
        best_feature = None
        best_threshold = None
        best_is_continuous = False

        for feature_idx in range(X.shape[1]):
            gain_ratio, threshold, is_continuous = self._information_gain_ratio(X, y, feature_idx)

            if gain_ratio is not None and gain_ratio > best_gain_ratio:  # 修复：检查gain_ratio是否为None
                best_gain_ratio = gain_ratio
                best_feature = feature_idx
                best_threshold = threshold
                best_is_continuous = is_continuous

        return best_feature, best_threshold, best_is_continuous, best_gain_ratio  # 修复：返回best_gain_ratio

    def _pre_pruning_check(self, X, y, depth):
        """预剪枝检查"""
        if not self.pre_pruning:
            return False

        # 如果所有样本属于同一类别
        if len(np.unique(y)) == 1:
            return True

        # 如果样本数小于最小分割样本数
        if len(y) < self.min_samples_split:
            return True

        # 如果达到最大深度
        if self.max_depth is not None and depth >= self.max_depth:
            return True

        # 如果没有特征可用于分割
        if X.shape[1] == 0:
            return True

        return False

    def _build_tree(self, X, y, depth=0):
        """递归构建决策树"""
        # 预剪枝检查
        if self._pre_pruning_check(X, y, depth):
            return self._create_leaf_node(y)

        # 找到最佳分割
        best_feature, best_threshold, is_continuous, best_gain_ratio = self._find_best_split(X, y)  # 修复：接收返回值

        # 如果没有合适的分割
        if best_feature is None or best_gain_ratio <= 0:  # 修复：使用返回的best_gain_ratio
            return self._create_leaf_node(y)

        # 创建节点
        node = {
            'feature': best_feature,
            'threshold': best_threshold,
            'is_continuous': is_continuous,
            'children': {}
        }

        # 根据特征类型进行分割
        if is_continuous:
            # 连续特征二分
            left_mask = X[:, best_feature] <= best_threshold
            right_mask = X[:, best_feature] > best_threshold

            # 检查子节点样本数是否满足最小要求
            if np.sum(left_mask) >= self.min_samples_leaf:
                node['children']['left'] = self._build_tree(X[left_mask], y[left_mask], depth + 1)
            else:
                node['children']['left'] = self._create_leaf_node(y[left_mask])

            if np.sum(right_mask) >= self.min_samples_leaf:
                node['children']['right'] = self._build_tree(X[right_mask], y[right_mask], depth + 1)
            else:
                node['children']['right'] = self._create_leaf_node(y[right_mask])
        else:
            # 离散特征多分
            feature_values = X[:, best_feature]
            unique_values = np.unique(feature_values)

            for value in unique_values:
                mask = feature_values == value
                if np.sum(mask) > 0:  # 确保有样本
                    if np.sum(mask) >= self.min_samples_leaf:
                        node['children'][value] = self._build_tree(X[mask], y[mask], depth + 1)
                    else:
                        node['children'][value] = self._create_leaf_node(y[mask])

        return node

    def _create_leaf_node(self, y):
        """创建叶节点"""
        if len(y) == 0:
            return {
                'leaf': True,
                'class': 0,  # 默认类别
                'probability': np.array([1.0, 0, 0])[:len(np.unique(np.concatenate([y, [0, 1, 2]])))],  # 适应类别数
                'samples': 0
            }

        counts = np.bincount(y)
        return {
            'leaf': True,
            'class': np.argmax(counts),
            'probability': counts / np.sum(counts),
            'samples': len(y)
        }

    def _post_prune(self, tree, X_val, y_val):
        """后剪枝"""
        if not self.post_pruning or len(y_val) == 0:
            return tree

        if 'leaf' in tree:
            return tree

        # 递归剪枝子节点
        feature = tree['feature']
        is_continuous = tree['is_continuous']

        if is_continuous:
            left_mask = X_val[:, feature] <= tree['threshold']
            right_mask = X_val[:, feature] > tree['threshold']

            if np.sum(left_mask) > 0:
                tree['children']['left'] = self._post_prune(tree['children']['left'],
                                                            X_val[left_mask], y_val[left_mask])
            if np.sum(right_mask) > 0:
                tree['children']['right'] = self._post_prune(tree['children']['right'],
                                                             X_val[right_mask], y_val[right_mask])
        else:
            feature_values = X_val[:, feature]
            unique_values = np.unique(feature_values)

            for value in unique_values:
                mask = feature_values == value
                if np.sum(mask) > 0 and value in tree['children']:
                    tree['children'][value] = self._post_prune(tree['children'][value],
                                                               X_val[mask], y_val[mask])

        # 计算剪枝前后的误差
        error_before = 1 - self._calculate_accuracy(tree, X_val, y_val)

        # 创建叶节点
        leaf_node = self._create_leaf_node(y_val)
        error_after = 1 - np.max(leaf_node['probability']) if len(y_val) > 0 else 1.0

        # 计算复杂度惩罚
        n_leaves = self._count_leaves(tree)
        cost_before = error_before + self.ccp_alpha * n_leaves
        cost_after = error_after + self.ccp_alpha * 1  # 叶节点数为1

        # 如果剪枝后成本更低，则进行剪枝
        if cost_after <= cost_before:
            return leaf_node

        return tree

    def _calculate_accuracy(self, tree, X, y):
        """计算准确率"""
        if len(y) == 0:
            return 0
        predictions = self._predict_single_tree(tree, X)
        return accuracy_score(y, predictions)

    def _count_leaves(self, tree):
        """计算叶节点数量"""
        if 'leaf' in tree:
            return 1

        count = 0
        for child in tree['children'].values():
            count += self._count_leaves(child)
        return count

    def _predict_single_tree(self, tree, X):
        """单棵树预测"""
        predictions = []
        for i in range(len(X)):
            node = tree
            while 'leaf' not in node:
                feature_val = X[i][node['feature']]

                if node['is_continuous']:
                    if feature_val <= node['threshold']:
                        if 'left' in node['children']:
                            node = node['children']['left']
                        else:
                            break
                    else:
                        if 'right' in node['children']:
                            node = node['children']['right']
                        else:
                            break
                else:
                    if feature_val in node['children']:
                        node = node['children'][feature_val]
                    else:
                        # 如果遇到未见过的特征值，选择第一个子节点
                        if node['children']:
                            node = list(node['children'].values())[0]
                        else:
                            break

            predictions.append(node['class'])

        return np.array(predictions)

    def fit(self, X, y, validation_ratio=0.2):
        """训练模型"""
        self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # 分割训练集和验证集用于后剪枝
        if self.post_pruning and len(X) > 10:
            split_idx = int(len(X) * (1 - validation_ratio))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            X_train, y_train = X, y
            X_val, y_val = X, y

        # 构建树
        self.tree = self._build_tree(X_train, y_train)

        # 后剪枝
        if self.post_pruning and len(X_val) > 0:
            self.tree = self._post_prune(self.tree, X_val, y_val)

    def predict(self, X):
        """预测"""
        return self._predict_single_tree(self.tree, X)

    def predict_proba(self, X):
        """预测概率"""
        probabilities = []
        for i in range(len(X)):
            node = self.tree
            while 'leaf' not in node:
                feature_val = X[i][node['feature']]

                if node['is_continuous']:
                    if feature_val <= node['threshold']:
                        if 'left' in node['children']:
                            node = node['children']['left']
                        else:
                            break
                    else:
                        if 'right' in node['children']:
                            node = node['children']['right']
                        else:
                            break
                else:
                    if feature_val in node['children']:
                        node = node['children'][feature_val]
                    else:
                        if node['children']:
                            node = list(node['children'].values())[0]
                        else:
                            break

            probabilities.append(node['probability'])

        return np.array(probabilities)


def evaluate_model(model, X, y):
    """评估模型性能"""
    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='macro', zero_division=0)
    recall = recall_score(y, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y, y_pred, average='macro', zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def main():
    # 1. 加载数据集
    print("1. 加载Iris数据集...")
    iris = load_iris()
    X, y = iris.data, iris.target

    print(f"数据集形状: {X.shape}")
    print(f"目标变量分布: {np.bincount(y)}")
    print(f"特征名称: {iris.feature_names}")
    print(f"类别名称: {iris.target_names}")

    # 2. 五折交叉验证
    print("\n2. 开始五折交叉验证...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    fold = 1

    for train_index, test_index in kf.split(X):
        print(f"\n--- 第 {fold} 折 ---")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 训练模型（带预剪枝和后剪枝）
        model = C45DecisionTree(
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            pre_pruning=True,
            post_pruning=True,
            ccp_alpha=0.01
        )

        model.fit(X_train, y_train)

        # 评估模型
        metrics = evaluate_model(model, X_test, y_test)
        results.append(metrics)

        print(f"准确度: {metrics['accuracy']:.4f}")
        print(f"精度: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1值: {metrics['f1_score']:.4f}")

        fold += 1

    # 3. 结果分析
    print("\n3. 五折交叉验证结果分析:")
    print("=" * 50)

    # 计算平均指标
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    avg_precision = np.mean([r['precision'] for r in results])
    avg_recall = np.mean([r['recall'] for r in results])
    avg_f1 = np.mean([r['f1_score'] for r in results])

    std_accuracy = np.std([r['accuracy'] for r in results])
    std_precision = np.std([r['precision'] for r in results])
    std_recall = np.std([r['recall'] for r in results])
    std_f1 = np.std([r['f1_score'] for r in results])

    print(f"平均准确度: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"平均精度: {avg_precision:.4f} ± {std_precision:.4f}")
    print(f"平均召回率: {avg_recall:.4f} ± {std_recall:.4f}")
    print(f"平均F1值: {avg_f1:.4f} ± {std_f1:.4f}")

    # 4. 比较不同配置的性能
    print("\n4. 不同剪枝策略比较:")
    print("=" * 50)

    configurations = [
        {"pre_pruning": True, "post_pruning": True, "name": "预剪枝+后剪枝"},
        {"pre_pruning": True, "post_pruning": False, "name": "仅预剪枝"},
        {"pre_pruning": False, "post_pruning": True, "name": "仅后剪枝"},
        {"pre_pruning": False, "post_pruning": False, "name": "无剪枝"}
    ]

    comparison_results = []

    for config in configurations:
        fold_accuracies = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = C45DecisionTree(
                max_depth=5,
                min_samples_split=2,
                min_samples_leaf=1,
                pre_pruning=config["pre_pruning"],
                post_pruning=config["post_pruning"],
                ccp_alpha=0.01
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            fold_accuracies.append(accuracy)

        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        comparison_results.append({
            'name': config['name'],
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy
        })

        print(f"{config['name']}: {mean_accuracy:.4f} ± {std_accuracy:.4f}")

    # 找出最佳配置
    best_config = max(comparison_results, key=lambda x: x['mean_accuracy'])
    print(f"\n最佳配置: {best_config['name']} (准确度: {best_config['mean_accuracy']:.4f})")


if __name__ == "__main__":
    main()