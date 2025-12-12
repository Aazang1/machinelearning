import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


class BPNeuralNetwork:
    """
    BP神经网络实现
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, reg_lambda=0.01):
        """
        初始化神经网络

        参数说明：
        -----------
        input_size: 输入层神经元数量
        hidden_size: 隐藏层神经元数量
        output_size: 输出层神经元数量
        learning_rate: 学习率
        reg_lambda: 正则化参数
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda

        # 初始化权重
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

        # 存储训练历史
        self.loss_history = []
        self.accuracy_history = []

    def sigmoid(self, x):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Sigmoid导数"""
        return x * (1 - x)

    def softmax(self, x):
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """前向传播"""
        # 输入层到隐藏层
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        # 隐藏层到输出层
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)

        return self.a2

    def compute_loss(self, y, y_pred):
        """计算交叉熵损失"""
        m = y.shape[0]
        # 避免log(0)的情况
        y_pred = np.clip(y_pred, 1e-12, 1. - 1e-12)

        # 交叉熵损失
        cross_entropy = -np.sum(y * np.log(y_pred)) / m

        # L2正则化
        reg_loss = 0.5 * self.reg_lambda * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2))) / m

        return cross_entropy + reg_loss

    def backward(self, X, y, y_pred):
        """反向传播"""
        m = X.shape[0]

        # 输出层误差
        delta3 = y_pred - y

        # 隐藏层误差
        delta2 = np.dot(delta3, self.W2.T) * self.sigmoid_derivative(self.a1)

        # 计算梯度（包含正则化项）
        dW2 = np.dot(self.a1.T, delta3) / m + self.reg_lambda * self.W2 / m
        db2 = np.sum(delta3, axis=0, keepdims=True) / m
        dW1 = np.dot(X.T, delta2) / m + self.reg_lambda * self.W1 / m
        db1 = np.sum(delta2, axis=0, keepdims=True) / m

        return dW1, db1, dW2, db2

    def update_parameters(self, dW1, db1, dW2, db2):
        """更新权重参数"""
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X, y, epochs=1000, batch_size=32, verbose=True):
        """
        训练神经网络

        参数说明：
        -----------
        X: 训练数据特征
        y: 训练数据标签（one-hot编码）
        epochs: 训练轮数
        batch_size: 批大小
        verbose: 是否显示训练信息
        """
        n_samples = X.shape[0]

        for epoch in range(epochs):
            # 打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            # 小批量训练
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]

                # 前向传播
                y_pred = self.forward(X_batch)

                # 计算损失
                loss = self.compute_loss(y_batch, y_pred)
                epoch_loss += loss

                # 反向传播
                dW1, db1, dW2, db2 = self.backward(X_batch, y_batch, y_pred)

                # 更新参数
                self.update_parameters(dW1, db1, dW2, db2)

            # 计算平均损失
            epoch_loss /= (n_samples / batch_size)
            self.loss_history.append(epoch_loss)

            # 每100轮显示一次训练信息
            if verbose and (epoch + 1) % 100 == 0:
                y_pred_all = self.predict(X)
                y_true_labels = np.argmax(y, axis=1)
                y_pred_labels = np.argmax(y_pred_all, axis=1)
                accuracy = accuracy_score(y_true_labels, y_pred_labels)
                self.accuracy_history.append(accuracy)

                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

    def predict(self, X):
        """预测"""
        y_pred = self.forward(X)
        return y_pred

    def predict_classes(self, X):
        """预测类别"""
        y_pred = self.predict(X)
        return np.argmax(y_pred, axis=1)


def load_and_preprocess_data():
    """
    加载和预处理数据
    """
    # 加载iris数据集
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    print("=" * 60)
    print("数据集基本信息:")
    print("=" * 60)
    print(f"特征数量: {X.shape[1]}")
    print(f"样本数量: {X.shape[0]}")
    print(f"类别数量: {len(np.unique(y))}")
    print(f"特征名称: {feature_names}")
    print(f"类别名称: {target_names}")

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 标签one-hot编码
    # 使用新版本的OneHotEncoder
    encoder = OneHotEncoder()
    y_onehot = encoder.fit_transform(y.reshape(-1, 1)).toarray()

    return X_scaled, y, y_onehot, feature_names, target_names, scaler


def evaluate_model(model, X_test, y_test, y_test_onehot):
    """
    评估模型性能
    """
    # 预测
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = y_test

    # 计算各项指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_true': y_true,
        'y_pred': y_pred
    }


def plot_training_history(loss_history, accuracy_history):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 绘制损失曲线
    ax1.plot(loss_history)
    ax1.set_title('训练损失曲线')
    ax1.set_xlabel('训练轮数')
    ax1.set_ylabel('损失值')
    ax1.grid(True, alpha=0.3)

    # 绘制准确率曲线
    ax2.plot(accuracy_history)
    ax2.set_title('训练准确率曲线')
    ax2.set_xlabel('训练轮数 (x100)')
    ax2.set_ylabel('准确率')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.show()


def plot_performance_comparison(fold_metrics):
    """绘制五折交叉验证的性能比较"""
    metrics_names = ['准确率', '精确率', '召回率', 'F1分数']
    folds = list(range(1, len(fold_metrics) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(['accuracy', 'precision', 'recall', 'f1']):
        values = [m[metric] for m in fold_metrics]

        axes[idx].bar(folds, values, color='skyblue', edgecolor='black')
        axes[idx].set_title(f'{metrics_names[idx]} (五折交叉验证)')
        axes[idx].set_xlabel('折数')
        axes[idx].set_ylabel(metrics_names[idx])
        axes[idx].set_xticks(folds)
        axes[idx].set_ylim([0.8, 1.0])
        axes[idx].grid(True, alpha=0.3, axis='y')

        # 在柱子上显示数值
        for i, v in enumerate(values):
            axes[idx].text(i + 1, v + 0.005, f'{v:.3f}',
                           ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_data_distribution(y, target_names):
    """绘制数据分布"""
    unique, counts = np.unique(y, return_counts=True)

    plt.figure(figsize=(8, 6))
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    bars = plt.bar([target_names[i] for i in unique], counts, color=colors, edgecolor='black')

    plt.title('数据集类别分布')
    plt.xlabel('类别')
    plt.ylabel('样本数量')

    # 在柱子上显示数量
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{count}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def main():
    """
    主函数：执行五折交叉验证
    """
    # 1. 加载和预处理数据
    X, y, y_onehot, feature_names, target_names, scaler = load_and_preprocess_data()

    # 绘制数据分布
    print("\n绘制数据分布图...")
    plot_data_distribution(y, target_names)

    # 2. 五折交叉验证
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 存储每折的结果
    fold_results = []
    fold_metrics = []

    print("\n" + "=" * 60)
    print("开始五折交叉验证...")
    print("=" * 60)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n第 {fold} 折:")
        print(f"训练集大小: {len(train_idx)}，测试集大小: {len(test_idx)}")

        # 划分训练集和测试集
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_train_onehot, y_test_onehot = y_onehot[train_idx], y_onehot[test_idx]

        # 创建和训练模型
        input_size = X_train.shape[1]
        hidden_size = 8
        output_size = len(target_names)

        model = BPNeuralNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            learning_rate=0.1,
            reg_lambda=0.001
        )

        # 训练模型
        print(f"训练模型中...")
        model.train(X_train, y_train_onehot, epochs=500, batch_size=16, verbose=False)

        # 评估模型
        print(f"评估模型...")
        result = evaluate_model(model, X_test, y_test, y_test_onehot)

        fold_results.append(result)
        fold_metrics.append({
            'accuracy': result['accuracy'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1': result['f1']
        })

        print(f"准确率: {result['accuracy']:.4f}")
        print(f"精确率: {result['precision']:.4f}")
        print(f"召回率: {result['recall']:.4f}")
        print(f"F1分数: {result['f1']:.4f}")

    # 3. 计算平均性能
    print("\n" + "=" * 60)
    print("五折交叉验证结果汇总:")
    print("=" * 60)

    # 创建表格显示每折结果
    print("\n各折性能指标:")
    print("-" * 60)
    print(f"{'折数':<6} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10}")
    print("-" * 60)

    for i, metrics in enumerate(fold_metrics, 1):
        print(f"{i:<6} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")

    print("-" * 60)

    # 计算平均值和标准差
    avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
    avg_precision = np.mean([m['precision'] for m in fold_metrics])
    avg_recall = np.mean([m['recall'] for m in fold_metrics])
    avg_f1 = np.mean([m['f1'] for m in fold_metrics])

    std_accuracy = np.std([m['accuracy'] for m in fold_metrics])
    std_precision = np.std([m['precision'] for m in fold_metrics])
    std_recall = np.std([m['recall'] for m in fold_metrics])
    std_f1 = np.std([m['f1'] for m in fold_metrics])

    print(f"\n平均性能指标 (±标准差):")
    print(f"平均准确率: {avg_accuracy:.4f} (±{std_accuracy:.4f})")
    print(f"平均精确率: {avg_precision:.4f} (±{std_precision:.4f})")
    print(f"平均召回率: {avg_recall:.4f} (±{std_recall:.4f})")
    print(f"平均F1分数: {avg_f1:.4f} (±{std_f1:.4f})")

    # 4. 绘制训练历史（最后一折）
    print("\n绘制训练历史（最后一折）...")
    plot_training_history(model.loss_history, model.accuracy_history)

    # 5. 绘制五折交叉验证性能比较
    print("绘制五折交叉验证性能比较图...")
    plot_performance_comparison(fold_metrics)

    # 6. 使用全部数据训练最终模型
    print("\n" + "=" * 60)
    print("使用全部数据训练最终模型...")
    print("=" * 60)

    final_model = BPNeuralNetwork(
        input_size=X.shape[1],
        hidden_size=8,
        output_size=len(target_names),
        learning_rate=0.1,
        reg_lambda=0.001
    )

    final_model.train(X, y_onehot, epochs=1000, batch_size=16, verbose=True)

    # 7. 绘制最终模型性能
    y_pred_all = final_model.predict_classes(X)

    print("\n绘制混淆矩阵...")
    plot_confusion_matrix(y, y_pred_all, target_names)

    # 8. 打印详细分类报告
    print("\n" + "=" * 60)
    print("详细分类报告:")
    print("=" * 60)
    print(classification_report(y, y_pred_all, target_names=target_names, zero_division=0))

    # 9. 模型参数分析
    print("\n" + "=" * 60)
    print("模型参数分析:")
    print("=" * 60)
    print(f"输入层到隐藏层权重 W1 形状: {final_model.W1.shape}")
    print(f"隐藏层偏置 b1 形状: {final_model.b1.shape}")
    print(f"隐藏层到输出层权重 W2 形状: {final_model.W2.shape}")
    print(f"输出层偏置 b2 形状: {final_model.b2.shape}")

    # 计算最终模型在全部数据上的准确率
    final_accuracy = accuracy_score(y, y_pred_all)

    return {
        'model': final_model,
        'X': X,
        'y': y,
        'target_names': target_names,
        'scaler': scaler,
        'avg_metrics': {
            'accuracy': avg_accuracy,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1
        },
        'std_metrics': {
            'accuracy': std_accuracy,
            'precision': std_precision,
            'recall': std_recall,
            'f1': std_f1
        },
        'final_accuracy': final_accuracy,
        'fold_results': fold_results
    }


if __name__ == "__main__":
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 运行主程序
    print("实验五：BP神经网络算法实现与测试")
    print("=" * 60)

    try:
        results = main()
        print("\n实验完成！")
        print("=" * 60)

        # 输出总结
        print("\n实验总结:")
        print(f"1. 数据集: Iris (150个样本, 3个类别)")
        print(f"2. 神经网络结构: 4-8-3 (输入层4个神经元, 隐藏层8个神经元, 输出层3个神经元)")
        print(f"3. 五折交叉验证平均准确率: {results['avg_metrics']['accuracy']:.4f}")
        print(f"4. 最终模型在全部数据上的准确率: {results['final_accuracy']:.4f}")
        print(f"5. 模型训练完成，可以用于新的数据预测")

        # 示例：使用模型进行预测
        print("\n" + "=" * 60)
        print("示例：使用训练好的模型进行预测")
        print("=" * 60)

        # 创建一个新的样本进行预测
        new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # setosa
        new_sample_scaled = results['scaler'].transform(new_sample)

        prediction_prob = results['model'].predict(new_sample_scaled)
        prediction_class = results['model'].predict_classes(new_sample_scaled)

        print(f"\n新样本特征: {new_sample[0]}")
        print(f"预测概率: {prediction_prob[0]}")
        print(f"预测类别: {results['target_names'][prediction_class[0]]}")
        print(f"类别索引: {prediction_class[0]}")

    except Exception as e:
        print(f"运行过程中出现错误: {e}")
        import traceback

        traceback.print_exc()