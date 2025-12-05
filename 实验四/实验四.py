import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class SimpleSMO:
    def __init__(self, C=1.0, tol=0.001, max_iter=1000, kernel='linear'):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.kernel_type = kernel
        self.alpha = None
        self.b = 0
        self.X = None
        self.y = None
        self.support_vectors_ = None
        self.support_vector_labels_ = None

    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2)

    def polynomial_kernel(self, x1, x2, degree=3):
        return (np.dot(x1, x2) + 1) ** degree

    def rbf_kernel(self, x1, x2, gamma=0.1):
        return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

    def kernel(self, x1, x2):
        if self.kernel_type == 'linear':
            return self.linear_kernel(x1, x2)
        elif self.kernel_type == 'poly':
            return self.polynomial_kernel(x1, x2)
        elif self.kernel_type == 'rbf':
            return self.rbf_kernel(x1, x2)
        else:
            return self.linear_kernel(x1, x2)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.X = X
        self.y = y
        self.alpha = np.zeros(n_samples)
        self.b = 0.0

        # 主循环
        iter_count = 0
        while iter_count < self.max_iter:
            alpha_changed = 0

            for i in range(n_samples):
                # 计算样本i的预测值和误差
                E_i = self.decision_function(X[i]) - y[i]

                # 检查KKT条件
                if ((y[i] * E_i < -self.tol) and (self.alpha[i] < self.C)) or \
                        ((y[i] * E_i > self.tol) and (self.alpha[i] > 0)):

                    # 随机选择另一个样本j
                    j = i
                    while j == i:
                        j = np.random.randint(0, n_samples)

                    E_j = self.decision_function(X[j]) - y[j]

                    alpha_i_old = self.alpha[i].copy()
                    alpha_j_old = self.alpha[j].copy()

                    # 计算L和H边界
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])

                    if L == H:
                        continue

                    # 计算eta
                    eta = 2.0 * self.kernel(X[i], X[j]) - \
                          self.kernel(X[i], X[i]) - self.kernel(X[j], X[j])

                    if eta >= 0:
                        continue

                    # 更新alpha_j
                    self.alpha[j] -= y[j] * (E_i - E_j) / eta

                    # 修剪alpha_j
                    if self.alpha[j] > H:
                        self.alpha[j] = H
                    elif self.alpha[j] < L:
                        self.alpha[j] = L

                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    # 更新alpha_i
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

                    # 更新偏置b
                    b1 = self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * \
                         self.kernel(X[i], X[i]) - y[j] * (self.alpha[j] - alpha_j_old) * \
                         self.kernel(X[i], X[j])
                    b2 = self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * \
                         self.kernel(X[i], X[j]) - y[j] * (self.alpha[j] - alpha_j_old) * \
                         self.kernel(X[j], X[j])

                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0

                    alpha_changed += 1

            if alpha_changed == 0:
                iter_count += 1
            else:
                iter_count = 0

        # 提取支持向量
        sv_index = self.alpha > 1e-5
        self.support_vectors_ = X[sv_index]
        self.support_vector_labels_ = y[sv_index]
        self.alpha = self.alpha[sv_index]

    def decision_function(self, x):
        result = 0.0
        for i in range(len(self.alpha)):
            result += self.alpha[i] * self.y[i] * self.kernel(self.X[i], x)
        return result + self.b

    def predict(self, X):
        return np.sign([self.decision_function(x) for x in X])


# 评估函数
def evaluate_svm(X, y, use_library=False):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accuracies, precisions, recalls, f1_scores = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if use_library:
            # 使用sklearn的SVM
            svm = SVC(kernel='linear', C=1.0)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
        else:
            # 使用自实现的SMO
            svm = SimpleSMO(C=1.0, max_iter=1000, kernel='linear')
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

        print(f"Fold {fold}: Accuracy = {acc:.4f}")

    return (np.mean(accuracies), np.mean(precisions),
            np.mean(recalls), np.mean(f1_scores))


# 主程序
def main():
    # 加载数据
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 标准化数据
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print("=" * 50)
    print("自实现SMO算法结果：")
    print("=" * 50)

    # 自实现SMO算法
    acc_manual, prec_manual, rec_manual, f1_manual = evaluate_svm(X, y, use_library=False)

    print(f"\n自实现SMO算法平均结果：")
    print(f"准确率: {acc_manual:.4f}")
    print(f"精度: {prec_manual:.4f}")
    print(f"召回率: {rec_manual:.4f}")
    print(f"F1值: {f1_manual:.4f}")

    print("\n" + "=" * 50)
    print("sklearn SVM库结果：")
    print("=" * 50)

    # sklearn SVM库
    acc_lib, prec_lib, rec_lib, f1_lib = evaluate_svm(X, y, use_library=True)

    print(f"\nsklearn SVM库平均结果：")
    print(f"准确率: {acc_lib:.4f}")
    print(f"精度: {prec_lib:.4f}")
    print(f"召回率: {rec_lib:.4f}")
    print(f"F1值: {f1_lib:.4f}")

    # 可视化结果对比
    metrics = ['准确率', '精度', '召回率', 'F1值']
    manual_scores = [acc_manual, prec_manual, rec_manual, f1_manual]
    lib_scores = [acc_lib, prec_lib, rec_lib, f1_lib]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, manual_scores, width, label='自实现SMO', alpha=0.8)
    plt.bar(x + width / 2, lib_scores, width, label='sklearn SVM', alpha=0.8)

    plt.xlabel('评估指标')
    plt.ylabel('分数')
    plt.title('SVM算法性能对比')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()