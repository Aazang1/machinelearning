import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import time
from sklearn.utils.multiclass import unique_labels

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 类别名称映射
CLASS_NAMES_CN = {
    'setosa': '山鸢尾',
    'versicolor': '杂色鸢尾',
    'virginica': '维吉尼亚鸢尾'
}

# 评估指标名称映射
METRIC_NAMES_CN = {
    'accuracy': '准确度',
    'precision': '精确率',
    'recall': '召回率',
    'f1': 'F1分数',
    'support': '支持数'
}


def chinese_classification_report(y_true, y_pred, target_names=None):
    """
    生成中文版分类报告
    """
    if target_names is None:
        target_names = [f'类别{i}' for i in range(len(np.unique(y_true)))]

    # 获取指标
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)

    # 转换为DataFrame
    df = pd.DataFrame(report).transpose()

    # 重命名列名为中文
    df = df.rename(columns={
        'precision': '精确率',
        'recall': '召回率',
        'f1-score': 'F1分数',
        'support': '支持数'
    })

    # 重命名索引为中文
    df.index = df.index.map(lambda x: CLASS_NAMES_CN.get(x, x) if x in CLASS_NAMES_CN else x)

    # 格式化数字显示
    for col in ['精确率', '召回率', 'F1分数']:
        df[col] = df[col].apply(lambda x: f'{x:.4f}' if isinstance(x, (int, float)) else x)

    return df


def evaluate_model_sklearn(y_true, y_pred, target_names=None, average='macro'):
    """
    使用scikit-learn库函数评估模型性能，返回中文结果
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    # 生成中文分类报告
    report_df = chinese_classification_report(y_true, y_pred, target_names)
    report_str = report_df.to_string(float_format=lambda x: f'{x:.4f}')

    return accuracy, precision, recall, f1, report_str, report_df


def plot_decision_boundary_sklearn(X, y, model, title="SVM决策边界", feature_names=None):
    """绘制决策边界（仅适用于二分类）"""
    if X.shape[1] != 2:
        print("警告：只能为二维特征绘制决策边界")
        return

    if len(np.unique(y)) > 2:
        print("警告：此可视化仅适用于二分类问题")
        return

    plt.figure(figsize=(10, 8))

    # 创建网格
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # 预测整个网格
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界和区域
    from matplotlib.colors import ListedColormap
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
    plt.contour(xx, yy, Z, colors='black', linewidths=0.5, alpha=0.5)

    # 绘制数据点
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=50,
                          edgecolor='black', cmap='coolwarm', alpha=0.8)

    # 标记支持向量
    if hasattr(model, 'support_vectors_'):
        plt.scatter(model.support_vectors_[:, 0],
                    model.support_vectors_[:, 1],
                    s=200, facecolors='none', edgecolors='gold',
                    linewidths=2, label='支持向量')

    if feature_names is not None and len(feature_names) >= 2:
        plt.xlabel(feature_names[0], fontsize=12)
        plt.ylabel(feature_names[1], fontsize=12)
    else:
        plt.xlabel('特征1', fontsize=12)
        plt.ylabel('特征2', fontsize=12)

    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter)
    plt.show()


def main():
    """主函数：使用scikit-learn的SVM进行实验"""

    print("=" * 60)
    print("实验：使用scikit-learn的SVM进行鸢尾花分类")
    print("=" * 60)

    # 1. 加载Iris数据集
    print("\n1. 加载Iris数据集并进行数据分析")
    print("-" * 40)

    iris = load_iris()
    X = iris.data
    y = iris.target

    # 将英文特征名转换为中文
    feature_names_en = iris.feature_names
    feature_names_cn = ['花萼长度(cm)', '花萼宽度(cm)', '花瓣长度(cm)', '花瓣宽度(cm)']

    # 中文类别名称
    target_names_cn = ['山鸢尾', '杂色鸢尾', '维吉尼亚鸢尾']
    target_names_en = iris.target_names

    print(f"数据集形状: {X.shape}")
    print(f"特征名称: {feature_names_cn}")
    print(f"类别名称: {target_names_cn}")
    print(f"各类别样本数: {np.bincount(y)}")

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. 五折交叉验证
    print("\n2. 执行五折交叉验证")
    print("-" * 40)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    detailed_reports = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
        print(f"\n正在处理第 {fold + 1} 折...")

        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 使用scikit-learn的SVM
        svm = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)

        # 评估模型
        accuracy, precision, recall, f1, report_str, report_df = evaluate_model_sklearn(
            y_test, y_pred, target_names=target_names_cn, average='macro'
        )

        fold_results = {
            'fold': fold + 1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_support_vectors': len(svm.support_vectors_),
            'support_vectors_per_class': svm.n_support_.tolist()
        }

        results.append(fold_results)
        detailed_reports.append(report_df)

        print(f"第 {fold + 1} 折结果:")
        print(f"  准确度: {accuracy:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  支持向量数: {len(svm.support_vectors_)}")
        print(f"  各类别支持向量数: {svm.n_support_}")

    # 3. 分析结果
    print("\n3. 模型性能分析")
    print("=" * 60)

    print("\n第5折详细分类报告：")
    print("-" * 40)
    print(detailed_reports[-1].to_string())

    # 计算平均性能
    print("\n各折交叉验证平均结果：")
    print("-" * 40)

    results_df = pd.DataFrame(results)

    for metric, metric_cn in zip(['accuracy', 'precision', 'recall', 'f1'],
                                 ['准确度', '精确率', '召回率', 'F1分数']):
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        print(f"{metric_cn}: {mean_val:.4f} (±{std_val:.4f})")

    print(
        f"\n平均支持向量数: {results_df['n_support_vectors'].mean():.1f} (±{results_df['n_support_vectors'].std():.1f})")

    # 4. 可视化结果
    print("\n4. 结果可视化")
    print("-" * 40)

    # 创建性能比较图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names_cn = ['准确度', '精确率', '召回率', 'F1分数']
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']

    for idx, (metric, metric_cn) in enumerate(zip(metrics, metric_names_cn)):
        ax = axes[idx // 2, idx % 2]
        values = results_df[metric].values
        box = ax.boxplot([values], labels=['SVM'], patch_artist=True)
        box['boxes'][0].set_facecolor(colors[idx])
        ax.scatter(np.ones_like(values), values, alpha=0.6, color='darkblue', s=50)
        ax.set_title(f'{metric_cn}分布', fontsize=12)
        ax.set_ylabel(metric_cn, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.8, 1.02)

    plt.suptitle('SVM五折交叉验证性能评估', fontsize=16)
    plt.tight_layout()
    plt.show()

    # 5. 不同核函数比较
    print("\n5. 不同核函数性能比较")
    print("-" * 40)

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    kernel_names_cn = ['线性核', '多项式核', 'RBF核', 'Sigmoid核']
    kernel_results = []

    for kernel, kernel_cn in zip(kernels, kernel_names_cn):
        print(f"\n测试核函数: {kernel_cn}")

        # 使用第一折数据进行快速测试
        train_idx, test_idx = list(kf.split(X_scaled))[0]
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if kernel == 'poly':
            svm = SVC(kernel=kernel, degree=3, C=1.0, random_state=42)
        else:
            svm = SVC(kernel=kernel, C=1.0, random_state=42)

        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        kernel_results.append({
            'kernel': kernel_cn,
            'kernel_en': kernel,
            'accuracy': accuracy,
            'f1': f1,
            'n_support_vectors': len(svm.support_vectors_)
        })

        print(f"  准确度: {accuracy:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  支持向量数: {len(svm.support_vectors_)}")

    # 绘制核函数比较图
    kernel_df = pd.DataFrame(kernel_results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 准确度比较
    bars1 = ax1.bar(kernel_df['kernel'], kernel_df['accuracy'],
                    color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    ax1.set_xlabel('核函数类型', fontsize=12)
    ax1.set_ylabel('准确度', fontsize=12)
    ax1.set_title('不同核函数的准确度比较', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=15)

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 支持向量数比较
    bars2 = ax2.bar(kernel_df['kernel'], kernel_df['n_support_vectors'],
                    color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    ax2.set_xlabel('核函数类型', fontsize=12)
    ax2.set_ylabel('支持向量数', fontsize=12)
    ax2.set_title('不同核函数的支持向量数', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=15)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{int(height)}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

    # 6. 决策边界可视化（使用前两个特征）
    print("\n6. 决策边界可视化（使用前两个特征）")
    print("-" * 40)

    # 选择两类数据进行可视化
    binary_classes = [0, 1]  # 使用前两个类别
    binary_mask = np.isin(y, binary_classes)
    X_binary = X_scaled[binary_mask, :2]  # 只使用前两个特征以便可视化
    y_binary = y[binary_mask]

    # 训练SVM
    print("训练线性SVM用于可视化...")
    svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
    svm_linear.fit(X_binary, y_binary)

    # 绘制决策边界
    plot_decision_boundary_sklearn(
        X_binary, y_binary, svm_linear,
        f"SVM决策边界 ({target_names_cn[binary_classes[0]]} vs {target_names_cn[binary_classes[1]]})",
        feature_names=feature_names_cn[:2]
    )

    # 7. 不同C参数的影响
    print("\n7. 正则化参数C对模型性能的影响")
    print("-" * 40)

    C_values = [0.01, 0.1, 1, 10, 100]
    C_names = ['0.01', '0.1', '1', '10', '100']
    accuracies = []
    n_sv_list = []

    for C, C_name in zip(C_values, C_names):
        svm = SVC(C=C, kernel='rbf', gamma='scale', random_state=42)

        # 使用第一折数据进行测试
        train_idx, test_idx = list(kf.split(X_scaled))[0]
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        n_sv = len(svm.support_vectors_)

        accuracies.append(accuracy)
        n_sv_list.append(n_sv)

        print(f"C={C_name}: 准确度 = {accuracy:.4f}, 支持向量数 = {n_sv}")

    # 绘制C参数影响图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 准确度 vs C
    ax1.plot(C_values, accuracies, 'bo-', linewidth=2, markersize=8)
    ax1.set_xscale('log')
    ax1.set_xlabel('正则化参数C (对数尺度)', fontsize=12)
    ax1.set_ylabel('准确度', fontsize=12)
    ax1.set_title('正则化参数C对准确度的影响', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(C_values)
    ax1.set_xticklabels(C_names)

    for i, (c, acc) in enumerate(zip(C_values, accuracies)):
        ax1.annotate(f'{acc:.3f}', (c, acc), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

    # 支持向量数 vs C
    ax2.plot(C_values, n_sv_list, 'ro-', linewidth=2, markersize=8)
    ax2.set_xscale('log')
    ax2.set_xlabel('正则化参数C (对数尺度)', fontsize=12)
    ax2.set_ylabel('支持向量数', fontsize=12)
    ax2.set_title('正则化参数C对支持向量的影响', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(C_values)
    ax2.set_xticklabels(C_names)

    for i, (c, n_sv) in enumerate(zip(C_values, n_sv_list)):
        ax2.annotate(f'{n_sv}', (c, n_sv), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()

    # 8. 多分类策略比较
    print("\n8. 多分类策略比较")
    print("-" * 40)

    strategies = ['ovr', 'ovo']
    strategy_names_cn = ['一对多(One-vs-Rest)', '一对一(One-vs-One)']
    strategy_results = []

    for strategy, strategy_cn in zip(strategies, strategy_names_cn):
        print(f"\n测试多分类策略: {strategy_cn}")

        # 使用第一折数据
        train_idx, test_idx = list(kf.split(X_scaled))[0]
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if strategy == 'ovr':
            svm = SVC(kernel='rbf', C=1.0, decision_function_shape='ovr', random_state=42)
        else:
            svm = SVC(kernel='rbf', C=1.0, decision_function_shape='ovo', random_state=42)

        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        strategy_results.append({
            'strategy': strategy_cn,
            'accuracy': accuracy,
            'f1': f1
        })

        print(f"  准确度: {accuracy:.4f}")
        print(f"  F1分数: {f1:.4f}")

    # 绘制多分类策略比较图
    strategy_df = pd.DataFrame(strategy_results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 准确度比较
    bars1 = ax1.bar(strategy_df['strategy'], strategy_df['accuracy'],
                    color=['skyblue', 'lightgreen'])
    ax1.set_xlabel('多分类策略', fontsize=12)
    ax1.set_ylabel('准确度', fontsize=12)
    ax1.set_title('不同多分类策略的准确度比较', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=15)

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    # F1分数比较
    bars2 = ax2.bar(strategy_df['strategy'], strategy_df['f1'],
                    color=['lightcoral', 'gold'])
    ax2.set_xlabel('多分类策略', fontsize=12)
    ax2.set_ylabel('F1分数', fontsize=12)
    ax2.set_title('不同多分类策略的F1分数比较', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=15)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

    # 9. 模型复杂度分析（修复版）
    print("\n9. 模型复杂度与泛化能力分析")
    print("-" * 40)

    # 使用不同的训练集大小
    train_sizes = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    train_size_names = ['40%', '50%', '60%', '70%', '80%', '90%']
    train_accuracies = []
    test_accuracies = []

    for size, size_name in zip(train_sizes, train_size_names):
        # 使用train_test_split来划分指定比例的训练集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, train_size=size, random_state=42, stratify=y
        )

        # 检查类别数
        unique_train = np.unique(y_train)
        unique_test = np.unique(y_test)

        if len(unique_train) < 2 or len(unique_test) < 2:
            print(f"训练集比例 {size_name}: 跳过（类别数不足）")
            continue

        svm = SVC(kernel='rbf', C=1.0, random_state=42)
        svm.fit(X_train, y_train)

        train_acc = svm.score(X_train, y_train)
        test_acc = svm.score(X_test, y_test)

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"训练集比例: {size_name}")
        print(f"  训练集大小: {len(X_train)}")
        print(f"  测试集大小: {len(X_test)}")
        print(f"  训练准确度: {train_acc:.4f}")
        print(f"  测试准确度: {test_acc:.4f}")
        print(f"  支持向量数: {len(svm.support_vectors_)}")
        print()

    # 绘制学习曲线
    if train_accuracies:
        fig, ax = plt.subplots(figsize=(10, 6))

        sizes_to_plot = train_sizes[:len(train_accuracies)]
        size_names_to_plot = train_size_names[:len(train_accuracies)]

        ax.plot(sizes_to_plot, train_accuracies, 'o-', linewidth=2, markersize=8,
                label='训练准确度', color='blue')
        ax.plot(sizes_to_plot, test_accuracies, 's-', linewidth=2, markersize=8,
                label='测试准确度', color='red')

        ax.set_xlabel('训练集比例', fontsize=12)
        ax.set_ylabel('准确度', fontsize=12)
        ax.set_title('SVM学习曲线', fontsize=14)
        ax.set_xticks(sizes_to_plot)
        ax.set_xticklabels(size_names_to_plot)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.8, 1.05)

        plt.tight_layout()
        plt.show()

    # 10. 快速参数调优演示
    print("\n10. 快速参数调优演示")
    print("-" * 40)

    # 使用较小的参数网格
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.1, 1]
    }

    print("正在进行快速网格搜索...")

    # 使用较小的数据集
    X_sample, _, y_sample, _ = train_test_split(X_scaled, y, train_size=0.7,
                                                random_state=42, stratify=y)

    grid_search = GridSearchCV(
        SVC(kernel='rbf', random_state=42),
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=1,
        verbose=0
    )

    start_time = time.time()
    grid_search.fit(X_sample, y_sample)
    end_time = time.time()

    print(f"网格搜索完成！耗时: {end_time - start_time:.2f}秒")
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

    # 在完整测试集上评估最佳模型
    _, X_test_full, _, y_test_full = train_test_split(X_scaled, y, test_size=0.3,
                                                      random_state=42, stratify=y)
    best_score = grid_search.score(X_test_full, y_test_full)
    print(f"最佳模型在测试集上的准确度: {best_score:.4f}")

    # 显示参数调优结果
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results = cv_results.sort_values('mean_test_score', ascending=False)

    print("\n参数调优结果（按平均测试分数排序）:")
    print("-" * 40)

    # 创建中文列名
    cv_results_display = cv_results[['param_C', 'param_gamma', 'mean_test_score', 'std_test_score']].copy()
    cv_results_display = cv_results_display.rename(columns={
        'param_C': '参数C',
        'param_gamma': '参数gamma',
        'mean_test_score': '平均准确度',
        'std_test_score': '准确度标准差'
    })

    # 格式化显示
    cv_results_display['平均准确度'] = cv_results_display['平均准确度'].apply(lambda x: f'{x:.4f}')
    cv_results_display['准确度标准差'] = cv_results_display['准确度标准差'].apply(lambda x: f'{x:.4f}')

    print(cv_results_display.head().to_string(index=False))

    # 11. 实验总结
    print("\n11. 实验总结")
    print("=" * 60)



    # 打印最佳模型详细信息
    print("\n最佳模型详细信息：")
    print("-" * 40)
    best_model = grid_search.best_estimator_

    # 将gamma参数值转换为可读格式
    gamma_value = best_model.gamma
    if gamma_value == 'scale':
        gamma_display = 'scale (1/(n_features * 方差))'
    elif gamma_value == 'auto':
        gamma_display = 'auto (1/n_features)'
    else:
        gamma_display = f'{gamma_value}'

    print(f"核函数: RBF核")
    print(f"正则化参数C: {best_model.C}")
    print(f"Gamma参数: {gamma_display}")
    print(f"支持向量数: {len(best_model.support_vectors_)}")

    # 显示各类别支持向量数
    print(f"各类别支持向量数:")
    for i, (class_name, n_sv) in enumerate(zip(target_names_cn, best_model.n_support_)):
        print(f"  {class_name}: {n_sv}")

    # 12. 特征重要性分析
    print("\n12. 特征重要性分析")
    print("-" * 40)

    # 训练线性SVM查看特征权重
    print("训练线性SVM以分析特征重要性：")

    # 使用全部数据训练线性SVM
    svm_linear_all = SVC(kernel='linear', C=1.0, random_state=42)
    svm_linear_all.fit(X_scaled, y)

    # 获取特征权重
    if hasattr(svm_linear_all, 'coef_'):
        # 对于多分类，coef_是一个矩阵
        coef_abs_mean = np.mean(np.abs(svm_linear_all.coef_), axis=0)

        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            '特征': feature_names_cn,
            '平均权重绝对值': coef_abs_mean
        }).sort_values('平均权重绝对值', ascending=False)

        print("\n特征重要性（平均权重绝对值，数值越大越重要）：")
        print(feature_importance.to_string(index=False))

        # 绘制特征重要性图
        plt.figure(figsize=(10, 6))
        bars = plt.barh(feature_importance['特征'], feature_importance['平均权重绝对值'],
                        color='skyblue')
        plt.xlabel('平均权重绝对值', fontsize=12)
        plt.title('SVM特征重要性（线性核）', fontsize=14)
        plt.grid(True, alpha=0.3, axis='x')

        # 在条形上添加数值
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{width:.3f}', ha='left', va='center', fontsize=10)

        plt.tight_layout()
        plt.show()

    print("\n" + "=" * 60)
    print("实验完成！感谢使用中文版SVM实验程序！")
    print("=" * 60)


if __name__ == "__main__":
    main()