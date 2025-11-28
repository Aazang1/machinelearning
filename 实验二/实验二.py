import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate, KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class LogisticRegressionExperiment:
    """逻辑回归算法实现与测试实验"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.X = None
        self.y = None
        self.feature_names = None
        self.target_names = None
        self.df = None
        self.model = None
        self.cv_results = None

    def load_and_analyze_iris(self):
        """
        (1) 从scikit-learn库加载iris数据集并进行数据分析
        """
        print("=" * 60)
        print("步骤1: Iris数据集加载与数据分析")
        print("=" * 60)

        # 加载数据集
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names

        print("数据集加载成功!")
        print(f"特征数据形状: {self.X.shape}")
        print(f"目标变量形状: {self.y.shape}")
        print(f"特征名称: {self.feature_names}")
        print(f"目标类别: {list(self.target_names)}")
        print(f"样本分布: {np.bincount(self.y)} - {list(self.target_names)}")

        # 创建DataFrame用于数据分析
        self.df = pd.DataFrame(self.X, columns=self.feature_names)
        self.df['target'] = self.y
        self.df['class'] = self.df['target'].map({i: name for i, name in enumerate(self.target_names)})

        # 基本统计分析
        print("\n数据集前5行:")
        print(self.df.head())

        print("\n数据集基本信息:")
        print(self.df.describe())

        # 数据可视化分析
        self._visualize_data()

        return self.X, self.y, self.feature_names, self.target_names, self.df

    def _visualize_data(self):
        """数据可视化分析"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 特征分布直方图
        for i, feature in enumerate(self.feature_names):
            row, col = i // 2, i % 2
            for target_class in range(3):
                data = self.df[self.df['target'] == target_class][feature]
                axes[row, col].hist(data, alpha=0.7, label=self.target_names[target_class])
            axes[row, col].set_title(f'{feature}分布')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 散点矩阵图
        plt.figure(figsize=(12, 10))
        scatter_matrix = pd.plotting.scatter_matrix(
            self.df[self.feature_names],
            c=self.df['target'],
            figsize=(12, 10),
            marker='o',
            alpha=0.8,
            cmap='viridis'
        )
        plt.suptitle('特征散点矩阵图', y=0.95, fontsize=16)
        plt.show()

    def five_fold_cross_validation(self):
        """
        (2) 五折交叉验证训练逻辑回归模型
        """
        print("\n" + "=" * 60)
        print("步骤2: 五折交叉验证训练逻辑回归模型")
        print("=" * 60)

        # 创建逻辑回归模型（对数几率回归）
        self.model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,  # 确保收敛
            multi_class='multinomial',  # 多分类问题
            solver='lbfgs'  # 适用于多分类的优化算法
        )

        # 定义五折交叉验证
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)

        # 定义评估指标
        scoring = {
            'accuracy': 'accuracy',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro',
            'f1_macro': 'f1_macro'
        }

        # 执行交叉验证
        self.cv_results = cross_validate(
            self.model, self.X, self.y,
            cv=kf,
            scoring=scoring,
            return_train_score=True,
            return_estimator=True
        )

        print("五折交叉验证完成!")
        return self.cv_results, self.model

    def evaluate_model_performance(self):
        """
        (3) 使用五折交叉验证评估模型性能
        """
        print("\n" + "=" * 60)
        print("步骤3: 模型性能评估（五折交叉验证结果）")
        print("=" * 60)

        # 打印详细的交叉验证结果
        self._print_cross_validation_results()

        # 独立测试集验证
        self._independent_test_set_validation()

        # 模型系数分析
        self._analyze_model_coefficients()

    def _print_cross_validation_results(self):
        """打印交叉验证详细结果"""
        print("\n五折交叉验证统计结果（均值 ± 标准差）:")
        print("-" * 50)

        metrics = {
            '训练准确度': ('train_accuracy', '交叉验证训练集准确度'),
            '测试准确度': ('test_accuracy', '交叉验证测试集准确度'),
            '测试精度': ('test_precision_macro', '宏平均精度'),
            '测试召回率': ('test_recall_macro', '宏平均召回率'),
            '测试F1值': ('test_f1_macro', '宏平均F1分数')
        }

        for name, (key, desc) in metrics.items():
            mean_val = np.mean(self.cv_results[key])
            std_val = np.std(self.cv_results[key])
            print(f"{name}: {mean_val:.4f} (±{std_val:.4f}) - {desc}")

        # 各折详细结果
        print("\n各折详细结果:")
        print("折号\t训练准确度\t测试准确度\t测试精度\t测试召回率\t测试F1值")
        print("-" * 80)
        for i in range(5):
            print(f"{i + 1}\t"
                  f"{self.cv_results['train_accuracy'][i]:.4f}\t\t"
                  f"{self.cv_results['test_accuracy'][i]:.4f}\t\t"
                  f"{self.cv_results['test_precision_macro'][i]:.4f}\t\t"
                  f"{self.cv_results['test_recall_macro'][i]:.4f}\t\t"
                  f"{self.cv_results['test_f1_macro'][i]:.4f}")

    def _independent_test_set_validation(self):
        """独立测试集验证"""
        print("\n" + "-" * 50)
        print("独立测试集验证结果")
        print("-" * 50)

        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=self.random_state, stratify=self.y
        )

        # 训练模型
        model = LogisticRegression(random_state=self.random_state, max_iter=1000, multi_class='multinomial')
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        print(f"准确度 (Accuracy): {accuracy:.4f}")
        print(f"精度 (Precision-macro): {precision:.4f}")
        print(f"召回率 (Recall-macro): {recall:.4f}")
        print(f"F1值 (F1-macro): {f1:.4f}")

        # 各类别详细指标
        print("\n各类别详细指标:")
        precision_per_class = precision_score(y_test, y_pred, average=None)
        recall_per_class = recall_score(y_test, y_pred, average=None)
        f1_per_class = f1_score(y_test, y_pred, average=None)

        print("类别\t\t精度\t\t召回率\t\tF1值")
        print("-" * 45)
        for i, class_name in enumerate(self.target_names):
            print(f"{class_name:<12}\t{precision_per_class[i]:.4f}\t\t"
                  f"{recall_per_class[i]:.4f}\t\t{f1_per_class[i]:.4f}")

        # 混淆矩阵
        self._plot_confusion_matrix(y_test, y_pred)

        # 分类报告
        print(f"\n详细分类报告:")
        print(classification_report(y_test, y_pred, target_names=self.target_names))

    def _plot_confusion_matrix(self, y_true, y_pred):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.target_names, yticklabels=self.target_names)
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.show()

    def _analyze_model_coefficients(self):
        """分析模型系数（特征重要性）"""
        print("\n" + "-" * 50)
        print("逻辑回归模型系数分析")
        print("-" * 50)

        # 使用第一个折的模型进行分析
        model = self.cv_results['estimator'][0]

        # 多分类问题的系数矩阵
        coefficients = model.coef_
        intercepts = model.intercept_

        print("各类别的特征系数:")
        print("特征\t\t" + "\t".join(self.target_names))
        print("-" * 50)
        for i, feature in enumerate(self.feature_names):
            coef_str = "\t\t".join([f"{coef:.4f}" for coef in coefficients[:, i]])
            print(f"{feature:<15}\t{coef_str}")

        print(f"\n截距项: {intercepts}")

        # 特征重要性可视化
        self._plot_feature_importance(coefficients)

    def _plot_feature_importance(self, coefficients):
        """绘制特征重要性图"""
        # 计算特征重要性（系数绝对值均值）
        importance = np.mean(np.abs(coefficients), axis=0)

        plt.figure(figsize=(10, 6))
        bars = plt.barh(self.feature_names, importance)
        plt.xlabel('特征系数绝对值（重要性）')
        plt.title('逻辑回归特征重要性分析')
        plt.gca().invert_yaxis()

        # 在条形上添加数值
        for bar, imp in zip(bars, importance):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{imp:.4f}', ha='left', va='center')

        plt.tight_layout()
        plt.show()

    def performance_comparison_analysis(self):
        """
        (4) 性能比较分析
        """
        print("\n" + "=" * 60)
        print("步骤4: 模型性能比较分析")
        print("=" * 60)

        # 计算平均性能指标
        avg_accuracy = np.mean(self.cv_results['test_accuracy'])
        avg_precision = np.mean(self.cv_results['test_precision_macro'])
        avg_recall = np.mean(self.cv_results['test_recall_macro'])
        avg_f1 = np.mean(self.cv_results['test_f1_macro'])

        # 性能分析
        print("性能分析总结:")
        print(f"✅ 平均准确度: {avg_accuracy:.4f}")
        print(f"✅ 平均精度: {avg_precision:.4f}")
        print(f"✅ 平均召回率: {avg_recall:.4f}")
        print(f"✅ 平均F1分数: {avg_f1:.4f}")

        # 稳定性分析（标准差）
        std_accuracy = np.std(self.cv_results['test_accuracy'])
        print(f"📊 准确度稳定性（标准差）: {std_accuracy:.4f}")

        # 性能评估
        if avg_accuracy > 0.95:
            rating = "优秀"
        elif avg_accuracy > 0.90:
            rating = "良好"
        elif avg_accuracy > 0.85:
            rating = "一般"
        else:
            rating = "需要改进"

        print(f"📈 模型性能评级: {rating}")

        # 过拟合分析
        train_accuracy = np.mean(self.cv_results['train_accuracy'])
        overfitting_gap = train_accuracy - avg_accuracy
        print(f"🔍 过拟合程度（训练-测试差距）: {overfitting_gap:.4f}")

        if overfitting_gap < 0.02:
            print("✅ 模型泛化能力良好，过拟合程度较低")
        elif overfitting_gap < 0.05:
            print("⚠️  存在轻微过拟合")
        else:
            print("❌ 过拟合较明显，需要考虑正则化")

    def run_complete_experiment(self):
        """运行完整实验"""
        print("=" * 70)
        print("上机实验二：逻辑回归算法实现与测试")
        print("=" * 70)

        # (1) 数据加载与分析
        self.load_and_analyze_iris()

        # (2) 五折交叉验证训练
        self.five_fold_cross_validation()

        # (3) 模型性能评估
        self.evaluate_model_performance()

        # (4) 性能比较分析
        self.performance_comparison_analysis()

        print("\n" + "=" * 70)
        print("实验完成！")
        print("=" * 70)


# 运行实验
if __name__ == "__main__":
    experiment = LogisticRegressionExperiment(random_state=42)
    experiment.run_complete_experiment()