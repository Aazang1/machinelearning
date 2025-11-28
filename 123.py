import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate, KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler


class IrisRandomForest:
    def __init__(self, n_estimators=100, random_state=42):
        """初始化随机森林分类器"""
        self.rf_classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1
        )
        self.scaler = StandardScaler()

    def load_data(self):
        """加载Iris数据集"""
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names

        print("数据集信息:")
        print(f"样本数: {self.X.shape[0]}, 特征数: {self.X.shape[1]}")
        print(f"类别: {self.target_names}")
        return self.X, self.y

    def five_fold_cross_validation(self):
        """五折交叉验证"""
        # 定义评估指标
        scoring = {
            'accuracy': 'accuracy',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro',
            'f1_macro': 'f1_macro'
        }

        # 五折交叉验证
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(
            self.rf_classifier, self.X, self.y,
            cv=kf, scoring=scoring, return_train_score=True
        )

        return cv_results

    def train_and_evaluate(self):
        """训练模型并评估"""
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )

        # 训练模型
        self.rf_classifier.fit(X_train, y_train)

        # 预测
        y_pred = self.rf_classifier.predict(X_test)

        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        return accuracy, precision, recall, f1, y_test, y_pred

    def run_experiment(self):
        """运行完整实验"""
        # 加载数据
        self.load_data()

        # 五折交叉验证
        print("\n五折交叉验证结果:")
        cv_results = self.five_fold_cross_validation()

        print(f"平均准确率: {np.mean(cv_results['test_accuracy']):.4f}")
        print(f"平均精度: {np.mean(cv_results['test_precision_macro']):.4f}")
        print(f"平均召回率: {np.mean(cv_results['test_recall_macro']):.4f}")
        print(f"平均F1值: {np.mean(cv_results['test_f1_macro']):.4f}")

        # 详细评估
        print("\n详细评估结果:")
        accuracy, precision, recall, f1, y_test, y_pred = self.train_and_evaluate()

        print(f"准确率: {accuracy:.4f}")
        print(f"精度: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1值: {f1:.4f}")

        # 分类报告
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=self.target_names))

        return accuracy, precision, recall, f1


# 运行实验
if __name__ == "__main__":
    experiment = IrisRandomForest()
    accuracy, precision, recall, f1 = experiment.run_experiment()