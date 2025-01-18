#include <iostream>
#include <vector>
#include <cmath>

// 定义LinearRegression类，包含训练数据集、学习率、迭代次数等参数
class LinearRegression {
private:
    std::vector<double> weights; // 权重向量
    double learningRate;         // 学习率
    int iterations;              // 迭代次数

public:
    // 构造函数，初始化学习率和迭代次数
    LinearRegression(double lr, int iters) : learningRate(lr), iterations(iters) {}

    // 训练模型的方法，输入为特征矩阵X和标签向量Y，使用梯度下降法更新权重
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& Y) {
        int n_samples = X.size();       // 样本数量
        int n_features = X[0].size();   // 特征数量
        weights.resize(n_features + 1, 0.0); // 初始化权重为零

        for (int iter = 0; iter < iterations; ++iter) {
            double interceptSum = 0.0;
            std::vector<double> featureSums(n_features, 0.0);

            for (int i = 0; i < n_samples; ++i) {
                double predicted = predictSample(X[i]); // 预测值
                double error = predicted - Y[i];        // 误差
                interceptSum += error;
                for (int j = 0; j < n_features; ++j) {
                    featureSums[j] += error * X[i][j];
                }
            }

            // 更新截距和权重
            weights[0] -= (learningRate / n_samples) * interceptSum;
            for (int j = 0; j < n_features; ++j) {
                weights[j + 1] -= (learningRate / n_samples) * featureSums[j];
            }
        }
    }

    // 预测单个样本的输出值
    double predictSample(const std::vector<double>& x) {
        double prediction = weights[0]; // 初始预测值为截距
        for (size_t j = 0; j < x.size(); ++j) {
            prediction += weights[j + 1] * x[j]; // 加上每个特征对应的权重乘积
        }
        return prediction;
    }

    // 预测多个样本的输出值
    std::vector<double> predict(const std::vector<std::vector<double>>& X) {
        std::vector<double> predictions;
        for (const auto& sample : X) {
            predictions.push_back(predictSample(sample)); // 对每个样本进行预测
        }
        return predictions;
    }
};

int main() {
    // 示例数据集：特征矩阵X和标签向量Y
    //std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0}};
    std::vector<std::vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0},{5.0},{6.0}};
    //std::vector<double> Y = {2.0, 4.0, 6.0, 8.0};
    std::vector<double> Y = {2.0, 4.0, 6.0, 8.0,10.0,12.0};

    // 创建一个线性回归模型实例，设置学习率为0.001，迭代次数为10000
    LinearRegression model(0.001, 10000);

    // 使用示例数据集训练模型
    model.fit(X, Y);

    // 新的数据点用于测试预测功能
    std::vector<std::vector<double>> newX = {{5.0}, {6.0}};

    // 对新数据点进行预测
    std::vector<double> predictions = model.predict(newX);

    // 打印预测结果
    std::cout << "预测结果:" << std::endl;
    for (double pred : predictions) {
        std::cout << pred << std::endl;
    }

    return 0;
}




