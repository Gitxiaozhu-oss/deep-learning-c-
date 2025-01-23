#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>

class NeuralNetwork {
private:
    // 网络结构参数
    std::vector<int> layers;          // 存储每层节点数
    double learning_rate;             // 学习率
    
    // 网络权重和偏置
    std::vector<std::vector<std::vector<double>>> weights;  // 权重矩阵
    std::vector<std::vector<double>> biases;               // 偏置向量
    
    // 存储中间结果
    std::vector<std::vector<double>> activations;  // 各层激活值
    std::vector<std::vector<double>> z_values;     // 各层带权输入

public:
    NeuralNetwork(const std::vector<int>& layer_sizes, double lr = 0.01) 
        : layers(layer_sizes), learning_rate(lr) {
        
        // 初始化权重和偏置
        initializeParameters();
    }

private:
    void initializeParameters() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0.0, 0.1);  // 使用正态分布初始化

        // 初始化权重矩阵
        weights.resize(layers.size() - 1);
        for (size_t i = 0; i < layers.size() - 1; ++i) {
            weights[i].resize(layers[i + 1]);
            for (auto& w : weights[i]) {
                w.resize(layers[i]);
                for (auto& val : w) {
                    val = d(gen);
                }
            }
        }

        // 初始化偏置向量
        biases.resize(layers.size() - 1);
        for (size_t i = 0; i < layers.size() - 1; ++i) {
            biases[i].resize(layers[i + 1], 0.0);
        }
    }

    // ReLU激活函数
    static double relu(double x) {
        return std::max(0.0, x);
    }

    // ReLU导数
    static double relu_derivative(double x) {
        return x > 0 ? 1.0 : 0.0;
    }

    // Softmax激活函数
    std::vector<double> softmax(const std::vector<double>& x) {
        std::vector<double> output(x.size());
        double max_val = *std::max_element(x.begin(), x.end());
        double sum = 0.0;

        for (size_t i = 0; i < x.size(); ++i) {
            output[i] = std::exp(x[i] - max_val);
            sum += output[i];
        }

        for (double& val : output) {
            val /= sum;
        }

        return output;
    }

public:
    // 前向传播
    std::vector<double> forward(const std::vector<double>& input) {
        activations.clear();
        z_values.clear();
        
        // 存储输入层
        activations.push_back(input);
        
        // 隐藏层前向传播
        for (size_t i = 0; i < weights.size() - 1; ++i) {
            std::vector<double> z(layers[i + 1], 0.0);
            
            // 计算带权输入
            for (size_t j = 0; j < weights[i].size(); ++j) {
                for (size_t k = 0; k < weights[i][j].size(); ++k) {
                    z[j] += weights[i][j][k] * activations.back()[k];
                }
                z[j] += biases[i][j];
            }
            
            z_values.push_back(z);
            
            // 应用ReLU激活函数
            std::vector<double> activation(z.size());
            for (size_t j = 0; j < z.size(); ++j) {
                activation[j] = relu(z[j]);
            }
            activations.push_back(activation);
        }
        
        // 输出层前向传播（使用Softmax）
        std::vector<double> output_z(layers.back(), 0.0);
        for (size_t j = 0; j < weights.back().size(); ++j) {
            for (size_t k = 0; k < weights.back()[j].size(); ++k) {
                output_z[j] += weights.back()[j][k] * activations.back()[k];
            }
            output_z[j] += biases.back()[j];
        }
        
        z_values.push_back(output_z);
        activations.push_back(softmax(output_z));
        
        return activations.back();
    }

    // 反向传播
    void backward(const std::vector<double>& target) {
        std::vector<std::vector<double>> deltas;
        
        // 计算输出层误差
        std::vector<double> output_delta(layers.back());
        for (size_t i = 0; i < output_delta.size(); ++i) {
            output_delta[i] = activations.back()[i] - target[i];
        }
        deltas.push_back(output_delta);
        
        // 计算隐藏层误差
        for (int i = layers.size() - 2; i > 0; --i) {
            std::vector<double> delta(layers[i], 0.0);
            
            for (size_t j = 0; j < layers[i]; ++j) {
                double error = 0.0;
                for (size_t k = 0; k < layers[i + 1]; ++k) {
                    error += deltas.front()[k] * weights[i][k][j];
                }
                delta[j] = error * relu_derivative(z_values[i - 1][j]);
            }
            deltas.insert(deltas.begin(), delta);
        }
        
        // 更新权重和偏置
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[i].size(); ++j) {
                for (size_t k = 0; k < weights[i][j].size(); ++k) {
                    weights[i][j][k] -= learning_rate * deltas[i][j] * 
                                      activations[i][k];
                }
                biases[i][j] -= learning_rate * deltas[i][j];
            }
        }
    }

    // 训练函数
    void train(const std::vector<std::vector<double>>& training_data,
              const std::vector<std::vector<double>>& targets,
              int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_loss = 0.0;
            
            for (size_t i = 0; i < training_data.size(); ++i) {
                // 前向传播
                auto output = forward(training_data[i]);
                
                // 计算交叉熵损失
                double loss = 0.0;
                for (size_t j = 0; j < output.size(); ++j) {
                    loss -= targets[i][j] * std::log(output[j] + 1e-10);
                }
                total_loss += loss;
                
                // 反向传播
                backward(targets[i]);
            }
            
            // 打印训练进度
            if ((epoch + 1) % 100 == 0) {
                std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                         << ", Loss: " << total_loss / training_data.size() 
                         << std::endl;
            }
        }
    }
};

#if 0
// 使用示例
int main() {
    // 创建一个简单的神经网络：2输入，3隐藏，2输出
    NeuralNetwork nn({2, 3, 2}, 0.1);
    
    // 准备训练数据（XOR问题）
    std::vector<std::vector<double>> training_data = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    std::vector<std::vector<double>> targets = {
        {1, 0}, {0, 1}, {0, 1}, {1, 0}
    };
    
    // 训练网络
    nn.train(training_data, targets, 1000);
    
    // 测试网络
    for (const auto& input : training_data) {
        auto output = nn.forward(input);
        std::cout << "Input: " << input[0] << ", " << input[1] 
                 << " Output: " << output[0] << ", " << output[1] << std::endl;
    }
    
    return 0;
}
#elif 1

int main() {
    // 创建一个更大的网络用于手写数字识别
    // 输入层784节点(28x28像素)，隐藏层128节点，输出层10节点(0-9数字)
    NeuralNetwork nn({784, 128, 10}, 0.01);
    
    // 加载训练数据（这里用随机数据模拟）
    const int num_samples = 1000;
    std::vector<std::vector<double>> training_data;
    std::vector<std::vector<double>> targets;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    
    // 生成模拟数据
    for (int i = 0; i < num_samples; ++i) {
        // 生成输入数据
        std::vector<double> input(784);
        for (auto& val : input) {
            val = dis(gen);
        }
        training_data.push_back(input);
        
        // 生成目标输出（one-hot编码）
        std::vector<double> target(10, 0.0);
        target[i % 10] = 1.0;  // 模拟0-9的标签
        targets.push_back(target);
    }
    
    // 训练网络
    std::cout << "开始训练..." << std::endl;
    nn.train(training_data, targets, 100);
    
    // 测试网络
    std::cout << "\n测试结果：" << std::endl;
    for (int i = 0; i < 5; ++i) {  // 测试前5个样本
        auto output = nn.forward(training_data[i]);
        
        // 找出预测的数字（最大概率的索引）
        int predicted = std::max_element(output.begin(), output.end()) 
                       - output.begin();
        
        std::cout << "样本 " << i << " 预测结果: " << predicted << std::endl;
    }
}
#else
int main() {
    // 创建自定义结构的神经网络
    // 例如：4个输入特征，6个隐藏节点，3个输出类别
    NeuralNetwork nn({4, 6, 3}, 0.05);

    // 准备训练数据
    std::vector<std::vector<double>> training_data = {
        {1.0, 2.0, 3.0, 4.0},
        {2.0, 3.0, 4.0, 5.0},
        {3.0, 4.0, 5.0, 6.0},
        // ... 更多训练样本
    };

    // 准备对应的目标输出
    std::vector<std::vector<double>> targets = {
        {1.0, 0.0, 0.0},  // 类别1
        {0.0, 1.0, 0.0},  // 类别2
        {0.0, 0.0, 1.0},  // 类别3
        // ... 对应的目标输出
    };

    // 训练参数设置
    const int epochs = 500;

    // 训练网络
    std::cout << "开始训练..." << std::endl;
    nn.train(training_data, targets, epochs);

    // 使用训练好的网络进行预测
    std::vector<double> new_input = {1.5, 2.5, 3.5, 4.5};
    auto prediction = nn.forward(new_input);

    // 输出预测结果
    std::cout << "预测结果: ";
    for (const auto& val : prediction) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}
#endif
