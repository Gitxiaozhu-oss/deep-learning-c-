#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

using namespace cv;
using namespace std;

class HandDetector {
private:
    VideoCapture cap;
    const string window_name = "Hand Detection";
    bool camera_initialized = false;
    
    // 图像处理参数
    const Size BLUR_SIZE = Size(5, 5);
    const Size MORPH_SIZE = Size(3, 3);
    
    // 肤色检测的HSV阈值
    const Scalar HSV_MIN = Scalar(0, 30, 60);
    const Scalar HSV_MAX = Scalar(20, 150, 255);
    
    // 手部检测参数
    const double MIN_CONTOUR_AREA = 5000;
    const double MIN_DEFECT_DEPTH = 20.0;
    const double MAX_DEFECT_ANGLE = 90.0;

public:
    HandDetector() {
        initializeCamera();
    }

    bool initializeCamera() {
        // 尝试多次打开摄像头
        for (int i = 0; i < 3; i++) {
            cout << "尝试打开摄像头... 第 " << (i+1) << " 次" << endl;
            
            cap.open(0);
            if (cap.isOpened()) {
                // 设置摄像头参数
                cap.set(CAP_PROP_FRAME_WIDTH, 640);
                cap.set(CAP_PROP_FRAME_HEIGHT, 480);
                cap.set(CAP_PROP_FPS, 30);
                
                // 读取一帧测试摄像头是否正常工作
                Mat test_frame;
                cap.read(test_frame);
                if (!test_frame.empty()) {
                    camera_initialized = true;
                    cout << "摄像头初始化成功！" << endl;
                    
                    // 创建显示窗口
                    namedWindow(window_name, WINDOW_AUTOSIZE);
                    return true;
                }
            }
            
            // 如果失败，等待一秒后重试
            cap.release();
            this_thread::sleep_for(chrono::seconds(1));
        }
        
        throw runtime_error("无法初始化摄像头，请检查摄像头权限和连接状态");
        return false;
    }

    // 图像预处理函数
    Mat preprocessImage(const Mat& frame) {
        Mat processed, hsv, mask;
        
        // 高斯模糊去噪
        GaussianBlur(frame, processed, BLUR_SIZE, 0);
        
        // 转换到HSV颜色空间
        cvtColor(processed, hsv, COLOR_BGR2HSV);
        
        // 肤色检测
        inRange(hsv, HSV_MIN, HSV_MAX, mask);
        
        // 形态学操作
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, MORPH_SIZE);
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);
        morphologyEx(mask, mask, MORPH_OPEN, kernel);
        
        return mask;
    }

    // 计算两点之间的角度
    double calculateAngle(const Point& p1, const Point& p2, const Point& p0) {
        double dx1 = p1.x - p0.x;
        double dy1 = p1.y - p0.y;
        double dx2 = p2.x - p0.x;
        double dy2 = p2.y - p0.y;
        
        return acos((dx1*dx2 + dy1*dy2) / 
                   sqrt((dx1*dx1 + dy1*dy1) * (dx2*dx2 + dy2*dy2))) * 180.0 / CV_PI;
    }

    // 检测手指数量
    int detectFingers(const Mat& mask, Mat& output) {
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        int max_idx = -1;
        double max_area = 0;
        
        // 找到最大的轮廓（假设是手）
        for (int i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            if (area > max_area && area > MIN_CONTOUR_AREA) {
                max_area = area;
                max_idx = i;
            }
        }
        
        if (max_idx < 0) return 0;
        
        // 获取手的轮廓
        vector<Point>& hand_contour = contours[max_idx];
        
        // 计算凸包
        vector<int> hull_indices;
        vector<Point> hull_points;
        convexHull(hand_contour, hull_indices);
        convexHull(hand_contour, hull_points);
        
        // 计算凸缺陷
        vector<Vec4i> defects;
        if (hull_indices.size() > 3) {
            convexityDefects(hand_contour, hull_indices, defects);
        }
        
        // 计算手掌中心
        Moments m = moments(hand_contour);
        Point palm_center(m.m10/m.m00, m.m01/m.m00);
        
        // 绘制手的轮廓和凸包
        drawContours(output, vector<vector<Point>>{hand_contour}, -1, Scalar(0, 255, 0), 2);
        drawContours(output, vector<vector<Point>>{hull_points}, -1, Scalar(0, 0, 255), 2);
        
        // 统计手指数量
        int finger_count = 0;
        vector<Point> finger_tips;
        
        for (const Vec4i& defect : defects) {
            Point start = hand_contour[defect[0]];
            Point end = hand_contour[defect[1]];
            Point far = hand_contour[defect[2]];
            double depth = defect[3] / 256.0;
            
            if (depth > MIN_DEFECT_DEPTH) {
                double angle = calculateAngle(start, end, far);
                if (angle < MAX_DEFECT_ANGLE) {
                    finger_tips.push_back(start);
                    circle(output, start, 8, Scalar(255, 0, 0), -1);  // 指尖
                    line(output, start, palm_center, Scalar(0, 255, 255), 2);  // 连接线
                }
            }
        }
        
        // 手指数量等于检测到的指尖数量加1
        finger_count = finger_tips.size() + 1;
        if (finger_count > 5) finger_count = 5;  // 限制最大数量为5
        
        return finger_count;
    }

    void run() {
        if (!camera_initialized) {
            cout << "摄像头未正确初始化，无法运行程序" << endl;
            return;
        }

        Mat frame, mask;
        int failed_frames = 0;
        const int MAX_FAILED_FRAMES = 10;

        while (true) {
            // 读取摄像头帧
            bool success = cap.read(frame);
            
            if (!success || frame.empty()) {
                failed_frames++;
                cout << "无法读取摄像头画面..." << endl;
                
                if (failed_frames >= MAX_FAILED_FRAMES) {
                    cout << "连续读取失败，尝试重新初始化摄像头..." << endl;
                    if (!initializeCamera()) {
                        break;
                    }
                    failed_frames = 0;
                }
                
                this_thread::sleep_for(chrono::milliseconds(100));
                continue;
            }
            
            failed_frames = 0;  // 重置失败计数

            // 水平翻转（镜像效果）
            flip(frame, frame, 1);
            
            try {
                // 预处理
                mask = preprocessImage(frame);
                
                // 检测手指
                int fingers = detectFingers(mask, frame);
                
                // 显示手指数量
                string text = "Fingers: " + to_string(fingers);
                putText(frame, text, Point(30, 30), 
                        FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
                
                // 显示结果
                imshow(window_name, frame);
                
                // 显示预处理后的图像（调试用）
                imshow("Mask", mask);
            }
            catch (const exception& e) {
                cout << "处理图像时出错: " << e.what() << endl;
            }
            
            // 按'q'退出
            char key = waitKey(1);
            if (key == 'q' || key == 27) break;
        }
    }

    ~HandDetector() {
        if (cap.isOpened()) {
            cap.release();
        }
        destroyAllWindows();
    }
};

int main() {
    try {
        cout << "正在初始化手指检测程序..." << endl;
        HandDetector detector;
        
        cout << "开始手指检测，按'q'退出程序..." << endl;
        detector.run();
    }
    catch (const exception& e) {
        cerr << "错误: " << e.what() << endl;
        cout << "请确保：" << endl;
        cout << "1. 摄像头已正确连接" << endl;
        cout << "2. 已授予程序访问摄像头的权限" << endl;
        cout << "3. 没有其他程序正在使用摄像头" << endl;
        return -1;
    }
    
    cout << "程序正常结束" << endl;
    return 0;
}
