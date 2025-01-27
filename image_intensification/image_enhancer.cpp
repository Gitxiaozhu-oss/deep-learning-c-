#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <iomanip>

using namespace cv;
using namespace std;

class ImageEnhancer {
private:
    VideoCapture cap;
    const string window_name = "图像增强";
    atomic<bool> is_running{true};
    
    // 图像处理参数
    double brightness = 0.0;     // 亮度：-100 到 100
    double contrast = 1.0;       // 对比度：0.1 到 5.0
    mutex params_mutex;

    // 显示参数
    const int param_bar_width = 50;
    const char bar_char = '=';
    const char empty_char = '-';

public:
    ImageEnhancer() {
        initializeCamera();
    }

    bool initializeCamera() {
        cout << "正在初始化摄像头..." << endl;
        
        cap.open(0);
        if (!cap.isOpened()) {
            throw runtime_error("无法打开摄像头");
        }

        // 设置摄像头参数
        cap.set(CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(CAP_PROP_FRAME_HEIGHT, 720);
        cap.set(CAP_PROP_FPS, 30);
        cap.set(CAP_PROP_BUFFERSIZE, 1);
        cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
        
        cout << "摄像头初始化成功！" << endl;
        return true;
    }

    Mat processFrame(const Mat& frame) {
        Mat enhanced;
        lock_guard<mutex> lock(params_mutex);
        
        // 应用亮度和对比度调整
        frame.convertTo(enhanced, -1, contrast, brightness);
        return enhanced;
    }

    void displayParameters() {
        system("clear");  // Linux/macOS清屏，Windows使用system("cls")
        cout << "\n图像增强控制面板\n" << string(50, '=') << endl;
        
        lock_guard<mutex> lock(params_mutex);
        
        // 显示亮度条
        int brightness_pos = (brightness + 100) * param_bar_width / 200;
        cout << "亮度 [" << setw(6) << fixed << setprecision(1) << brightness << "]: [";
        for (int i = 0; i < param_bar_width; ++i) {
            cout << (i < brightness_pos ? bar_char : empty_char);
        }
        cout << "]" << endl;
        
        // 显示对比度条
        int contrast_pos = (contrast - 0.1) * param_bar_width / 4.9;
        cout << "对比度[" << setw(6) << contrast << "]: [";
        for (int i = 0; i < param_bar_width; ++i) {
            cout << (i < contrast_pos ? bar_char : empty_char);
        }
        cout << "]" << endl;
        
        cout << "\n控制说明：" << endl;
        cout << "A/D - 减少/增加亮度" << endl;
        cout << "W/S - 增加/减少对比度" << endl;
        cout << "R   - 重置参数" << endl;
        cout << "ESC - 退出程序" << endl;
    }

    void run() {
        namedWindow(window_name, WINDOW_NORMAL);
        displayParameters();

        Mat frame, enhanced;
        while (is_running) {
            if (!cap.read(frame)) {
                cout << "无法读取摄像头画面..." << endl;
                break;
            }

            enhanced = processFrame(frame);
            
            // 显示原始和增强后的图像
            Mat display;
            hconcat(frame, enhanced, display);
            
            putText(display, "Original", Point(10, 30), 
                    FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
            putText(display, "Enhanced", Point(frame.cols + 10, 30), 
                    FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
            
            imshow(window_name, display);

            // 处理键盘输入
            char key = waitKey(1);
            bool need_refresh = false;

            {
                lock_guard<mutex> lock(params_mutex);
                switch (key) {
                    case 'a': case 'A':  // 减少亮度
                        brightness = max(brightness - 5.0, -100.0);
                        need_refresh = true;
                        break;
                    case 'd': case 'D':  // 增加亮度
                        brightness = min(brightness + 5.0, 100.0);
                        need_refresh = true;
                        break;
                    case 's': case 'S':  // 减少对比度
                        contrast = max(contrast - 0.1, 0.1);
                        need_refresh = true;
                        break;
                    case 'w': case 'W':  // 增加对比度
                        contrast = min(contrast + 0.1, 5.0);
                        need_refresh = true;
                        break;
                    case 'r': case 'R':  // 重置参数
                        brightness = 0.0;
                        contrast = 1.0;
                        need_refresh = true;
                        break;
                    case 27:  // ESC键
                        is_running = false;
                        break;
                }
            }

            if (need_refresh) {
                displayParameters();
            }
        }
    }

    ~ImageEnhancer() {
        if (cap.isOpened()) {
            cap.release();
        }
        destroyAllWindows();
    }
};

int main() {
    try {
        cout << "正在初始化图像增强程序..." << endl;
        ImageEnhancer enhancer;
        enhancer.run();
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
