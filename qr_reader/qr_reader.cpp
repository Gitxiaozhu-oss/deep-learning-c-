#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>

using namespace cv;
using namespace std;

class QRCodeReader {
private:
    VideoCapture cap;                    // 视频捕获对象
    const string window_name = "二维码识别";  // 窗口名称
    atomic<bool> is_running{true};       // 程序运行状态
    
    // 图像处理参数
    const Size PROCESS_SIZE = Size(640, 480);  // 处理分辨率
    QRCodeDetector qr_detector;          // 二维码检测器
    
    // 缓冲队列和互斥锁
    queue<Mat> frame_buffer;             // 帧缓冲队列
    queue<pair<Mat, string>> display_buffer;  // 显示缓冲队列
    mutex buffer_mutex;                  // 缓冲区互斥锁
    mutex display_mutex;                 // 显示缓冲区互斥锁
    const size_t MAX_BUFFER_SIZE = 2;    // 最大缓冲区大小

    // 线程对象
    thread capture_thread;               // 图像捕获线程
    thread process_thread;               // 图像处理线程

public:
    QRCodeReader() {
        initializeCamera();
        startThreads();
    }

    bool initializeCamera() {
        cout << "正在初始化摄像头..." << endl;
        
        // 打开默认摄像头
        cap.open(0);
        if (!cap.isOpened()) {
            throw runtime_error("无法打开摄像头");
        }

        // 设置摄像头参数以获得最佳性能
        cap.set(CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(CAP_PROP_FRAME_HEIGHT, 720);
        cap.set(CAP_PROP_FPS, 30);
        cap.set(CAP_PROP_BUFFERSIZE, 1);
        cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
        
        cout << "摄像头初始化成功！" << endl;
        return true;
    }

    void startThreads() {
        // 启动图像捕获和处理线程
        capture_thread = thread(&QRCodeReader::captureLoop, this);
        process_thread = thread(&QRCodeReader::processLoop, this);
    }

    void captureLoop() {
        Mat frame;
        while (is_running) {
            // 从摄像头读取一帧
            if (!cap.read(frame)) {
                continue;
            }

            // 将帧放入缓冲区
            {
                lock_guard<mutex> lock(buffer_mutex);
                if (frame_buffer.size() >= MAX_BUFFER_SIZE) {
                    frame_buffer.pop();
                }
                frame_buffer.push(frame.clone());
            }
        }
    }

    void processLoop() {
        while (is_running) {
            Mat frame;
            // 从缓冲区获取一帧
            {
                lock_guard<mutex> lock(buffer_mutex);
                if (!frame_buffer.empty()) {
                    frame = frame_buffer.front();
                    frame_buffer.pop();
                }
            }
            
            if (!frame.empty()) {
                Mat processed = frame.clone();
                string qr_data;
                
                try {
                    // 调整图像大小以提高处理速度
                    Mat small_frame;
                    resize(frame, small_frame, PROCESS_SIZE);
                    
                    // 转换为灰度图像
                    Mat gray;
                    cvtColor(small_frame, gray, COLOR_BGR2GRAY);
                    
                    // 检测二维码
                    vector<Point> points;
                    qr_data = qr_detector.detectAndDecode(gray, points);
                    
                    if (!qr_data.empty()) {
                        // 调整检测到的点的坐标以匹配原始图像大小
                        float scale_x = float(frame.cols) / PROCESS_SIZE.width;
                        float scale_y = float(frame.rows) / PROCESS_SIZE.height;
                        
                        vector<Point> scaled_points;
                        for (const auto& point : points) {
                            scaled_points.push_back(Point(
                                point.x * scale_x,
                                point.y * scale_y
                            ));
                        }
                        
                        // 绘制二维码边界
                        for (size_t i = 0; i < scaled_points.size(); i++) {
                            line(processed, 
                                 scaled_points[i], 
                                 scaled_points[(i+1)%scaled_points.size()],
                                 Scalar(0, 255, 0), 3);
                        }
                        
                        cout << "检测到二维码：" << qr_data << endl;
                    }
                    
                } catch (const Exception& e) {
                    cout << "处理图像时出错: " << e.what() << endl;
                }
                
                // 将处理后的帧放入显示缓冲区
                lock_guard<mutex> lock(display_mutex);
                display_buffer.push(make_pair(processed, qr_data));
            }
        }
    }

    void run() {
        cout << "\n二维码识别程序使用说明：" << endl;
        cout << "1. 将二维码对准摄像头" << endl;
        cout << "2. 程序会自动识别并显示二维码内容" << endl;
        cout << "3. 按ESC键退出程序" << endl;
        
        // 在主线程中创建窗口
        namedWindow(window_name, WINDOW_NORMAL);
        
        // 主循环处理显示
        while (is_running) {
            Mat frame;
            string qr_data;
            
            // 从显示缓冲区获取处理后的帧
            {
                lock_guard<mutex> lock(display_mutex);
                if (!display_buffer.empty()) {
                    tie(frame, qr_data) = display_buffer.front();
                    display_buffer.pop();
                }
            }
            
            if (!frame.empty()) {
                // 在图像上显示二维码内容
                if (!qr_data.empty()) {
                    putText(frame, "QR Code: " + qr_data,
                            Point(10, frame.rows - 20),
                            FONT_HERSHEY_SIMPLEX, 0.8,
                            Scalar(0, 255, 0), 2);
                }
                
                // 显示图像
                imshow(window_name, frame);
            }
            
            // 处理键盘输入
            char key = waitKey(1);
            if (key == 27) { // ESC键退出
                is_running = false;
                break;
            }
        }
        
        // 等待线程结束
        if (process_thread.joinable()) {
            process_thread.join();
        }
        if (capture_thread.joinable()) {
            capture_thread.join();
        }
    }

    ~QRCodeReader() {
        // 清理资源
        is_running = false;
        
        if (capture_thread.joinable()) {
            capture_thread.join();
        }
        if (process_thread.joinable()) {
            process_thread.join();
        }
        
        if (cap.isOpened()) {
            cap.release();
        }
        destroyAllWindows();
    }
};

int main() {
    try {
        cout << "正在初始化二维码识别程序..." << endl;
        QRCodeReader reader;
        reader.run();
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
