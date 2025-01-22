#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <deque>

using namespace cv;
using namespace std;

class BlinkDetector {
private:
    VideoCapture cap;
    CascadeClassifier face_cascade;
    CascadeClassifier eye_cascade;
    deque<double> eye_areas;  // 存储最近几帧的眼睛区域面积
    const int HISTORY_FRAMES = 3;  // 保存的历史帧数
    const double BLINK_THRESHOLD = 0.6;  // 眨眼判定阈值
    int cc = 0;
public:
    BlinkDetector() {
        // 加载预训练的分类器
        if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
            throw runtime_error("无法加载人脸检测模型");
        }
        if (!eye_cascade.load("haarcascade_eye.xml")) {
            throw runtime_error("无法加载眼睛检测模型");
        }
        
        // 初始化摄像头
        cap.open(0);  // 打开默认摄像头
        if (!cap.isOpened()) {
            throw runtime_error("无法打开摄像头");
        }
        
        // 设置摄像头参数
        cap.set(CAP_PROP_FRAME_WIDTH, 640);
        cap.set(CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(CAP_PROP_FPS, 30);
    }

    // 析构函数移到 public 部分
    ~BlinkDetector() {
        cap.release();
        destroyAllWindows();
    }

    void run() {
        Mat frame;
        while (true) {
            cap.read(frame);
            if (frame.empty()) {
                cerr << "无法获取视频帧" << endl;
                break;
            }

            // 处理当前帧
            processFrame(frame);

            // 显示结果
            imshow("Blink Detection", frame);

            // 检查是否按下'q'键退出
            if (waitKey(1) == 'q') {
                break;
            }
        }
    }

private:
    void processFrame(Mat& frame) {
        // 转换为灰度图
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        // 检测人脸
        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.3, 5);

        for (const auto& face : faces) {
            // 绘制人脸矩形
            rectangle(frame, face, Scalar(255, 0, 0), 2);

            // 提取人脸区域
            Mat face_roi = gray(face);

            // 检测眼睛
            vector<Rect> eyes;
            eye_cascade.detectMultiScale(face_roi, eyes, 1.1, 3);

            // 计算当前帧中所有眼睛区域的总面积
            double current_eye_area = 0;
            for (const auto& eye : eyes) {
                // 在人脸上绘制眼睛矩形
                Point eye_center(face.x + eye.x + eye.width/2, 
                               face.y + eye.y + eye.height/2);
                int radius = cvRound((eye.width + eye.height)*0.25);
                circle(frame, eye_center, radius, Scalar(0, 255, 0), 2);

                current_eye_area += eye.width * eye.height;
            }

            // 更新历史记录
            eye_areas.push_back(current_eye_area);
            if (eye_areas.size() > HISTORY_FRAMES) {
                eye_areas.pop_front();
            }

            // 检测眨眼
            if (eye_areas.size() == HISTORY_FRAMES) {
                detectBlink(frame);
            }
        }
    }

    void detectBlink(Mat& frame) {
        // 获取最近几帧的眼睛面积
        double prev_area = eye_areas[HISTORY_FRAMES - 2];
        double curr_area = eye_areas[HISTORY_FRAMES - 1];

        // 如果当前面积显著小于前一帧，可能发生了眨眼
        if (prev_area > 0 && curr_area / prev_area < BLINK_THRESHOLD) {
            // 在图像上显示眨眼提示
            putText(frame, "Blink Detected!", 
                    Point(10, 30), FONT_HERSHEY_SIMPLEX, 1,
                    Scalar(0, 0, 255), 2);
	    cc++;
            cout << "检测到眨眼次数："<<cc << endl;
        }
    }
};

int main() {
    try {
        BlinkDetector detector;
        detector.run();
    }
    catch (const exception& e) {
        cerr << "错误: " << e.what() << endl;
        return -1;
    }
    return 0;
}
