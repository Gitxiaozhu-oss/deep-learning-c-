#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <mutex>

using namespace cv;
using namespace std;

class BackgroundReplacer {
private:
    VideoCapture cap;
    Mat background;
    Ptr<BackgroundSubtractorMOG2> pMOG2;
    bool isInitialized;
    mutex frameMutex;
    
    struct Parameters {
        double learning_rate;
        int history;
        double var_threshold;
        bool detect_shadows;
        int morph_size;
        int blur_size;
        double min_contour_area;
        int gaussian_blur_size;    // 背景虚化参数
        double gaussian_sigma;     // 高斯模糊sigma值
        
        Parameters() : 
            learning_rate(0.0001),
            history(300),          // 减少历史帧数
            var_threshold(32),
            detect_shadows(false),
            morph_size(7),
            blur_size(3),
            min_contour_area(5000),
            gaussian_blur_size(45), // 背景虚化程度
            gaussian_sigma(0)       // 自动计算sigma
        {}
    } params;

    // 降低帧率
    const int TARGET_FPS = 15;  // 降低目标帧率
    const int FRAME_DELAY = 1000 / TARGET_FPS;

    // 背景虚化处理
    Mat blurBackground(const Mat& background) {
        Mat blurred;
        GaussianBlur(background, blurred, 
                     Size(params.gaussian_blur_size, params.gaussian_blur_size),
                     params.gaussian_sigma);
        return blurred;
    }

public:
    BackgroundReplacer() : isInitialized(false) {
        pMOG2 = createBackgroundSubtractorMOG2(
            params.history,
            params.var_threshold,
            params.detect_shadows
        );
    }

    bool initialize(int camera_index, const string& background_path) {
        cap.open(camera_index);
        if (!cap.isOpened()) {
            cout << "无法打开摄像头！" << endl;
            return false;
        }

        // 设置较低的摄像头分辨率以提高性能
        cap.set(CAP_PROP_FRAME_WIDTH, 640);
        cap.set(CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(CAP_PROP_FPS, TARGET_FPS);
        cap.set(CAP_PROP_BUFFERSIZE, 1);

        background = imread(background_path);
        if (background.empty()) {
            cout << "无法读取背景图片：" << background_path << endl;
            return false;
        }

        // 调整背景图片大小并进行虚化处理
        resize(background, background, Size(640, 480));
        background = blurBackground(background);
        
        isInitialized = true;
        return true;
    }

    Mat processFrame(const Mat& frame) {
        Mat result;
        
        try {
            // 预处理
            Mat preprocessed;
            GaussianBlur(frame, preprocessed, Size(3, 3), 0);

            // 背景分割
            Mat fgMask;
            {
                lock_guard<mutex> lock(frameMutex);
                pMOG2->apply(preprocessed, fgMask, params.learning_rate);
            }

            // 二值化处理
            threshold(fgMask, fgMask, 200, 255, THRESH_BINARY);

            // 形态学操作
            Mat kernel = getStructuringElement(MORPH_ELLIPSE, 
                                            Size(params.morph_size, params.morph_size));
            
            // 去噪和填充
            morphologyEx(fgMask, fgMask, MORPH_OPEN, kernel, Point(-1,-1), 1);
            morphologyEx(fgMask, fgMask, MORPH_CLOSE, kernel, Point(-1,-1), 2);

            // 轮廓处理
            vector<vector<Point>> contours;
            findContours(fgMask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            Mat mask = Mat::zeros(fgMask.size(), CV_8UC1);
            
            // 处理轮廓
            for (const auto& contour : contours) {
                double area = contourArea(contour);
                if (area > params.min_contour_area) {
                    vector<Point> hull;
                    convexHull(contour, hull);
                    fillPoly(mask, vector<vector<Point>>{hull}, Scalar(255));
                }
            }

            // 平滑边缘
            GaussianBlur(mask, mask, 
                        Size(params.blur_size * 2 + 1, params.blur_size * 2 + 1), 0);

            // 创建前景掩码
            Mat foregroundMask;
            threshold(mask, foregroundMask, 128, 255, THRESH_BINARY);

            // 边缘羽化
            Mat edgeMask;
            Canny(foregroundMask, edgeMask, 100, 200);
            dilate(edgeMask, edgeMask, kernel);
            GaussianBlur(edgeMask, edgeMask, Size(5, 5), 0);
            foregroundMask -= edgeMask;

            // 创建三通道掩码
            vector<Mat> channels = {foregroundMask, foregroundMask, foregroundMask};
            Mat foregroundMask3C;
            merge(channels, foregroundMask3C);

            // 提取前景
            Mat foreground = frame.clone();
            bitwise_and(foreground, foregroundMask3C, foreground);

            // 创建背景掩码
            Mat backgroundMask3C;
            bitwise_not(foregroundMask3C, backgroundMask3C);

            // 提取虚化背景
            Mat backgroundPart;
            bitwise_and(background, backgroundMask3C, backgroundPart);

            // 合并结果
            add(foreground, backgroundPart, result);

            // 添加轻微的整体模糊以平滑边缘
            GaussianBlur(result, result, Size(3, 3), 0);

        } catch (const Exception& e) {
            cerr << "处理帧时发生错误: " << e.what() << endl;
            return frame;
        }

        return result;
    }

    void run() {
        if (!isInitialized) {
            cout << "请先初始化！" << endl;
            return;
        }

        Mat frame, result;
        namedWindow("Background Replacement", WINDOW_NORMAL);
        
        int64 lastTime = 0;
        
        while (true) {
            // 帧率控制
            int64 currentTime = getTickCount();
            if (lastTime != 0) {
                int64 elapsed = (currentTime - lastTime) * 1000 / getTickFrequency();
                if (elapsed < FRAME_DELAY) {
                    waitKey(FRAME_DELAY - elapsed);
                }
            }
            lastTime = currentTime;

            {
                lock_guard<mutex> lock(frameMutex);
                cap >> frame;
            }
            
            if (frame.empty()) {
                break;
            }

            result = processFrame(frame);

            // 显示帧率
            double fps = getTickFrequency() / (getTickCount() - currentTime);
            putText(result, "FPS: " + to_string(int(fps)), 
                    Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0,
                    Scalar(0, 255, 0), 2);

            imshow("Background Replacement", result);
            
            char key = (char)waitKey(1);
            if (key == 'q' || key == 27) {
                break;
            }
        }
    }

    ~BackgroundReplacer() {
        if (cap.isOpened()) {
            cap.release();
        }
        destroyAllWindows();
    }
};

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "使用方法: " << argv[0] << " <背景图片路径>" << endl;
        return -1;
    }

    BackgroundReplacer replacer;
    if (!replacer.initialize(0, argv[1])) {
        return -1;
    }

    replacer.run();
    return 0;
}
