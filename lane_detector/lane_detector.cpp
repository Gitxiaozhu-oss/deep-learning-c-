#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

class LaneDetector {
private:
    VideoCapture cap;        
    VideoWriter writer;      
    const string window_name = "Lane Detection";  
    
    // 调整参数
    const int ROI_HEIGHT_OFFSET = 300;    // 增加ROI高度偏移
    const int ROI_TOP_WIDTH = 150;        // 调整ROI顶部宽度
    const double DEVIATION_THRESHOLD = 30; 
    
    // 调整颜色阈值
    const Scalar YELLOW_MIN = Scalar(15, 80, 80);
    const Scalar YELLOW_MAX = Scalar(35, 255, 255);
    const int WHITE_THRESHOLD = 180;

public:
    LaneDetector(const string& input_path, const string& output_path) {
        cap.open(input_path);
        if (!cap.isOpened()) {
            throw runtime_error("Cannot open input video: " + input_path);
        }

        int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
        int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
        double fps = cap.get(CAP_PROP_FPS);

        writer.open(output_path, 
                   VideoWriter::fourcc('M','J','P','G'),
                   fps, 
                   Size(frame_width, frame_height),
                   true);

        if (!writer.isOpened()) {
            throw runtime_error("Cannot create output video: " + output_path);
        }

        namedWindow(window_name, WINDOW_AUTOSIZE);
    }

    Mat preprocess(const Mat& frame) {
        Mat processed, hsv, yellow_mask, white_mask;
        
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        inRange(hsv, YELLOW_MIN, YELLOW_MAX, yellow_mask);
        
        cvtColor(frame, processed, COLOR_BGR2GRAY);
        threshold(processed, white_mask, WHITE_THRESHOLD, 255, THRESH_BINARY);
        
        bitwise_or(yellow_mask, white_mask, processed);
        GaussianBlur(processed, processed, Size(5, 5), 0);
        Canny(processed, processed, 50, 150);
        
        return processed;
    }

    Mat extractROI(const Mat& frame) {
        int height = frame.rows;
        int width = frame.cols;
        
        vector<Point> points = {
            Point((width/2) - ROI_TOP_WIDTH, height - ROI_HEIGHT_OFFSET),
            Point((width/2) + ROI_TOP_WIDTH, height - ROI_HEIGHT_OFFSET),
            Point(width - 100, height),
            Point(100, height)
        };
        
        Mat mask = Mat::zeros(frame.size(), frame.type());
        fillConvexPoly(mask, points, Scalar(255));
        
        Mat roi;
        bitwise_and(frame, mask, roi);
        
        return roi;
    }

    vector<Vec4i> detectLanes(const Mat& processed) {
        vector<Vec4i> lines;
        HoughLinesP(processed, lines, 1, CV_PI/180, 
                   30, 40, 20);
        
        vector<Vec4i> filtered_lines;
        for (const auto& line : lines) {
            double angle = atan2(line[3] - line[1], line[2] - line[0]) * 180 / CV_PI;
            if ((angle > 20 && angle < 80) || (angle < -20 && angle > -80)) {
                filtered_lines.push_back(line);
            }
        }

        return filtered_lines;
    }

    double calculateDeviation(const vector<Vec4i>& lines, const Mat& frame) {
        if (lines.empty()) return 0.0;

        vector<Point2f> left_points, right_points;
        
        for (const auto& line : lines) {
            double angle = atan2(line[3] - line[1], line[2] - line[0]) * 180 / CV_PI;
            Point2f p1(line[0], line[1]);
            Point2f p2(line[2], line[3]);
            
            if (angle > 0) {
                right_points.push_back(p1);
                right_points.push_back(p2);
            } else {
                left_points.push_back(p1);
                left_points.push_back(p2);
            }
        }

        if (left_points.empty() || right_points.empty()) return 0.0;

        double left_x = 0, right_x = 0;
        for (const auto& p : left_points) left_x += p.x;
        for (const auto& p : right_points) right_x += p.x;
        
        left_x /= left_points.size();
        right_x /= right_points.size();

        double lane_center = (left_x + right_x) / 2.0;
        double image_center = frame.cols / 2.0;
        
        return lane_center - image_center;
    }

    void drawResults(Mat& frame, const vector<Vec4i>& lines, double deviation) {
        if (lines.empty()) return;

        vector<Point2f> left_points, right_points;
        
        // 分离左右车道线点
        for (const auto& line : lines) {
            double angle = atan2(line[3] - line[1], line[2] - line[0]) * 180 / CV_PI;
            Point2f p1(line[0], line[1]);
            Point2f p2(line[2], line[3]);
            
            if (angle > 0) {
                right_points.push_back(p1);
                right_points.push_back(p2);
            } else {
                left_points.push_back(p1);
                left_points.push_back(p2);
            }
        }

        if (left_points.empty() || right_points.empty()) return;

        // 找到左右车道线的端点
        float min_y = frame.rows, max_y = 0;
        float left_x_at_min_y = 0, left_x_at_max_y = 0;
        float right_x_at_min_y = 0, right_x_at_max_y = 0;

        // 找到最高和最低的y坐标
        for (const auto& p : left_points) {
            if (p.y < min_y) min_y = p.y;
            if (p.y > max_y) max_y = p.y;
        }
        for (const auto& p : right_points) {
            if (p.y < min_y) min_y = p.y;
            if (p.y > max_y) max_y = p.y;
        }

        // 计算左右车道线在最高和最低点的x坐标
        for (size_t i = 0; i < left_points.size() - 1; i += 2) {
            float x1 = left_points[i].x, y1 = left_points[i].y;
            float x2 = left_points[i+1].x, y2 = left_points[i+1].y;
            
            if (abs(y2 - y1) > 1e-6) {
                left_x_at_min_y = x1 + (x2 - x1) * (min_y - y1) / (y2 - y1);
                left_x_at_max_y = x1 + (x2 - x1) * (max_y - y1) / (y2 - y1);
            }
        }

        for (size_t i = 0; i < right_points.size() - 1; i += 2) {
            float x1 = right_points[i].x, y1 = right_points[i].y;
            float x2 = right_points[i+1].x, y2 = right_points[i+1].y;
            
            if (abs(y2 - y1) > 1e-6) {
                right_x_at_min_y = x1 + (x2 - x1) * (min_y - y1) / (y2 - y1);
                right_x_at_max_y = x1 + (x2 - x1) * (max_y - y1) / (y2 - y1);
            }
        }

        // 定义矩形的四个顶点
        vector<Point> poly_points = {
            Point(left_x_at_min_y, min_y),
            Point(right_x_at_min_y, min_y),
            Point(right_x_at_max_y, max_y),
            Point(left_x_at_max_y, max_y)
        };

        // 创建半透明的蓝色多边形
        Mat overlay = frame.clone();
        fillPoly(overlay, vector<vector<Point>>{poly_points}, Scalar(255, 0, 0));
        addWeighted(overlay, 0.3, frame, 0.7, 0, frame);

        // 绘制车道线
        for (const auto& line : lines) {
            double angle = atan2(line[3] - line[1], line[2] - line[0]) * 180 / CV_PI;
            Scalar color = (angle > 0) ? Scalar(255, 0, 0) : Scalar(0, 0, 255);
            
            cv::line(frame, 
                    Point(line[0], line[1]),
                    Point(line[2], line[3]),
                    color,
                    3
            );
        }

        // 显示偏移信息
        string warning;
        Scalar warning_color;

        if (abs(deviation) > DEVIATION_THRESHOLD) {
            warning = deviation > 0 ? "Right Offset" : "Left Offset";
            warning_color = Scalar(0, 0, 255);
        } else {
            warning = "Normal";
            warning_color = Scalar(0, 255, 0);
        }

        putText(frame, warning, Point(50, 50),
                FONT_HERSHEY_SIMPLEX, 1, warning_color, 2, LINE_AA);
        
        string deviation_text = "Offset: " + to_string(int(deviation)) + " px";
        putText(frame, deviation_text, Point(50, 80),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2, LINE_AA);
    }

    void process() {
        Mat frame;
        int frame_count = 0;
        int total_frames = cap.get(CAP_PROP_FRAME_COUNT);

        while (cap.read(frame)) {
            frame_count++;
            cout << "\rProcessing: " << frame_count << "/" << total_frames 
                 << " (" << int(100.0 * frame_count / total_frames) << "%)" 
                 << flush;

            Mat processed = preprocess(frame);
            Mat roi = extractROI(processed);
            vector<Vec4i> lanes = detectLanes(roi);
            double deviation = calculateDeviation(lanes, frame);
            drawResults(frame, lanes, deviation);
            
            imshow(window_name, frame);
            writer.write(frame);
            
            char key = waitKey(1);
            if (key == 'q' || key == 27) break;
        }
        cout << "\nProcessing completed!" << endl;
    }

    ~LaneDetector() {
        cap.release();
        writer.release();
        destroyAllWindows();
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <input_video> <output_video.avi>" << endl;
        return -1;
    }

    try {
        LaneDetector detector(argv[1], argv[2]);
        cout << "Starting video processing..." << endl;
        detector.process();
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return -1;
    }

    return 0;
}
