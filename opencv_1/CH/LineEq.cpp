//直线方程函数
#include"opencv_3.h"


// 计算两点间的直线方程和角度
LineEquation calculateLine(const cv::Point2d& p1, const cv::Point2d& p2) {
	LineEquation line;

	double dx = p2.x - p1.x;
	double dy = p2.y - p1.y;

	// 1. 计算角度
	line.angleRad = std::atan2(dy, dx);
	line.angleDeg = line.angleRad * 180.0 / CV_PI;

	// 2. 计算一般式 Ax + By + C = 0
	// 使用两点式：(y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
	line.A = dy;           // y2 - y1
	line.B = -dx;          // -(x2 - x1)
	line.C = dx * p1.y - dy * p1.x;  // (x2-x1)y1 - (y2-y1)x1

	return line;
}

// 打印直线信息
void printLineInfo(const LineEquation& line, const std::string& name = "直线") {
	std::cout << "=== " << name << " ===" << std::endl;
	std::cout << "一般式: " << line.A << "x + " << line.B << "y + " << line.C << " = 0" << std::endl;
	std::cout << "角度: " << line.angleDeg << "°" << std::endl;
	std::cout << std::endl;
}