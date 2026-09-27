#pragma once

#define  _CRT_SECURE_NO_WARNINGS
#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <numeric>
#include <cassert>
#include <climits>
#include <limits>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// ==================== 直线方程 ====================
struct LineEquation {
	double A;          // Ax + By + C = 0
	double B;
	double C;
	double angleDeg;   // 角度（度）
	double angleRad;   // 角度（弧度）
};

// LineEq.cpp 中实现
LineEquation calculateLine(const cv::Point2d& p1, const cv::Point2d& p2);
void printLineInfo(const LineEquation& line, const std::string& name);

// ==================== 小波变换类 ====================
// 使用给定的小波滤波器系数进行二维离散小波变换（DWT）
class WaveletTransform2D {
public:
	// 构造函数：初始化滤波器系数
	WaveletTransform2D();

	// 执行二维离散小波变换（多级分解）
	void transform2D(const std::vector<std::vector<double>>& input,
		std::vector<std::vector<double>>& output,
		int levels = 2);

	// 多尺度边缘检测：对每一列进行3层小波分解，计算 (a1*d1)*(a1*d2) 并取最大值
	void multiScaleEdgeDetection(const std::vector<std::vector<double>>& image,
		std::vector<std::vector<double>>& edgeImage,
		int maxScale = 3,
		double threshold = 0.0);

private:
	std::vector<double> g;
	std::vector<double> h;
	int filterLen;

	// 保存调试图像（输入数据格式：[cols][rows]）- 使用OpenCV
	void saveDebugImage(const std::string& filename, const std::vector<std::vector<double>>& data);
	void transformLevel(const std::vector<std::vector<double>>& input,
		std::vector<std::vector<double>>& output);
	std::vector<double> waveletDecomposition1D(const std::vector<double>& signal);
};

// ==================== 形态学孔洞填充 ====================
class MorphologicalHoleFillerComplete {
public:
	enum KernelShape {
		RECT = cv::MORPH_RECT,
		CROSS = cv::MORPH_CROSS,
		ELLIPSE = cv::MORPH_ELLIPSE
	};

	// 主函数：使用形态学填充孔洞
	static cv::Mat fillHoles(const cv::Mat& src,
		int kernelSize = 3,
		KernelShape shape = ELLIPSE,
		int iterations = 1,
		bool useReconstruction = true);

	// 自动选择最优核大小
	static cv::Mat fillHolesAuto(const cv::Mat& src);

	// 渐进式填充（从小到大）
	static cv::Mat fillHolesProgressive(const cv::Mat& src, int maxKernelSize = 7);

private:
	// 形态学重建闭运算
	static cv::Mat fillHolesReconstruction(const cv::Mat& src,
		const cv::Mat& kernel,
		int iterations = 1);

	// 查找最大孔洞大小
	static int findMaxHoleSize(const cv::Mat& src);
};

// ==================== 中心线提取 ====================
class CenterlineExtractor {
public:
	static void extractCenterlineFromEdgeMap(
		const std::vector<std::vector<double>>& edgeInput,
		std::vector<std::vector<double>>& output,
		int dilateIter = 1);

private:
	static void zhangSuenThinning(cv::Mat& binary);
	static bool thinningIteration(cv::Mat& img, bool firstIteration);
};

// ==================== 图像处理工具类 ====================
class ImageProcessor {
public:
	static std::vector<std::vector<double>> readImage(const std::string& filename, int& width, int& height);

	static void saveImage(const std::string& filename,
		const std::vector<std::vector<double>>& image,
		bool normalize = true);

	static void padToPowerOfTwo(std::vector<std::vector<double>>& image);

	// OpenCV Otsu 阈值分割
	static std::vector<std::vector<double>> preprocessWithOtsu(
		const std::vector<std::vector<double>>& image,
		double& otsuThreshold);

	// 巴特沃斯高通滤波器 (匹配MATLAB实现)
	static std::vector<std::vector<double>> butterworthHighPassFilter(
		const std::vector<std::vector<double>>& image,
		double D0 = 30.0,
		int n = 2);
};
