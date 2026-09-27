#include"opencv_3.h"

// MorphologicalHoleFillerComplete 类实现

cv::Mat MorphologicalHoleFillerComplete::fillHoles(const cv::Mat& src, 
		int kernelSize, 
		KernelShape shape, 
		int iterations, 
		bool useReconstruction) {

	CV_Assert(src.type() == CV_8U);

	cv::Mat kernel = cv::getStructuringElement(
		(int)shape,
		cv::Size(kernelSize, kernelSize)
		);

	if (useReconstruction) {
		// 使用重建闭运算
		return fillHolesReconstruction(src, kernel, iterations);
	}
	else {
		// 使用传统闭运算
		cv::Mat result = src.clone();
		for (int i = 0; i < iterations; ++i) {
			cv::morphologyEx(result, result, cv::MORPH_CLOSE, kernel);
		}
		return result;
	}

}

cv::Mat MorphologicalHoleFillerComplete::fillHolesAuto(const cv::Mat& src) {

	// 分析孔洞大小
	int maxHoleSize = findMaxHoleSize(src);

	// 根据最大孔洞大小选择核
	int kernelSize = std::max(3, maxHoleSize / 2);
	kernelSize = kernelSize % 2 == 0 ? kernelSize + 1 : kernelSize;
	kernelSize = std::min(kernelSize, 15);  // 限制最大值

	std::cout << "Auto selected kernel size: " << kernelSize << std::endl;

	return fillHoles(src, kernelSize, ELLIPSE, 1, true);

}

cv::Mat MorphologicalHoleFillerComplete::fillHolesProgressive(const cv::Mat& src,  int maxKernelSize) {

	cv::Mat result = src.clone();

	for (int size = 3; size <= maxKernelSize; size += 2) {
		cv::Mat kernel = cv::getStructuringElement(
			cv::MORPH_ELLIPSE,
			cv::Size(size, size)
			);

		// 使用重建闭运算
		result = fillHolesReconstruction(result, kernel, 1);

		// 保存中间结果（调试用）
		// cv::imwrite("step_" + std::to_string(size) + ".png", result);
	}

	return result;

}

cv::Mat MorphologicalHoleFillerComplete::fillHolesReconstruction(const cv::Mat& src, 
		const cv::Mat& kernel, 
		int iterations) {

	// 1. 闭运算
	cv::Mat closed = src.clone();
	for (int i = 0; i < iterations; ++i) {
		cv::morphologyEx(closed, closed, cv::MORPH_CLOSE, kernel);
	}

	// 2. 重建
	cv::Mat result = closed.clone();
	cv::Mat previous;
	cv::Mat temp;

	int maxIter = 50;
	int iter = 0;

	do {
		previous = result.clone();

		// 膨胀
		cv::dilate(result, temp, kernel);

		// 取最小值（限制在闭运算范围内）
		cv::min(temp, closed, result);

		iter++;
	} while (cv::countNonZero(previous != result) > 0 && iter < maxIter);

	return result;

}

int MorphologicalHoleFillerComplete::findMaxHoleSize(const cv::Mat& src) {

	cv::Mat inverted;
	cv::bitwise_not(src, inverted);

	cv::Mat labels, stats, centroids;
	int numObjects = cv::connectedComponentsWithStats(
		inverted, labels, stats, centroids, 8, CV_32S);

	int maxArea = 0;
	int backgroundLabel = 0;

	// 找到背景（最大的连通域）
	int maxBgArea = 0;
	for (int i = 1; i < numObjects; ++i) {
		int area = stats.at<int>(i, 4);
		if (area > maxBgArea) {
			maxBgArea = area;
			backgroundLabel = i;
		}
	}

	// 找到最大的孔洞
	for (int i = 1; i < numObjects; ++i) {
		if (i != backgroundLabel) {
			int area = stats.at<int>(i, 4);
			if (area > maxArea) {
				maxArea = area;
			}
		}
	}

	// 估算孔洞直径
	return (int)std::sqrt(maxArea);

}
