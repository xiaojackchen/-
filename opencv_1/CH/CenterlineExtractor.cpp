#include"opencv_3.h"

// CenterlineExtractor 类实现

void CenterlineExtractor::extractCenterlineFromEdgeMap(const std::vector<std::vector<double>>& edgeInput,  std::vector<std::vector<double>> &output, 
		int dilateIter) {

	int rows = static_cast<int>(edgeInput.size());
	int cols = static_cast<int>(edgeInput[0].size());
	  // edgeInput → cv::Mat → OpenCV Otsu 二值分割
	cv::Mat edgeMat(rows, cols, CV_64F, cv::Scalar(0));
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			edgeMat.at<double>(i, j) = edgeInput[i][j];

	cv::Mat edge8U;
	edgeMat.convertTo(edge8U, CV_8U, 255.0);

	cv::Mat binary;
	cv::threshold(edge8U, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	cv::imwrite("output.png", binary);
	cv::Mat src = binary.clone();

	cv::Mat result4 = MorphologicalHoleFillerComplete::fillHolesProgressive(src, 7);
	cv::Mat result3 = MorphologicalHoleFillerComplete::fillHolesAuto(result4);

	cv::Mat kernel = cv::getStructuringElement(
		cv::MORPH_RECT,
		cv::Size(3, 3)
		);
	cv::morphologyEx(result3, result3, cv::MORPH_CLOSE, kernel,cv::Point(-1, -1));

	cv::imwrite("out_2.png", result3);
	cv::Mat skeleton = result3.clone();
	zhangSuenThinning(skeleton);

	int skelCount = cv::countNonZero(skeleton);
	std::cout << "  Skeleton pixels (before dilation): " << skelCount << std::endl;

	cv::imwrite("skeleton.png", skeleton);//细化结果

	// Step 4: Dilate the skeleton to match target thickness (CROSS kernel for finer control)
	cv::Mat thickened;
	if (dilateIter > 0) {
		cv::dilate(skeleton, thickened,
			cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3)),
			cv::Point(-1, -1), dilateIter);
	}
	else {
		thickened = skeleton.clone();
	}//骨架膨胀

	 //cv::imwrite("thickened.png", thickened);

	int thickCount = cv::countNonZero(thickened);
	std::cout << "  Centerline pixels (after " << dilateIter << " dilations): "
		<< thickCount << std::endl;

	// Convert to vector format
	output.assign(rows, std::vector<double>(cols, 0.0));
	int outCount = 0;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			if (thickened.at<uchar>(i, j) > 0) {
				output[i][j] = 1.0;
				outCount++;
			}
		}
	}


}

void CenterlineExtractor::zhangSuenThinning(cv::Mat& binary) {

	bool changed = true;
	while (changed) {
		changed = thinningIteration(binary, true);
		bool changed2 = thinningIteration(binary, false);
		changed = changed || changed2;
	}


}

bool CenterlineExtractor::thinningIteration(cv::Mat& img,  bool firstIteration) {

	int rows = img.rows;
	int cols = img.cols;
	std::vector<cv::Point> toDelete;

	for (int i = 1; i < rows - 1; ++i) {
		for (int j = 1; j < cols - 1; ++j) {
			if (img.at<uchar>(i, j) == 0) continue;

			int p2 = img.at<uchar>(i - 1, j) > 0 ? 1 : 0;
			int p3 = img.at<uchar>(i - 1, j + 1) > 0 ? 1 : 0;
			int p4 = img.at<uchar>(i, j + 1) > 0 ? 1 : 0;
			int p5 = img.at<uchar>(i + 1, j + 1) > 0 ? 1 : 0;
			int p6 = img.at<uchar>(i + 1, j) > 0 ? 1 : 0;
			int p7 = img.at<uchar>(i + 1, j - 1) > 0 ? 1 : 0;
			int p8 = img.at<uchar>(i, j - 1) > 0 ? 1 : 0;
			int p9 = img.at<uchar>(i - 1, j - 1) > 0 ? 1 : 0;

			int neighbors[8] = { p2, p3, p4, p5, p6, p7, p8, p9 };

			int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
			if (B < 2 || B > 6) continue;

			int A = 0;
			for (int k = 0; k < 8; ++k) {
				if (neighbors[k] == 0 && neighbors[(k + 1) % 8] == 1) A++;
			}
			if (A != 1) continue;

			if (firstIteration) {
				if (p2 * p4 * p6 != 0) continue;
				if (p4 * p6 * p8 != 0) continue;
			}
			else {
				if (p2 * p4 * p8 != 0) continue;
				if (p2 * p6 * p8 != 0) continue;
			}

			toDelete.push_back(cv::Point(j, i));
		}
	}

	for (size_t k = 0; k < toDelete.size(); ++k) {
		img.at<uchar>(toDelete[k].y, toDelete[k].x) = 0;
	}
	return !toDelete.empty();

}
