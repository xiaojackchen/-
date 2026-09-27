#include"opencv_3.h"

// ImageProcessor 类实现

std::vector<std::vector<double>> ImageProcessor::readImage(const std::string& filename,  int& width,  int& height) {

	cv::Mat image = cv::imread(filename, cv::IMREAD_UNCHANGED);
	if (image.empty()) {
		throw std::runtime_error("Failed to load image: " + filename);
	}

	height = image.rows;
	width = image.cols;

	std::vector<std::vector<double>> result(height, std::vector<double>(width));

	cv::Mat gray;
	if (image.channels() == 1) {
		gray = image.clone();
	}
	else {
		cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	}

	cv::Mat doubleGray;
	gray.convertTo(doubleGray, CV_64F, 1.0 / 255.0);

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			result[i][j] = doubleGray.at<double>(i, j);
		}
	}

	return result;

}

void ImageProcessor::saveImage(const std::string& filename, 
		const std::vector<std::vector<double>>& image, 
		bool normalize) {

	int height = static_cast<int>(image.size());
	if (height == 0) {
		std::cerr << "Warning: Empty image, cannot save " << filename << std::endl;
		return;
	}
	int width = static_cast<int>(image[0].size());
	if (width == 0) {
		std::cerr << "Warning: Empty image, cannot save " << filename << std::endl;
		return;
	}

	std::cout << "Saving " << filename << " (" << height << "x" << width << ")" << std::endl;

	cv::Mat mat(height, width, CV_64F);
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			mat.at<double>(i, j) = image[i][j];
		}
	}

	double minVal, maxVal;
	cv::minMaxLoc(mat, &minVal, &maxVal);
	std::cout << "  Data range: [" << minVal << ", " << maxVal << "]" << std::endl;

	cv::Mat normalized;

	if (normalize) {
		if (maxVal > minVal) {
			mat.convertTo(normalized, CV_64F, 1.0 / (maxVal - minVal), -minVal / (maxVal - minVal));
		}
		else {
			if (minVal == 0.0) {
				normalized = cv::Mat(height, width, CV_64F, cv::Scalar(128.0 / 255.0));
				std::cout << "  WARNING: All black, generating gray image" << std::endl;
			}
			else {
				mat.convertTo(normalized, CV_64F, 1.0 / maxVal, 0);
			}
		}
	}
	else {
		normalized = mat.clone();
		cv::threshold(normalized, normalized, 0.0, 1.0, cv::THRESH_TRUNC);
		cv::threshold(normalized, normalized, 0.0, 1.0, cv::THRESH_TOZERO);
	}

	double newMin, newMax;
	cv::minMaxLoc(normalized, &newMin, &newMax);
	std::cout << "  Normalized range: [" << newMin << ", " << newMax << "]" << std::endl;

	cv::Mat u8Image;
	normalized.convertTo(u8Image, CV_8U, 255.0);

	bool success = cv::imwrite(filename, u8Image);
	if (success) {
		std::cout << "Image saved successfully: " << filename << std::endl;
	}
	else {
		std::cerr << "ERROR: Failed to save image: " << filename << std::endl;
	}

}

void ImageProcessor::padToPowerOfTwo(std::vector<std::vector<double>>& image) {

	int height = static_cast<int>(image.size());
	int width = static_cast<int>(image[0].size());

	int newHeight = 1;
	while (newHeight < height) newHeight *= 2;

	int newWidth = 1;
	while (newWidth < width) newWidth *= 2;

	if (newHeight != height || newWidth != width) {
		std::vector<std::vector<double>> padded(newHeight, std::vector<double>(newWidth, 0.0));
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				padded[i][j] = image[i][j];
			}
		}
		image = padded;
		std::cout << "Image padded to " << newHeight << "x" << newWidth << std::endl;
	}

}

std::vector<std::vector<double>> ImageProcessor::preprocessWithOtsu(const std::vector<std::vector<double>>& image, 
		double& otsuThreshold) {

	std::cout << "Applying OpenCV Otsu threshold segmentation to original image..." << std::endl;

	int rows = static_cast<int>(image.size());
	int cols = static_cast<int>(image[0].size());

	// 转换为 cv::Mat
	cv::Mat grayMat(rows, cols, CV_64F);
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			grayMat.at<double>(i, j) = image[i][j];

	// 转 8-bit 供 OpenCV Otsu 使用
	cv::Mat gray8U;
	grayMat.convertTo(gray8U, CV_8U, 255.0);

	// 使用 OpenCV 内置 Otsu 进行二值分割
	cv::Mat binary8U;
	double otsuRaw = cv::threshold(gray8U, binary8U, 0, 255,
		cv::THRESH_BINARY | cv::THRESH_OTSU);
	// 2. 应用调整量
	double finalThreshold = otsuRaw*0.8;
	finalThreshold = std::max(0.0, std::min(255.0, finalThreshold));  // 限制在 [0, 255]

																	  // 3. 用调整后的阈值重新分割
	cv::threshold(gray8U, binary8U, finalThreshold, 255, cv::THRESH_BINARY);
	otsuThreshold = finalThreshold / 255.0;

	std::cout << "OpenCV Otsu threshold: " << (int)otsuRaw << "/255  ("
		<< otsuThreshold << ")" << std::endl;

	// 转回 vector 格式
	std::vector<std::vector<double>> segmentedImage(rows, std::vector<double>(cols, 0.0));
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			segmentedImage[i][j] = (binary8U.at<uchar>(i, j) > 0) ? 1.0 : 0.0;

	int ones = 0;
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			if (segmentedImage[i][j] > 0.5) ones++;
	std::cout << "Segmented image: " << ones << " white pixels out of " << rows * cols
		<< " (" << (double)ones / (rows * cols) * 100 << "%)" << std::endl;

	saveImage("otsu_segmented_original.png", segmentedImage, true);
	return segmentedImage;

}

std::vector<std::vector<double>> ImageProcessor::butterworthHighPassFilter(const std::vector<std::vector<double>>& image, 
		double D0, 
		int n) {

	int rows = static_cast<int>(image.size());
	int cols = static_cast<int>(image[0].size());

	// 转换为 cv::Mat (double, 对应 MATLAB im2double)
	cv::Mat src(rows, cols, CV_64F);
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			src.at<double>(i, j) = image[i][j];

	// 构建巴特沃斯高通滤波器核 H(u,v)=1/(1+(D0/D)^(2n))
	// 使用居中坐标对应 MATLAB meshgrid(-cols/2:cols/2-1, -rows/2:rows/2-1)
	int cx = cols / 2;
	int cy = rows / 2;
	cv::Mat H(rows, cols, CV_64F, cv::Scalar(0.0));
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			double du = static_cast<double>(i - cy);
			double dv = static_cast<double>(j - cx);
			double D = std::sqrt(du * du + dv * dv);
			if (D > 1e-12) {
				double ratio = D0 / D;
				H.at<double>(i, j) = 1.0 / (1.0 + std::pow(ratio, 2 * n));
			} else {
				H.at<double>(i, j) = 0.0; // 直流分量置零
			}
		}
	}

	// FFT → fftshift → 乘滤波器 → ifftshift → IFFT
	// 对应 MATLAB: I_fft = fft2(I); I_filtered = real(ifft2(ifftshift(fftshift(I_fft) .* H)));

	// FFT2 (扩展为双通道复数)
	cv::Mat planes[] = {src, cv::Mat::zeros(rows, cols, CV_64F)};
	cv::Mat complexI;
	cv::merge(planes, 2, complexI);
	cv::dft(complexI, complexI);

	// fftshift: 交换 TL↔BR, TR↔BL (用临时备份避免重叠)
	{
		cv::Mat tmp;
		complexI.copyTo(tmp);
		tmp(cv::Rect(cx, cy, cols - cx, rows - cy)).copyTo(
			complexI(cv::Rect(0, 0, cx, cy)));   // BR → TL
		tmp(cv::Rect(0, 0, cx, cy)).copyTo(
			complexI(cv::Rect(cx, cy, cols - cx, rows - cy)));   // TL → BR
		tmp(cv::Rect(cx, 0, cols - cx, cy)).copyTo(
			complexI(cv::Rect(0, cy, cx, rows - cy)));   // TR → BL
		tmp(cv::Rect(0, cy, cx, rows - cy)).copyTo(
			complexI(cv::Rect(cx, 0, cols - cx, cy)));   // BL → TR
	}

	// 应用滤波器: 复数频谱逐点乘 H (对应 MATLAB .*)
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			double re = complexI.at<cv::Vec2d>(i, j)[0];
			double im = complexI.at<cv::Vec2d>(i, j)[1];
			double w = H.at<double>(i, j);
			complexI.at<cv::Vec2d>(i, j)[0] = re * w;
			complexI.at<cv::Vec2d>(i, j)[1] = im * w;
		}
	}

	// ifftshift: 逆象限交换 (和 fftshift 相同操作，因为交换是自逆的)
	{
		cv::Mat tmp;
		complexI.copyTo(tmp);
		tmp(cv::Rect(cx, cy, cols - cx, rows - cy)).copyTo(
			complexI(cv::Rect(0, 0, cx, cy)));   // BR → TL
		tmp(cv::Rect(0, 0, cx, cy)).copyTo(
			complexI(cv::Rect(cx, cy, cols - cx, rows - cy)));   // TL → BR
		tmp(cv::Rect(cx, 0, cols - cx, cy)).copyTo(
			complexI(cv::Rect(0, cy, cx, rows - cy)));   // TR → BL
		tmp(cv::Rect(0, cy, cx, rows - cy)).copyTo(
			complexI(cv::Rect(cx, 0, cols - cx, cy)));   // BL → TR
	}

	// IDFT → 取实部 → 取绝对值 (使背景≈0为黑，边缘响应为正)
	cv::Mat resultReal;
	cv::idft(complexI, resultReal, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

	cv::Mat resultAbs = cv::abs(resultReal);

	double minV, maxV;
	cv::minMaxLoc(resultAbs, &minV, &maxV);
	std::cout << "Butterworth HPF: D0=" << D0 << ", n=" << n
		<< ", abs range=[" << minV << ", " << maxV << "]" << std::endl;

	std::vector<std::vector<double>> out(rows, std::vector<double>(cols, 0.0));
	if (maxV > 1e-12) {
		for (int i = 0; i < rows; ++i)
			for (int j = 0; j < cols; ++j)
				out[i][j] = std::abs(resultReal.at<double>(i, j)) / maxV;
	}

	saveImage("butterworth_hpf.png", out, true);
	return out;

}
