#include"opencv_3.h"

// WaveletTransform2D 类实现

WaveletTransform2D::WaveletTransform2D() {

	// 给定的低通滤波器 g(n)
	g = { 0.00000, 0.0625, 0.25000, 0.37550, 0.25000, 0.0625, 0.00000, 0.00000 };
	// 给定的高通滤波器 h(n)
	h = { -0.000008, -0.01643, -0.10872, -0.59261, 0.59261, 0.10872, 0.01643, 0.000008 };

	assert(g.size() == h.size());
	// 修复：使用 static_cast 进行显式类型转换
	filterLen = static_cast<int>(g.size());
	assert(filterLen % 2 == 0);

}

void WaveletTransform2D::transform2D(const std::vector<std::vector<double>>& input, 
		std::vector<std::vector<double>>& output, 
		int levels) {

	int rows = static_cast<int>(input.size());
	int cols = static_cast<int>(input[0].size());

	if (rows == 0 || cols == 0) return;

	// 复制输入到输出
	output = input;

	int currentRows = rows;
	int currentCols = cols;
	int rowOffset = 0;
	int colOffset = 0;

	// 多级分解（仅对列进行）
	for (int level = 0; level < levels; ++level) {
		if (currentRows < 2) break;

		std::vector<std::vector<double>> subMatrix(currentRows, std::vector<double>(currentCols));
		for (int i = 0; i < currentRows; ++i) {
			for (int j = 0; j < currentCols; ++j) {
				subMatrix[i][j] = output[rowOffset + i][colOffset + j];
			}
		}

		std::vector<std::vector<double>> decomposed;
		transformLevel(subMatrix, decomposed);

		for (int i = 0; i < currentRows; ++i) {
			for (int j = 0; j < currentCols; ++j) {
				output[rowOffset + i][colOffset + j] = decomposed[i][j];
			}
		}

		currentRows /= 2;
	}

}

void WaveletTransform2D::multiScaleEdgeDetection(const std::vector<std::vector<double>>& image, 
		std::vector<std::vector<double>>& edgeImage, 
		int maxScale, 
		double threshold) {

	int rows = static_cast<int>(image.size());
	int cols = static_cast<int>(image[0].size());

	edgeImage.assign(rows, std::vector<double>(cols, 0.0));

	// Store per-column decomposition results
	std::vector<std::vector<double>> a1(cols);  // scale 1 approximation
	std::vector<std::vector<double>> a2(cols);  // scale 2 approximation
	std::vector<std::vector<double>> a3(cols);  // scale 3 approximation
	std::vector<std::vector<double>> d1(cols);  // scale 1 detail
	std::vector<std::vector<double>> d2(cols);  // scale 2 detail
	std::vector<std::vector<double>> d3(cols);  // scale 3 detail
	std::vector<std::vector<double>> edgeImage_1;
	edgeImage_1.assign(rows, std::vector<double>(cols, 0.0));

	// 3-level wavelet decomposition for each column
	for (int j = 0; j < cols; ++j) {
		// Extract column j
		std::vector<double> column(rows);
		for (int i = 0; i < rows; ++i) {
			column[i] = image[i][j];
		}

		std::vector<double> decomposed1 = waveletDecomposition1D(column);
		int halfLen1 = static_cast<int>(decomposed1.size() / 2);

		a1[j].resize(halfLen1);
		d1[j].resize(halfLen1);
		for (int i = 0; i < halfLen1; ++i) {
			a1[j][i] = decomposed1[i];
			d1[j][i] = decomposed1[i + halfLen1];
		}

		std::vector<double> decomposed2 = waveletDecomposition1D(a1[j]);
		int halfLen2 = static_cast<int>(decomposed2.size() / 2);

		a2[j].resize(halfLen2);
		d2[j].resize(halfLen2);
		for (int i = 0; i < halfLen2; ++i) {
			a2[j][i] = decomposed2[i];
			d2[j][i] = decomposed2[i + halfLen2];
		}

		std::vector<double> decomposed3 = waveletDecomposition1D(a2[j]);
		int halfLen3 = static_cast<int>(decomposed3.size() / 2);

		a3[j].resize(halfLen3);
		d3[j].resize(halfLen3);
		for (int i = 0; i < halfLen3; ++i) {
			a3[j][i] = decomposed3[i];
			d3[j][i] = decomposed3[i + halfLen3];
		}
	}

	// Compute edge response for each column
	for (int j = 0; j < cols; ++j) {
		int nRows = static_cast<int>(a1[j].size());

		std::vector<double> a1UpSampled(rows, 0.0);
		for (int i = 0; i < rows; ++i) {
			int idx = i / 2;
			if (idx < nRows) {
				a1UpSampled[i] = a1[j][idx];
			}
		}

		std::vector<double> d1UpSampled(rows, 0.0);
		std::vector<double> d2UpSampled(rows, 0.0);

		int d1Rows = static_cast<int>(d1[j].size());
		int d2Rows = static_cast<int>(d2[j].size());

		for (int i = 0; i < rows; ++i) {
			int idx1 = i / 2;
			int idx2 = i / 4;

			if (idx1 < d1Rows) {
				d1UpSampled[i] = d1[j][idx1];
			}
			if (idx2 < d2Rows) {
				d2UpSampled[i] = d2[j][idx2];
			}
		}

		std::vector<double> edgeColumn(rows, 0.0);
		std::vector<double> edge_last(rows, 0.0);

		for (int i = 0; i < rows; ++i) {
			double val = a1UpSampled[i] * a1UpSampled[i] * d1UpSampled[i] * d2UpSampled[i];
			edgeColumn[i] = val;
		}

		// Non-maximum suppression over 5-pixel window
		for (int i = 2; i < rows - 2; ++i) {
			if (edgeColumn[i] > edgeColumn[i - 2] && edgeColumn[i] > edgeColumn[i + 2]) {
				edge_last[i] = 1;
				i++;
			}
			else
				edge_last[i] = 0;
		}

		for (int i = 0; i < rows; ++i) {
			double normalized = edge_last[i];
			if (normalized >= threshold) {
				edgeImage_1[i][j] = normalized;
			}
		}
	}

	// Store per-row decomposition results
	std::vector<std::vector<double>> b1(rows);  // scale 1 approximation
	std::vector<std::vector<double>> b2(rows);  // scale 2 approximation
	std::vector<std::vector<double>> b3(rows);  // scale 3 approximation
	std::vector<std::vector<double>> c1(rows);  // scale 1 detail
	std::vector<std::vector<double>> c2(rows);  // scale 2 detail
	std::vector<std::vector<double>> c3(rows);  // scale 3 detail
	std::vector<std::vector<double>> edgeImage_2;
	edgeImage_2.assign(rows, std::vector<double>(cols, 0.0));

	// 3-level wavelet decomposition for each row
	for (int i = 0; i < rows; ++i) {
		std::vector<double> rowData(cols);
		for (int j = 0; j < cols; ++j) {
			rowData[j] = image[i][j];
		}

		std::vector<double> decomposed1 = waveletDecomposition1D(rowData);
		int halfLen1 = static_cast<int>(decomposed1.size() / 2);

		b1[i].resize(halfLen1);
		c1[i].resize(halfLen1);
		for (int j = 0; j < halfLen1; ++j) {
			b1[i][j] = decomposed1[j];
			c1[i][j] = decomposed1[j + halfLen1];
		}

		std::vector<double> decomposed2 = waveletDecomposition1D(b1[i]);
		int halfLen2 = static_cast<int>(decomposed2.size() / 2);

		b2[i].resize(halfLen2);
		c2[i].resize(halfLen2);
		for (int j = 0; j < halfLen2; ++j) {
			b2[i][j] = decomposed2[j];
			c2[i][j] = decomposed2[j + halfLen2];
		}

		std::vector<double> decomposed3 = waveletDecomposition1D(b2[i]);
		int halfLen3 = static_cast<int>(decomposed3.size() / 2);

		b3[i].resize(halfLen3);
		c3[i].resize(halfLen3);
		for (int j = 0; j < halfLen3; ++j) {
			b3[i][j] = decomposed3[j];
			c3[i][j] = decomposed3[j + halfLen3];
		}
	}

	// Compute edge response for each row
	for (int i = 0; i < rows; ++i) {
		int nCols = static_cast<int>(b1[i].size());

		std::vector<double> b1UpSampled(cols, 0.0);
		for (int j = 0; j < cols; ++j) {
			int idx = j / 2;
			if (idx < nCols) {
				b1UpSampled[j] = b1[i][idx];
			}
		}

		std::vector<double> c1UpSampled(cols, 0.0);
		std::vector<double> c2UpSampled(cols, 0.0);

		int c1Cols = static_cast<int>(c1[i].size());
		int c2Cols = static_cast<int>(c2[i].size());

		for (int j = 0; j < cols; ++j) {
			int idx1 = j / 2;
			int idx2 = j / 4;

			if (idx1 < c1Cols) {
				c1UpSampled[j] = c1[i][idx1];
			}
			if (idx2 < c2Cols) {
				c2UpSampled[j] = c2[i][idx2];
			}
		}

		std::vector<double> edgeRow(cols, 0.0);

		for (int j = 0; j < cols; ++j) {
			double val = b1UpSampled[j] * b1UpSampled[j] * c1UpSampled[j] * c2UpSampled[j];
			edgeRow[j] = val;
		}

		std::vector<double> edge_last(cols, 0.0);
		for (int j = 2; j < cols - 2; ++j) {
			if (edgeRow[j] > edgeRow[j - 2] && edgeRow[j] > edgeRow[j + 2]) {
				edge_last[j] = 1;
			}
			else {
				edge_last[j] = 0;
			}
		}

		double maxVal = 0.0;
		for (int j = 0; j < cols; ++j) {
			if (edge_last[j] > maxVal) maxVal = edge_last[j];
		}

		if (maxVal > 1e-10) {
			for (int j = 0; j < cols; ++j) {
				double normalized = edge_last[j];
				if (normalized >= 0) {
					edgeImage_2[i][j] = normalized;
				}
			}
		}
	}

	// Combine column and row edge responses
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j) {
			edgeImage[i][j] = edgeImage_1[i][j] + edgeImage_2[i][j];
		}



	#ifdef DEBUG_OUTPUT
	saveDebugImage("a1_debug.png", a1);
	saveDebugImage("a2_debug.png", a2);
	saveDebugImage("a3_debug.png", a3);
	saveDebugImage("d1_debug.png", d1);
	saveDebugImage("d2_debug.png", d2);
	saveDebugImage("d3_debug.png", d3);
	#endif

}

void WaveletTransform2D::saveDebugImage(const std::string& filename,  const std::vector<std::vector<double>>& data) {

	if (data.empty() || data[0].empty()) {
		std::cerr << "Warning: Empty debug data, cannot save " << filename << std::endl;
		return;
	}

	int cols = static_cast<int>(data.size());
	int rows = static_cast<int>(data[0].size());

	std::cout << "Saving debug: " << filename << " (cols=" << cols << ", rows=" << rows << ")" << std::endl;

	// 创建一个 [rows][cols] 的 Mat（行优先）
	cv::Mat image(rows, cols, CV_64F);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			image.at<double>(i, j) = data[j][i];
		}
	}

	// ===== 新增：打印数据统计 =====
	double sum = 0.0;
	int count = 0;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			sum += image.at<double>(i, j);
			count++;
		}
	}
	std::cout << "  Data sum: " << sum << ", average: " << (count > 0 ? sum / count : 0) << std::endl;

	// 检查数据范围
	double minVal, maxVal;
	cv::minMaxLoc(image, &minVal, &maxVal);
	std::cout << "  Data range: [" << minVal << ", " << maxVal << "]" << std::endl;

	// ===== 关键修改：直接保存，不归一化 =====
	cv::Mat u8Image;
	image.convertTo(u8Image, CV_8U, 255.0);
	cv::imwrite(filename, u8Image);
	std::cout << "  Debug image saved (no normalization): " << filename << std::endl;



}

void WaveletTransform2D::transformLevel(const std::vector<std::vector<double>>& input, 
		std::vector<std::vector<double>>& output) {

	int rows = static_cast<int>(input.size());
	int cols = static_cast<int>(input[0].size());

	output.assign(rows, std::vector<double>(cols, 0.0));

	for (int j = 0; j < cols; ++j) {
		std::vector<double> colData(rows);
		for (int i = 0; i < rows; ++i) {
			colData[i] = input[i][j];
		}

		std::vector<double> transformedCol = waveletDecomposition1D(colData);

		for (int i = 0; i < rows; ++i) {
			output[i][j] = transformedCol[i];
		}
	}

}

std::vector<double> WaveletTransform2D::waveletDecomposition1D(const std::vector<double>& signal) {

	int N = static_cast<int>(signal.size());
	int halfN = N / 2;
	std::vector<double> approx(halfN, 0.0);
	std::vector<double> detail(halfN, 0.0);

	for (int k = 0; k < halfN; ++k) {
		double sumG = 0.0;
		double sumH = 0.0;
		for (int n = 0; n < filterLen; ++n) {
			int idx = 2 * k + (n - 3);
			double signalVal = 0.0;
			if (idx >= 0 && idx < N) {
				signalVal = signal[idx];
			}
			sumG += g[n] * signalVal;
			sumH += h[n] * signalVal;
		}
		approx[k] = sumG;
		detail[k] = sumH;
	}

	std::vector<double> result(N);
	for (int k = 0; k < halfN; ++k) {
		result[k] = approx[k];
		result[k + halfN] = detail[k];
	}
	return result;

}
