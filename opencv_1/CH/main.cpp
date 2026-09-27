#include"opencv_3.h"

int main(int argc, char* argv[]) {
	try {
		std::string inputImage = "D:/Picture/8.1.3.jpg";
		std::string outputTransform = "transform_output.png";
		std::string outputBinaryEdge = "binary_edge_output.png";
		std::string outputMultiScale = "multiscale_edge.png";
		std::string outputMultiScale_S = "multiscale_edge_s.png";
		std::string outputOrgin= "Orgin.png";
		int levels = 1;
		double threshold = 0.0;

		if (argc >= 2) {
			inputImage = argv[1];
		}
	
		if (argc >= 3) {
			levels = std::atoi(argv[3]);
			if (levels < 1) levels = 1;
		}
		if (argc >= 4) {
			threshold = std::atof(argv[4]);
		}

		std::cout << "=== Wavelet Edge Detection with Multi-Scale Analysis ===" << std::endl;
		std::cout << "Processing: " << inputImage << std::endl;
		std::cout << "Levels: " << levels << ", Threshold: " << threshold << std::endl;

		// ============================================================
		// Step 1: Read and verify image is grayscale; convert if needed
		// ============================================================
		cv::Mat rawImage = cv::imread(inputImage, cv::IMREAD_UNCHANGED);
		if (rawImage.empty()) {
			throw std::runtime_error("Failed to load image: " + inputImage);
		}
		int width = rawImage.cols, height = rawImage.rows;

		cv::Mat grayMat;
		if (rawImage.channels() != 1) {
			std::cout << "Input is color (" << rawImage.channels() << " channels). Converting to grayscale." << std::endl;
			cv::cvtColor(rawImage, grayMat, cv::COLOR_BGR2GRAY);
		}
		else {
			std::cout << "Input is grayscale (1 channel)." << std::endl;
			grayMat = rawImage.clone();
		}
		// 1.5 双边保边滤波 (保留边缘同时平滑噪声，避免低对比度线段丢失)
		cv::Mat bilateralMat;
		cv::bilateralFilter(grayMat, bilateralMat, 3, 10,10);
		cv::imwrite("bilateral_filtered.png", bilateralMat);
		grayMat = bilateralMat;  // 替换 grayMat 继续管线

		// Convert to [0,1] double matrix
		cv::Mat doubleGray;
		grayMat.convertTo(doubleGray, CV_64F, 1.0 / 255.0);
		std::vector<std::vector<double>> originalImage(height, std::vector<double>(width));
		for (int i = 0; i < height; ++i)
			for (int j = 0; j < width; ++j)
				originalImage[i][j] = doubleGray.at<double>(i, j);

		std::cout << "Original image size: " << height << "x" << width << std::endl;
		ImageProcessor::saveImage(outputOrgin, originalImage, true);
		
		// 2. 对原始图像做巴特沃斯高通滤波 (D0=30, n=2)
		std::cout << "\n=== Butterworth High-Pass Filter ===" << std::endl;
		std::vector<std::vector<double>> filteredImage =
			ImageProcessor::butterworthHighPassFilter(originalImage, 30.0, 2);
		// 3. 非锐化掩蔽: I_sharpened = I*2 + 3*I_filtered → mat2gray
		std::cout << "\n=== Unsharp Masking (I*2 + 3*I_filtered) ===" << std::endl;
		double minS = std::numeric_limits<double>::max(), maxS = std::numeric_limits<double>::lowest();
		std::vector<std::vector<double>> sharpenedImage(height, std::vector<double>(width));
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				double val = originalImage[i][j]  + filteredImage[i][j] *1.5;
				sharpenedImage[i][j] = val;
				if (val < minS) minS = val;
				if (val > maxS) maxS = val;
			}
		}
		double rangeS = maxS - minS;
		std::cout << "  Unsharp range=[" << minS << ", " << maxS << "]" << std::endl;
		if (rangeS > 1e-12) {
			for (int i = 0; i < height; ++i)
				for (int j = 0; j < width; ++j)
					sharpenedImage[i][j] = (sharpenedImage[i][j] - minS) / rangeS;
		}
		ImageProcessor::saveImage("unsharp_sharpened.png", sharpenedImage, true);
		
		
		// 4. 对滤波后的图像进行Otsu阈值分割
		double otsuThreshold;
		std::vector<std::vector<double>> segmentedImage = ImageProcessor::preprocessWithOtsu(sharpenedImage, otsuThreshold);
		std::cout << "Otsu segmentation completed. Threshold: " << otsuThreshold << std::endl;

		// 5. 使用多尺度边缘检测
		WaveletTransform2D wt;

		std::cout << "\n=== Multi-Scale Edge Detection ===" << std::endl;
		std::vector<std::vector<double>> multiScaleEdge;
		wt.multiScaleEdgeDetection(segmentedImage, multiScaleEdge, 3, threshold);
		ImageProcessor::saveImage(outputMultiScale, multiScaleEdge);
		
		//6
		CenterlineExtractor ft;
		std::vector<std::vector<double>> out_mid;
		ft.extractCenterlineFromEdgeMap(multiScaleEdge, out_mid,1);
		ImageProcessor::saveImage("fill_out.png", out_mid);


		// ============================================================
		// Step 5: Hough Transform → angle cluster → endpoint connect → fitLine
		// ============================================================
		{
			std::cout << "\n=== Hough Line Detection with Angle Clustering ===" << std::endl;

			// Use Otsu-based centerline for Hough
			int hRows = static_cast<int>(out_mid.size());
			int hCols = static_cast<int>(out_mid[0].size());

			cv::Mat houghSrc(hRows, hCols, CV_8U, cv::Scalar(0));
			for (int i = 0; i < hRows; ++i)
				for (int j = 0; j < hCols; ++j)
					houghSrc.at<uchar>(i, j) = (out_mid[i][j] > 0.5) ? 255 : 0;

			// Step 5a: HoughLinesP to get short segments
			std::vector<cv::Vec4i> segments;
			cv::HoughLinesP(houghSrc, segments, 1, CV_PI / 180, 20, 15, 8);//弧度制
			std::cout << "  Raw Hough segments: " << segments.size() << std::endl;

			if (segments.empty()) {
				std::cout << "  No lines detected." << std::endl;
			}
			else {
				// Structure for a line segment with angle info
				struct SegInfo {
					cv::Point p1, p2;
					double angle;  // [0, PI)
					double len;
				};

				std::vector<SegInfo> segs;
				for (const auto& s : segments) {
					SegInfo info;
					info.p1 = cv::Point(s[0], s[1]);
					info.p2 = cv::Point(s[2], s[3]);
					double dx = static_cast<double>(info.p2.x - info.p1.x);
					double dy = static_cast<double>(info.p2.y - info.p1.y);
					info.len = std::sqrt(dx * dx + dy * dy);
					info.angle = std::atan2(dy, dx);
					// Normalize to [0, PI) — line direction is undirected
					if (info.angle < 0) info.angle += CV_PI;
					if (info.angle >= CV_PI) info.angle -= CV_PI;
					segs.push_back(info);
				}

				// Step 5b: Cluster by angle (3-degree bins for finer separation)
				//集合角度差距较小的短线段
				const double ANGLE_BIN = 5 * CV_PI / 180.0;
				std::vector<std::vector<size_t>> clusters;
				std::vector<bool> used(segs.size(), false);

				for (size_t i = 0; i < segs.size(); ++i) {
					if (used[i]) continue;
					std::vector<size_t> cluster;
					cluster.push_back(i);
					used[i] = true;
					for (size_t j = i + 1; j < segs.size(); ++j) {
						if (used[j]) continue;
						double diff = std::abs(segs[i].angle - segs[j].angle);
						if (diff > CV_PI / 2) diff = CV_PI - diff;  // handle wrap-around
						if (diff < ANGLE_BIN) {
							cluster.push_back(j);
							used[j] = true;
						}
					}
					clusters.push_back(cluster);
				}
				std::cout << "  Angle clusters: " << clusters.size() << std::endl;

				// Step 5c & 5d: For each cluster, connect endpoints and fitLine
				struct FittedLine {
					cv::Point pt;       // point on line
					cv::Point2f dir;      // direction vector
					double angle;       // angle in radians
					int numSegments;    // how many segments merged
					double totalLength; // total original length
					cv::Point2f extentMin, extentMax; // projection range
				};
				std::vector<FittedLine> fittedLines;

				cv::Mat colorImg;
				cv::cvtColor(houghSrc, colorImg, cv::COLOR_GRAY2BGR);//houghSrc图灰度转彩色
				cv::Mat linesOutput(hRows, hCols, CV_8U, cv::Scalar(0));// 创建空白二值图像
				//通过cluster引用clusters
				for (const auto& cluster : clusters) {
					if (cluster.size() < 1) continue;

					// ---- Sub-cluster by spatial proximity ----
					// Within same angle group, separate lines that are far apart
					//聚合相聚较短的短线
					std::vector<std::vector<size_t>> spatialGroups;
					std::vector<bool> usedSegs(cluster.size(), false);
					const double ENDPOINT_DIST_THRESH = 40.0;  // max distance to connect

					for (size_t si = 0; si < cluster.size(); ++si) {
						if (usedSegs[si]) continue;
						std::vector<size_t> group;
						group.push_back(cluster[si]);
						usedSegs[si] = true;

						// Grow group: add any segment with endpoint close to group
						bool changed = true;
						while (changed) {
							changed = false;
							for (size_t sj = 0; sj < cluster.size(); ++sj) {
								if (usedSegs[sj]) continue;
								// Check distance from this segment to any segment in group
								for (size_t gidx : group) {
									auto d = [](const cv::Point& a, const cv::Point& b) {
										return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
									};
									double d1 = d(segs[gidx].p1, segs[cluster[sj]].p1);
									double d2 = d(segs[gidx].p1, segs[cluster[sj]].p2);
									double d3 = d(segs[gidx].p2, segs[cluster[sj]].p1);
									double d4 = d(segs[gidx].p2, segs[cluster[sj]].p2);
									double minD = std::min({ d1, d2, d3, d4 });
									if (minD < ENDPOINT_DIST_THRESH) {
										group.push_back(cluster[sj]);
										usedSegs[sj] = true;
										changed = true;
										break;
									}
								}
							}
						}
						spatialGroups.push_back(group);
					}

					// ---- Fit a line per spatial group ----
					for (const auto& group : spatialGroups) {
						std::vector<cv::Point2f> allPts;
						double totalLen = 0;
						for (size_t idx : group) {
							allPts.push_back(cv::Point2f(segs[idx].p1));
							allPts.push_back(cv::Point2f(segs[idx].p2));
							totalLen += segs[idx].len;
						}
						if (allPts.size() < 2) continue;

						double clusterAngle = segs[group[0]].angle;

						cv::Vec4f lineParams;
						cv::fitLine(allPts, lineParams, cv::DIST_L2, 0, 0.01, 0.001);
						float vx = lineParams[0], vy = lineParams[1];
						float x0 = lineParams[2], y0 = lineParams[3];

						// Project all endpoints onto the fitted line
						//拟合直线，通过投影点确定实际长度
						std::vector<float> projections;
						for (const auto& pt : allPts) {
							float proj = (pt.x - x0) * vx + (pt.y - y0) * vy;
							projections.push_back(proj);
						}
						auto minmax = std::minmax_element(projections.begin(), projections.end());
						float tMin = *minmax.first;
						float tMax = *minmax.second;

						// Extend slightly，扩展10%直线
						float extension = (tMax - tMin) * 0.1f;
						if (extension < 5.0f) extension = 5.0f;
						tMin -= extension;
						tMax += extension;

						auto clipToImage = [&](float t) -> cv::Point {
							float x = x0 + t * vx;
							float y = y0 + t * vy;
							float tMinClip = t, tMaxClip = t;
							if (std::abs(vx) > 1e-6) {
								tMinClip = std::min(tMinClip, (0.0f - x0) / vx);//先和0，x0的投影比较
								tMaxClip = std::max(tMaxClip, (float(hCols - 1) - x0) / vx);
							}
							if (std::abs(vy) > 1e-6) {
								tMinClip = std::min(tMinClip, (0.0f - y0) / vy);//再和0，y0的投影比较
								tMaxClip = std::max(tMaxClip, (float(hRows - 1) - y0) / vy);
							}
							float tClipped = std::max(tMinClip, std::min(tMaxClip, t));//择中
							return cv::Point(cvRound(x0 + tClipped * vx), cvRound(y0 + tClipped * vy));
						};

						cv::Point p1 = clipToImage(tMin);
						cv::Point p2 = clipToImage(tMax);
						//std::cout << "Vx" << vx << "  " << "Vy"<<vy << std::endl;
						FittedLine fl;
						fl.pt = cv::Point(cvRound(x0), cvRound(y0));
						fl.dir = cv::Point2f(vx, vy );
						fl.angle = clusterAngle;
						fl.numSegments = static_cast<int>(group.size());
						fl.totalLength = totalLen;
						fittedLines.push_back(fl);

						// Draw fitted line in red
						cv::line(colorImg, p1, p2, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
						// Draw original segments in green
						for (size_t idx : group) {
							cv::line(colorImg, segs[idx].p1, segs[idx].p2,
								cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
						}
						// Draw fitted line on binary output
						if(fl.totalLength>70)
						cv::line(linesOutput, p1, p2, cv::Scalar(255), 1, cv::LINE_AA);
					}
				}

				std::cout << "  Fitted lines: " << fittedLines.size() << std::endl;
				double grandTotal = 0;
				for (const auto& fl : fittedLines) {
					grandTotal += fl.totalLength;
					std::cout << "    angle=" << (fl.angle * 180 / CV_PI) << "deg"
						<< " segments=" << fl.numSegments
						<< " totalLen=" << (int)fl.totalLength << "px"
						<< std::endl;
				}
				std::cout << "  Grand total fitted length: " << (int)grandTotal << "px" << std::endl;

				cv::imwrite("hough_clustered_overlay.png", colorImg);
				cv::imwrite("hough_clustered_lines.png", linesOutput);
				std::cout << "  Saved: hough_clustered_overlay.png" << std::endl;
				std::cout << "  Saved: hough_clustered_lines.png" << std::endl;

				// ============================================================
				// Step 8: 从 fittedLines 中识别5条主要直线，输出方程和交点
				// ============================================================
				{
					std::cout << "\n=== Top 5 Lines Analysis ===" << std::endl;

					// 按总长度降序排序，取前5条
					std::vector<FittedLine> topLines = fittedLines;
					std::sort(topLines.begin(), topLines.end(),
						[](const FittedLine& a, const FittedLine& b) {
							return a.totalLength > b.totalLength;
						});
					if (topLines.size() > 5)
						topLines.resize(5);

					if (topLines.empty()) {
						std::cout << "  No lines detected." << std::endl;
					}
					else {
						// 存储每条直线的方程参数 A*x + B*y + C = 0
						struct LineEq {
							double A, B, C;
							double angleDeg;
							double length;
						};
						std::vector<LineEq> eqs;

						for (size_t i = 0; i < topLines.size(); ++i) {
							const auto& fl = topLines[i];

							// 还原方向向量 
							double vx = fl.dir.x;
							double vy = fl.dir.y;
							double x0 = static_cast<double>(fl.pt.x);
							double y0 = static_cast<double>(fl.pt.y);
							//std::cout << "X0" << x0 << "  " << "Y0"<<y0 << std::endl;
							// 直线一般式: A*x + B*y + C = 0
							double A = vy;
							double B = -vx;
							double C = -vy * x0 + vx * y0;

							double norm = std::sqrt(A * A + B * B);
							if (norm > 1e-12) {
								A /= norm;
								B /= norm;
								C /= norm;
							}

							double angleDeg = fl.angle * 180.0 / CV_PI;
							eqs.push_back({ A, B, C, angleDeg, fl.totalLength });

							std::cout << "  Line " << (i + 1) << ": "
								<< std::fixed << std::setprecision(4)
								<< A << "x + " << B << "y + " << C << " = 0"
								<< "  (angle=" << angleDeg << "deg, length="
								<< (int)fl.totalLength << "px)" << std::endl;
						}

						// 计算所有两两交点
						std::cout << "\n  Intersection Points:" << std::endl;
						bool anyIntersection = false;
						struct Points{
							double x;
							double y;
							double Angle_1;
							double Angle_2;
						};
						std::vector<Points> intersectionPoints;
						for (size_t i = 0; i < eqs.size(); ++i) {
							for (size_t j = i + 1; j < eqs.size(); ++j) {
								double det = eqs[i].A * eqs[j].B - eqs[j].A * eqs[i].B;
								if (std::abs(det) < 1e-12) {
									std::cout << "    Line" << (i + 1) << " x Line" << (j + 1)
										<< ": parallel (no intersection)" << std::endl;
									continue;
								}
								double x = (eqs[i].B * eqs[j].C - eqs[j].B * eqs[i].C) / det;
								double y = (eqs[j].A * eqs[i].C - eqs[i].A * eqs[j].C) / det;
								intersectionPoints.push_back({ x,y,eqs[i].angleDeg,eqs[j].angleDeg });
								anyIntersection = true;
								std::cout << "    Angle_1: " << eqs[i].angleDeg << " x Angle_2: " << eqs[j].angleDeg << " = ("
									<< std::fixed << std::setprecision(2) << x << ", "
									<< std::setprecision(2) << y << ")" << std::endl;
							}
						}
						cv::Point2d Q1, Q2;
						bool isValid_Q1 = false, isValid_Q2= false;
						for (auto it = intersectionPoints.begin(); it != intersectionPoints.end(); ++it) {
							if ((abs(105 - it->Angle_1) < 5 && abs(87 - it->Angle_2) < 2)
								|| (abs(105 - it->Angle_2) < 5 && abs(87 - it->Angle_1) < 2))
							{
								Q1.x = it->x;
								Q1.y = it->y;
								isValid_Q1 = true;
								std::cout << "Q1.x: " << Q1.x << " Q1.y: " << Q1.y << std::endl;
							}
							else if ((abs(93- it->Angle_1) < 4 && abs(78 - it->Angle_2) < 5)
								|| (abs(93 - it->Angle_2) < 4 && abs(78 - it->Angle_1) < 5))
							{
								Q2.x = it->x;
								Q2.y = it->y;
								isValid_Q2 = true;
								std::cout << "Q2.x: " << Q2.x << " Q2.y: " << Q2.y << std::endl;
							}
						}
						LineEquation line1 = calculateLine(Q1, Q2);
						printLineInfo(line1, "直线1");

						if (!anyIntersection) {
							std::cout << "    (all lines are parallel)" << std::endl;
						}
					}
				}
			}


		// ============================================================
		// Step 7: Validate — pixel-wise sum of edge map + Hough lines
		// ============================================================
		
			std::cout << "\n=== Validation: edge_map + hough_lines (direct sum) ===" << std::endl;

			int mh = static_cast<int>(multiScaleEdge.size());
			int mw = static_cast<int>(multiScaleEdge[0].size());

			// Normalize multiScaleEdge to 8-bit
			cv::Mat edgeNorm(mh, mw, CV_8U, cv::Scalar(0));
			double maxVal = 0;
			for (int i = 0; i < mh; ++i)
				for (int j = 0; j < mw; ++j)
				{
					if (multiScaleEdge[i][j] > 0.3)
						edgeNorm.at<uchar>(i, j) = 255;
					else
						edgeNorm.at<uchar>(i, j) = 0;
				}

			// Load Hough lines (binary)
			cv::Mat houghLines = cv::imread("hough_clustered_lines.png", cv::IMREAD_GRAYSCALE);
			if (!houghLines.empty()) {
				if (houghLines.size() != edgeNorm.size())
					cv::resize(houghLines, houghLines, edgeNorm.size(), 0, 0, cv::INTER_NEAREST);

				cv::Mat sum(mh, mw, CV_8U, cv::Scalar(0));

				for (int i = 0; i < mh; ++i)
					for (int j = 0; j < mw; ++j)
					{
						sum.at<uchar>(i, j) = edgeNorm.at<uchar>(i, j) + houghLines.at<uchar>(i, j);
					}
				cv::imwrite("validation_sum.png", sum);
			}
			else {
				std::cout << "  WARNING: hough_clustered_lines.png not found." << std::endl;
			}
		

		}

		system("pause");
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	return 0;
	
}
