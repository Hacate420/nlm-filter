#include<iostream>
#include<opencv2/opencv.hpp>
#include<ctime>
using namespace std;
using namespace cv;

void Bilateral_Filter(Mat src, Mat& dst, int ksize, double sita_r, double sita_s);//˫���˲�


//int main()
//{
//	Mat src = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\���\\ǰ����\\3.bmp");
//	//src = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\���\\����\\2.bmp", 1);
//	src = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\Mono8_Stoke_DoP_20_28_2.bmp");
//	src = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����5\\Mono8_Stoke_DoP_20_26_53.bmp");
//	//Mat blackimg = imread("black.bmp", 1);
//	Mat dst;
//	Mat dst2;
//	//Bilateral_Filter(src, dst, 10, 50, 5);
//	//bilateralfiter(src, dst, Size_<int>(20,20), 10, 35);//�ܲ�ͨ
//	bilateralFilter(src, dst, 10, 50, 5);//opencv
//	bilateralFilter(dst, dst2, 10, 50, 5);//opencv
//	//cv::GaussianBlur(src, dst, cv::Size(5, 5), 5);//��˹�˲�
//	namedWindow("src", WINDOW_NORMAL);
//	namedWindow("dst", WINDOW_NORMAL);
//	imshow("src", src);
//	imshow("dst", dst2);
//	waitKey(0);
//	imwrite("./˫��_10_50_5_3.bmp", dst);
//	imwrite("./����˫��_10_50_5_3.bmp", dst2);
//	return 0;
//}




//��ȡɫ��ģ�壨ֵ��ģ�壩
///
void getColorMask(std::vector<double>& colorMask, double colorSigma) {

	for (int i = 0; i < 256; ++i) {
		double colordiff = exp(-(i * i) / (2 * colorSigma * colorSigma));
		colorMask.push_back(colordiff);
	}

}


//��ȡ��˹ģ�壨�ռ�ģ�壩
///
void getGausssianMask(cv::Mat& Mask, cv::Size wsize, double spaceSigma) {
	Mask.create(wsize, CV_64F);
	int h = wsize.height;
	int w = wsize.width;
	int center_h = (h - 1) / 2;
	int center_w = (w - 1) / 2;
	double sum = 0.0;
	double x, y;

	for (int i = 0; i < h; ++i) {
		y = pow(i - center_h, 2);
		double* Maskdate = Mask.ptr<double>(i);
		for (int j = 0; j < w; ++j) {
			x = pow(j - center_w, 2);
			double g = exp(-(x + y) / (2 * spaceSigma * spaceSigma));
			Maskdate[j] = g;
			sum += g;
		}
	}
}

//˫���˲�
///
void bilateralfiter(cv::Mat& src, cv::Mat& dst, cv::Size wsize, double spaceSigma, double colorSigma) {
	cv::Mat spaceMask;
	std::vector<double> colorMask;
	cv::Mat Mask0 = cv::Mat::zeros(wsize, CV_64F);
	cv::Mat Mask1 = cv::Mat::zeros(wsize, CV_64F);
	cv::Mat Mask2 = cv::Mat::zeros(wsize, CV_64F);

	getGausssianMask(spaceMask, wsize, spaceSigma);//�ռ�ģ��
	getColorMask(colorMask, colorSigma);//ֵ��ģ��
	int hh = (wsize.height - 1) / 2;
	int ww = (wsize.width - 1) / 2;
	dst.create(src.size(), src.type());
	//�߽����
	cv::Mat Newsrc;
	cv::copyMakeBorder(src, Newsrc, hh, hh, ww, ww, cv::BORDER_REPLICATE);//�߽縴��;

	for (int i = hh; i < src.rows + hh; ++i) {
		for (int j = ww; j < src.cols + ww; ++j) {
			double sum[3] = { 0 };
			int graydiff[3] = { 0 };
			double space_color_sum[3] = { 0.0 };

			for (int r = -hh; r <= hh; ++r) {
				for (int c = -ww; c <= ww; ++c) {
					if (src.channels() == 1) {
						int centerPix = Newsrc.at<uchar>(i, j);
						int pix = Newsrc.at<uchar>(i + r, j + c);
						graydiff[0] = abs(pix - centerPix);
						double colorWeight = colorMask[graydiff[0]];
						Mask0.at<double>(r + hh, c + ww) = colorWeight * spaceMask.at<double>(r + hh, c + ww);//�˲�ģ��
						space_color_sum[0] = space_color_sum[0] + Mask0.at<double>(r + hh, c + ww);

					}
					else if (src.channels() == 3) {
						cv::Vec3b centerPix = Newsrc.at<cv::Vec3b>(i, j);
						cv::Vec3b bgr = Newsrc.at<cv::Vec3b>(i + r, j + c);
						graydiff[0] = abs(bgr[0] - centerPix[0]); graydiff[1] = abs(bgr[1] - centerPix[1]); graydiff[2] = abs(bgr[2] - centerPix[2]);
						double colorWeight0 = colorMask[graydiff[0]];
						double colorWeight1 = colorMask[graydiff[1]];
						double colorWeight2 = colorMask[graydiff[2]];
						Mask0.at<double>(r + hh, c + ww) = colorWeight0 * spaceMask.at<double>(r + hh, c + ww);//�˲�ģ��
						Mask1.at<double>(r + hh, c + ww) = colorWeight1 * spaceMask.at<double>(r + hh, c + ww);
						Mask2.at<double>(r + hh, c + ww) = colorWeight2 * spaceMask.at<double>(r + hh, c + ww);
						space_color_sum[0] = space_color_sum[0] + Mask0.at<double>(r + hh, c + ww);
						space_color_sum[1] = space_color_sum[1] + Mask1.at<double>(r + hh, c + ww);
						space_color_sum[2] = space_color_sum[2] + Mask2.at<double>(r + hh, c + ww);

					}
				}
			}

			//�˲�ģ���һ��
			if (src.channels() == 1)
				Mask0 = Mask0 / space_color_sum[0];
			else {
				Mask0 = Mask0 / space_color_sum[0];
				Mask1 = Mask1 / space_color_sum[1];
				Mask2 = Mask2 / space_color_sum[2];
			}


			for (int r = -hh; r <= hh; ++r) {
				for (int c = -ww; c <= ww; ++c) {

					if (src.channels() == 1) {
						sum[0] = sum[0] + Newsrc.at<uchar>(i + r, j + c) * Mask0.at<double>(r + hh, c + ww); //�˲�
					}
					else if (src.channels() == 3) {
						cv::Vec3b bgr = Newsrc.at<cv::Vec3b>(i + r, j + c); //�˲�
						sum[0] = sum[0] + bgr[0] * Mask0.at<double>(r + hh, c + ww);//B
						sum[1] = sum[1] + bgr[1] * Mask1.at<double>(r + hh, c + ww);//G
						sum[2] = sum[2] + bgr[2] * Mask2.at<double>(r + hh, c + ww);//R
					}
				}
			}

			for (int k = 0; k < src.channels(); ++k) {
				if (sum[k] < 0)
					sum[k] = 0;
				else if (sum[k] > 255)
					sum[k] = 255;
			}
			if (src.channels() == 1)
			{
				dst.at<uchar>(i - hh, j - ww) = static_cast<uchar>(sum[0]);
			}
			else if (src.channels() == 3)
			{
				cv::Vec3b bgr = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
				dst.at<cv::Vec3b>(i - hh, j - ww) = bgr;
			}

		}
	}

}



//˫���˲�
void Bilateral_Filter(Mat src, Mat& dst, int ksize, double sita_r, double sita_s)
{
	int h = src.rows;
	int w = src.cols;
	int k = ksize / 2;
	dst = Mat::zeros(h, w, CV_32FC3);
	Mat ws_mat(ksize, ksize, CV_32FC1);//����Ȩ�ؾ���
	for (int i = 0; i < ksize; i++)
	{
		for (int j = 0; j < ksize; j++)
		{
			ws_mat.at<float>(i, j) = exp(-1.0 * ((i - k) * (i - k) + (j - k) * (j - k)) / (2 * sita_s * sita_s));
		}
	}

	for (int i = k; i < h - k; i++)
	{
		for (int j = k; j < w - k; j++)
		{
			Mat roi = src(Rect(j - k, i - k, ksize, ksize));//��ȡĿ�����ΧKsize*ksize����
			Mat wc_mat(ksize, ksize, CV_32FC1);//ֵ��Ȩ�ؾ���
			Mat w_mat(ksize, ksize, CV_32FC1); //��Ȩ�ؾ���
			for (int ri = 0; ri < ksize; ri++)
			{
				for (int rj = 0; rj < ksize; rj++)
				{
					int temp = abs(roi.at<Vec3b>(ri, rj)[0] + roi.at<Vec3b>(ri, rj)[1] + roi.at<Vec3b>(ri, rj)[2]
						- roi.at<Vec3b>(k, k)[0] - roi.at<Vec3b>(k, k)[1] - roi.at<Vec3b>(k, k)[2]);
					temp = -1.0 * (temp * temp) / (2 * sita_r * sita_r);
					wc_mat.at<float>(ri, rj) = exp(temp);//ֵ��Ȩֵ
					w_mat.at<float>(ri, rj) = ws_mat.at<float>(ri, rj) * wc_mat.at<float>(ri, rj);//����Ȩֵ
					for (int channel = 0; channel < 3; channel++)
					{
						dst.at<Vec3f>(i, j)[channel] += 1.0 * w_mat.at<float>(ri, rj) * roi.at<Vec3b>(ri, rj)[channel];//Ŀ������ֵ=��Χ������ֵ*���Ե�Ȩ��
					}
				}
			}
			float w_mat_sum = sum(w_mat)[0];//Ȩ��֮��
			dst.at<Vec3f>(i, j) = dst.at<Vec3f>(i, j) / w_mat_sum;//Ŀ�������ֵ/Ȩ��֮�ͣ��൱��Ȩ�ع�һ������Ϊ1����
		}
	}
	normalize(dst, dst, 0, 255, NORM_MINMAX, CV_8UC3);//���ͼ���һ��
	dst = dst(Rect(k, k, dst.cols - 2 * k, dst.rows - 2 * k));
	//�߽紦�������и��õĴ���ʽ���˴�Ϊ��������
	vector<Mat> dst_bgr(3);
	split(dst, dst_bgr);
	for (int i = 0; i < 3; i++)
	{
		copyMakeBorder(dst_bgr[i], dst_bgr[i], k, k, k, k, BORDER_REPLICATE);
	}
	merge(dst_bgr, dst);
}
