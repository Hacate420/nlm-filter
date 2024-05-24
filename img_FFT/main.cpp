#include<iostream>
#include<opencv2/opencv.hpp>
#include<cmath>
#include"MY_DFT.h"

using namespace cv;
using namespace std;

//int main()
//{
//	Mat image, image_gray, image_output, image_transform;   //定义输入图像，灰度图像，输出图像
//	//image = imread("F:\\偏振成像\\测试数据\\数据采集\\组合\\左右弧形\\1.bmp"); //读取图像；
//	//image = imread("F:\\偏振成像\\测试数据\\数据采集\\组合\\前后弧形\\1.bmp"); //读取图像；
//	image = imread("F:\\偏振成像\\测试数据\\数据采集\\金属6\\Mono8_Stoke_DoP_20_28_2.bmp");
//	//image = imread("F:\\偏振成像\\测试数据\\数据采集\\金属5\\Mono8_Stoke_DoP_20_26_53.bmp");
//	//image = imread("F:\\偏振成像\\测试数据\\数据采集\\金属7\\Mono8_Stoke_DoP_21_26_20.bmp");
//	
//
//
//	//Mat image, image_gray, image_output, image_transform;   //定义输入图像，灰度图像，输出图像
//	if (image.empty())
//	{
//		cout << "读取错误" << endl;
//		return -1;
//	}
//	namedWindow("image", WINDOW_NORMAL);
//	namedWindow("image_gray", WINDOW_NORMAL);
//	namedWindow("image_output", WINDOW_NORMAL);
//	namedWindow("idft", WINDOW_NORMAL);
//	namedWindow("test", WINDOW_NORMAL);
//	
//	imshow("image", image);
//
//	cvtColor(image, image_gray, COLOR_BGR2GRAY); //转换为灰度图
//	imshow("image_gray", image_gray); //显示灰度图
//
//	//1、傅里叶变换，image_output为可显示的频谱图，image_transform为傅里叶变换的复数结果
//	My_DFT(image_gray, image_output, image_transform);
//	imshow("image_output", image_output);
//
//	//2、高斯高通滤波
//	Mat planes[] = { Mat_<float>(image_output), Mat::zeros(image_output.size(),CV_32F) };
//	split(image_transform, planes);//分离通道，获取实部虚部
//	Mat image_transform_real = planes[0];
//	Mat image_transform_imag = planes[1];
//
//	int core_x = image_transform_real.rows / 2;//频谱图中心坐标
//	int core_y = image_transform_real.cols / 2;
//	//较大的滤波半径会导致更多的低频成分被保留，而较小的滤波半径则强调更高的频率成分
//	int r = 500;  //滤波半径
//	float h;
//	for (int i = 0; i < image_transform_real.rows; i++)
//	{
//		for (int j = 0; j < image_transform_real.cols; j++)
//		{
//			h = 1 - exp(-((i - core_x) * (i - core_x) + (j - core_y) * (j - core_y)) / (2 * r * r));
//			image_transform_real.at<float>(i, j) = image_transform_real.at<float>(i, j) * h;
//			image_transform_imag.at<float>(i, j) = image_transform_imag.at<float>(i, j) * h;
//
//		}
//	}
//	planes[0] = image_transform_real;
//	planes[1] = image_transform_imag;
//	Mat image_transform_ilpf;//定义高斯高通滤波结果
//	merge(planes, 2, image_transform_ilpf);
//
//	//3、傅里叶逆变换
//	Mat iDft[] = { Mat_<float>(image_output), Mat::zeros(image_output.size(),CV_32F) };
//	idft(image_transform_ilpf, image_transform_ilpf);//傅立叶逆变换
//	split(image_transform_ilpf, iDft);//分离通道，主要获取0通道
//	magnitude(iDft[0], iDft[1], iDft[0]); //计算复数的幅值，保存在iDft[0]
//	normalize(iDft[0], iDft[0], 0, 1, NORM_MINMAX);//归一化处理
//	imshow("idft", iDft[0]);//显示逆变换图像
//	imwrite("./idft.bmp", iDft[0]);
//	Mat idft = imread("./idft.bmp");
//	resize(idft, idft, Size(1224, 1024));
//	Mat test = image - idft;
//	imshow("test", test);
//
//	waitKey(0);  //暂停，保持图像显示，等待按键结束
//	return 0;
//
//}

