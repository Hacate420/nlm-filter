#include<iostream>
#include<opencv2/opencv.hpp>
#include<cmath>
#include"MY_DFT.h"

using namespace cv;
using namespace std;

//int main()
//{
//	Mat image, image_gray, image_output, image_transform;   //��������ͼ�񣬻Ҷ�ͼ�����ͼ��
//	//image = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\���\\���һ���\\1.bmp"); //��ȡͼ��
//	//image = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\���\\ǰ����\\1.bmp"); //��ȡͼ��
//	image = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����6\\Mono8_Stoke_DoP_20_28_2.bmp");
//	//image = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����5\\Mono8_Stoke_DoP_20_26_53.bmp");
//	//image = imread("F:\\ƫ�����\\��������\\���ݲɼ�\\����7\\Mono8_Stoke_DoP_21_26_20.bmp");
//	
//
//
//	//Mat image, image_gray, image_output, image_transform;   //��������ͼ�񣬻Ҷ�ͼ�����ͼ��
//	if (image.empty())
//	{
//		cout << "��ȡ����" << endl;
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
//	cvtColor(image, image_gray, COLOR_BGR2GRAY); //ת��Ϊ�Ҷ�ͼ
//	imshow("image_gray", image_gray); //��ʾ�Ҷ�ͼ
//
//	//1������Ҷ�任��image_outputΪ����ʾ��Ƶ��ͼ��image_transformΪ����Ҷ�任�ĸ������
//	My_DFT(image_gray, image_output, image_transform);
//	imshow("image_output", image_output);
//
//	//2����˹��ͨ�˲�
//	Mat planes[] = { Mat_<float>(image_output), Mat::zeros(image_output.size(),CV_32F) };
//	split(image_transform, planes);//����ͨ������ȡʵ���鲿
//	Mat image_transform_real = planes[0];
//	Mat image_transform_imag = planes[1];
//
//	int core_x = image_transform_real.rows / 2;//Ƶ��ͼ��������
//	int core_y = image_transform_real.cols / 2;
//	//�ϴ���˲��뾶�ᵼ�¸���ĵ�Ƶ�ɷֱ�����������С���˲��뾶��ǿ�����ߵ�Ƶ�ʳɷ�
//	int r = 500;  //�˲��뾶
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
//	Mat image_transform_ilpf;//�����˹��ͨ�˲����
//	merge(planes, 2, image_transform_ilpf);
//
//	//3������Ҷ��任
//	Mat iDft[] = { Mat_<float>(image_output), Mat::zeros(image_output.size(),CV_32F) };
//	idft(image_transform_ilpf, image_transform_ilpf);//����Ҷ��任
//	split(image_transform_ilpf, iDft);//����ͨ������Ҫ��ȡ0ͨ��
//	magnitude(iDft[0], iDft[1], iDft[0]); //���㸴���ķ�ֵ��������iDft[0]
//	normalize(iDft[0], iDft[0], 0, 1, NORM_MINMAX);//��һ������
//	imshow("idft", iDft[0]);//��ʾ��任ͼ��
//	imwrite("./idft.bmp", iDft[0]);
//	Mat idft = imread("./idft.bmp");
//	resize(idft, idft, Size(1224, 1024));
//	Mat test = image - idft;
//	imshow("test", test);
//
//	waitKey(0);  //��ͣ������ͼ����ʾ���ȴ���������
//	return 0;
//
//}

