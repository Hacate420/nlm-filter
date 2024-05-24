#include "MY_DFT.h"

//����Ҷ�任�õ�Ƶ��ͼ�͸�������
void My_DFT(Mat input_image, Mat& output_image, Mat& transform_image)
{
	//1.��չͼ�����Ϊ2��3��5�ı���ʱ�����ٶȿ�
	int m = getOptimalDFTSize(input_image.rows);
	int n = getOptimalDFTSize(input_image.cols);
	copyMakeBorder(input_image, input_image, 0, m - input_image.rows, 0, n - input_image.cols, BORDER_CONSTANT, Scalar::all(0));

	//2.����һ��˫ͨ������planes���������渴����ʵ�����鲿
	Mat planes[] = { Mat_<float>(input_image), Mat::zeros(input_image.size(), CV_32F) };

	//3.�Ӷ����ͨ�������д���һ����ͨ������:transform_image������Merge����������ϲ�Ϊһ����ͨ�����У�����������ÿ��Ԫ�ؽ�����������Ԫ�صļ���
	merge(planes, 2, transform_image);

	//4.���и���Ҷ�任
	dft(transform_image, transform_image);

	//5.���㸴���ķ�ֵ��������output_image��Ƶ��ͼ��
	split(transform_image, planes); // ��˫ͨ����Ϊ������ͨ����һ����ʾʵ����һ����ʾ�鲿
	Mat transform_image_real = planes[0];
	Mat transform_image_imag = planes[1];

	magnitude(planes[0], planes[1], output_image); //���㸴���ķ�ֵ��������output_image��Ƶ��ͼ��

	//6.ǰ��õ���Ƶ��ͼ�������󣬲�����ʾ�����ת��
	output_image += Scalar(1);   // ȡ����ǰ�����е����ض���1����ֹlog0
	log(output_image, output_image);   // ȡ����
	normalize(output_image, output_image, 0, 1, NORM_MINMAX); //��һ��

	//7.���к��طֲ�����ͼ����
	output_image = output_image(Rect(0, 0, output_image.cols & -2, output_image.rows & -2));

	// �������и���Ҷͼ���е����ޣ�ʹԭ��λ��ͼ������
	int cx = output_image.cols / 2;
	int cy = output_image.rows / 2;
	Mat q0(output_image, Rect(0, 0, cx, cy));   // ��������
	Mat q1(output_image, Rect(cx, 0, cx, cy));  // ��������
	Mat q2(output_image, Rect(0, cy, cx, cy));  // ��������
	Mat q3(output_image, Rect(cx, cy, cx, cy)); // ��������

	  //�����������Ļ�
	Mat tmp;
	q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);//���������½��н���
	q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);//���������½��н���


	Mat q00(transform_image_real, Rect(0, 0, cx, cy));   // ��������
	Mat q01(transform_image_real, Rect(cx, 0, cx, cy));  // ��������
	Mat q02(transform_image_real, Rect(0, cy, cx, cy));  // ��������
	Mat q03(transform_image_real, Rect(cx, cy, cx, cy)); // ��������
	q00.copyTo(tmp); q03.copyTo(q00); tmp.copyTo(q03);//���������½��н���
	q01.copyTo(tmp); q02.copyTo(q01); tmp.copyTo(q02);//���������½��н���

	Mat q10(transform_image_imag, Rect(0, 0, cx, cy));   // ��������
	Mat q11(transform_image_imag, Rect(cx, 0, cx, cy));  // ��������
	Mat q12(transform_image_imag, Rect(0, cy, cx, cy));  // ��������
	Mat q13(transform_image_imag, Rect(cx, cy, cx, cy)); // ��������
	q10.copyTo(tmp); q13.copyTo(q10); tmp.copyTo(q13);//���������½��н���
	q11.copyTo(tmp); q12.copyTo(q11); tmp.copyTo(q12);//���������½��н���

	planes[0] = transform_image_real;
	planes[1] = transform_image_imag;
	merge(planes, 2, transform_image);//������Ҷ�任������Ļ�
}
