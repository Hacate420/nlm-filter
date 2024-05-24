//实际还是非局部均值滤波
#include<iostream>
#include<opencv2/opencv.hpp>
#include<ctime>
using namespace std;
using namespace cv;

//计算0~255的平方查找表
float table1[256];
static void cal_lookup_table1(void)
{
    for (int i = 0; i < 256; i++)
    {
        table1[i] = (float)(i * i);
    }
}


//计算两个0~255的数的绝对差值的查找表
uchar table2[256][256];
static void cal_lookup_table2(void)
{
    for (int i = 0; i < 256; i++)
    {
        for (int j = i; j < 256; j++)
        {
            table2[i][j] = abs(i - j);
            table2[j][i] = table2[i][j];
        }
    }
}

//计算两个邻域块的MSE
float MSE_block(Mat m1, Mat m2)
{
    float sum = 0.0;
    for (int j = 0; j < m1.rows; j++)
    {
        uchar* data1 = m1.ptr<uchar>(j);
        uchar* data2 = m2.ptr<uchar>(j);
        for (int i = 0; i < m1.cols; i++)
        {
            sum += table1[table2[data1[i]][data2[i]]];
        }
    }
    sum = sum / (m1.rows * m2.cols);
    return sum;
}

//h越大越平滑
//halfKernelSize小框
//halfSearchSize大框
void NL_mean(Mat src, Mat& dst, double h, int halfKernelSize, int halfSearchSize)
{
    Mat boardSrc;
    dst.create(src.rows, src.cols, CV_8UC1);
    int boardSize = halfKernelSize + halfSearchSize;
    copyMakeBorder(src, boardSrc, boardSize, boardSize, boardSize, boardSize, BORDER_REFLECT);   //边界扩展
    double h2 = h * h;


    int rows = src.rows;
    int cols = src.cols;


    cal_lookup_table1();
    cal_lookup_table2();


    for (int j = boardSize; j < boardSize + rows; j++)
    {
        uchar* dst_p = dst.ptr<uchar>(j - boardSize);
        for (int i = boardSize; i < boardSize + cols; i++)
        {
            Mat patchA = boardSrc(Range(j - halfKernelSize, j + halfKernelSize), Range(i - halfKernelSize, i + halfKernelSize));
            double w = 0;
            double p = 0;
            double sumw = 0;


            for (int sr = -halfSearchSize; sr <= halfSearchSize; sr++)   //在搜索框内滑动
            {
                uchar* boardSrc_p = boardSrc.ptr<uchar>(j + sr);
                for (int sc = -halfSearchSize; sc <= halfSearchSize; sc++)
                {
                    Mat patchB = boardSrc(Range(j + sr - halfKernelSize, j + sr + halfKernelSize), Range(i + sc - halfKernelSize, i + sc + halfKernelSize));
                    float d2 = MSE_block(patchA, patchB);


                    w = exp(-d2 / h2);
                    p += boardSrc_p[i + sc] * w;
                    sumw += w;
                }
            }


            dst_p[i - boardSize] = saturate_cast<uchar>(p / sumw);


        }
    }


}

//int main(void)
//{
//    Mat img = imread("F:\\偏振成像\\测试数据\\数据采集\\组合\\前后弧形\\3.bmp", CV_LOAD_IMAGE_GRAYSCALE);
//
//    Mat out;
//    NL_mean(img, out, 3, 7, 21);   //NL-means
//
//    Mat out1;
//    blur(img, out1, Size(11, 11));   //均值滤波
//
//
//    Mat out2;
//    GaussianBlur(img, out2, Size(11, 11), 2.5);   //高斯滤波
//
//
//    Mat out3;
//    medianBlur(img, out3, 9);   //中值滤波
//
//    namedWindow("img", WINDOW_NORMAL);
//    namedWindow("非局部", WINDOW_NORMAL);
//    namedWindow("均值", WINDOW_NORMAL);
//    namedWindow("高斯", WINDOW_NORMAL);
//    namedWindow("中值", WINDOW_NORMAL);
//    imshow("img", img);
//    imshow("非局部", out);
//    imshow("均值", out1);
//    imshow("高斯", out2);
//    imshow("中值", out3);
//    waitKey();
//}
