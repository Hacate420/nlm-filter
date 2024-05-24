#include<iostream>
#include<opencv2/opencv.hpp>
#include<ctime>
using namespace std;
using namespace cv;

// 快速NL-means算法主体函数
void fastNLmeans(Mat src, Mat& dst, int ds, int Ds, float h);
//基于Mat指针实现
Mat NonLocalMeansFilter2(const Mat& src, int searchWindowSize, int templateWindowSize, double sigma, double h);

int main()
{
	Mat src = imread("F:\\偏振成像\\测试数据\\数据采集\\组合\\前后弧形\\3.bmp");
	//src = imread("F:\\偏振成像\\测试数据\\数据采集\\组合\\左右\\2.bmp", 1);
	src = imread("F:\\偏振成像\\测试数据\\数据采集\\金属6\\Mono8_Stoke_DoP_20_28_2.bmp");
	//src = imread("F:\\偏振成像\\测试数据\\数据采集\\金属5\\Mono8_Stoke_DoP_20_26_53.bmp");
	//Mat blackimg = imread("black.bmp", 1);
	Mat dst;
	Mat dst2;
	//fastNlMeansDenoisingColored(src, dst, 5, 5, 7, 21);//彩色图像非局部均值滤波
	//fastNlMeansDenoising(src, dst, 8, 10, 30);//灰度图像非局部均值滤波
	//fastNlMeansDenoising(dst, dst2, 8, 10, 30);//灰度图像非局部均值滤波
    fastNlMeansDenoisingMulti(src, dst, 8, 10, 30);//灰度图像非局部均值滤波
	fastNLmeans(src, dst, 7, 21, 3);//得到的结果是一张部分图，运行速度3.30
	namedWindow("src", WINDOW_NORMAL);
	namedWindow("dst", WINDOW_NORMAL);
	imshow("src", src);
	imshow("dst", dst);
	waitKey(0);
	/*imwrite("./非局部均值_8_10_30_1.bmp", dst);
	imwrite("./二次非局部均值_8_10_30_1.bmp", dst2);*/
	return 0;
}

//计算积分图
void integralImgSqDiff(Mat src, Mat& dst, int Ds, int t1, int t2, int m1, int n1)
{
    //计算图像A与图像B的差值图C
    Mat Dist2 = src(Range(Ds, src.rows - Ds), Range(Ds, src.cols - Ds)) - src(Range(Ds + t1, src.rows - Ds + t1), Range(Ds + t2, src.cols - Ds + t2));
    float* Dist2_data;
    for (int i = 0; i < m1; i++)
    {
        Dist2_data = Dist2.ptr<float>(i);
        for (int j = 0; j < n1; j++)
        {
            Dist2_data[j] *= Dist2_data[j];  //计算图像C的平方图D
        }
    }
    integral(Dist2, dst, CV_32F); //计算图像D的积分图
}

//快速NL-means算法主体函数代码
void fastNLmeans(Mat src, Mat& dst, int ds, int Ds, float h)
{
    Mat src_tmp;
    src.convertTo(src_tmp, CV_32F);
    int m = src_tmp.rows;
    int n = src_tmp.cols;
    int boardSize = Ds + ds + 1;
    Mat src_board;
    copyMakeBorder(src_tmp, src_board, boardSize, boardSize, boardSize, boardSize, BORDER_REFLECT);

    Mat average(m, n, CV_32FC1, 0.0);
    Mat sweight(m, n, CV_32FC1, 0.0);

    float h2 = h * h;
    int d2 = (2 * ds + 1) * (2 * ds + 1);

    int m1 = src_board.rows - 2 * Ds;   //行
    int n1 = src_board.cols - 2 * Ds;   //列
    Mat St(m1, n1, CV_32FC1, 0.0);

    for (int t1 = -Ds; t1 <= Ds; t1++)
    {
        int Dst1 = Ds + t1;
        for (int t2 = -Ds; t2 <= Ds; t2++)
        {
            int Dst2 = Ds + t2;
            integralImgSqDiff(src_board, St, Ds, t1, t2, m1, n1);

            for (int i = 0; i < m; i++)
            {
                float* sweight_p = sweight.ptr<float>(i);
                float* average_p = average.ptr<float>(i);
                float* v_p = src_board.ptr<float>(i + Ds + t1 + ds);
                int i1 = i + ds + 1;   //row
                float* St_p1 = St.ptr<float>(i1 + ds);
                float* St_p2 = St.ptr<float>(i1 - ds - 1);

                for (int j = 0; j < n; j++)
                {

                    int j1 = j + ds + 1;   //col
                    float Dist2 = (St_p1[j1 + ds] + St_p2[j1 - ds - 1]) - (St_p1[j1 - ds - 1] + St_p2[j1 + ds]);

                    Dist2 /= (-d2 * h2);
                    float w = exp(Dist2);
                    sweight_p[j] += w;
                    average_p[j] += w * v_p[j + Ds + t2 + ds];
                }
            }

        }
    }

    average = average / sweight;
    average.convertTo(dst, CV_8U);
}

//基于Mat指针实现
Mat NonLocalMeansFilter2(const Mat& src, int searchWindowSize, int templateWindowSize, double sigma, double h)
{
    Mat dst, pad;
    dst = Mat::zeros(src.rows, src.cols, CV_8UC1);

    //构建边界
    int padSize = (searchWindowSize + templateWindowSize) / 2;
    copyMakeBorder(src, pad, padSize, padSize, padSize, padSize, cv::BORDER_CONSTANT);

    int tN = templateWindowSize * templateWindowSize;
    int sN = searchWindowSize * searchWindowSize;
    int tR = templateWindowSize / 2;
    int sR = searchWindowSize / 2;

    vector<double> gaussian(256 * 256, 0);
    for (int i = 0; i < 256 * 256; i++)
    {
        double g = exp(-max(i - 2.0 * sigma * sigma, 0.0)) / (h * h);
        gaussian[i] = g;
        if (g < 0.001)
            break;
    }

    double* pGaussian = &gaussian[0];

    const int searchWindowStep = (int)pad.step - searchWindowSize;
    const int templateWindowStep = (int)pad.step - templateWindowSize;

    for (int i = 0; i < src.rows; i++)
    {
        uchar* pDst = dst.ptr(i);
        for (int j = 0; j < src.cols; j++)
        {
            cout << i << " " << j << endl;
            int* pVariance = new int[sN];
            double* pWeight = new double[sN];
            int cnt = sN - 1;
            double weightSum = 0;

            uchar* pCenter = pad.data + pad.step * (sR + i) + (sR + j);//搜索区域中心指针
            uchar* pUpLeft = pad.data + pad.step * i + j;//搜索区域左上角指针
            for (int m = searchWindowSize; m > 0; m--)
            {
                uchar* pDownLeft = pUpLeft + pad.step * m;

                for (int n = searchWindowSize; n > 0; n--)
                {
                    uchar* pC = pCenter;
                    uchar* pD = pDownLeft + n;

                    int w = 0;
                    for (int k = templateWindowSize; k > 0; k--)
                    {
                        for (int l = templateWindowSize; l > 0; l--)
                        {
                            w += (*pC - *pD) * (*pC - *pD);
                            pC++;
                            pD++;
                        }
                        pC += templateWindowStep;
                        pD += templateWindowStep;
                    }
                    w = (int)(w / tN);
                    pVariance[cnt--] = w;
                    weightSum += pGaussian[w];
                }
            }

            for (int m = 0; m < sN; m++)
            {
                pWeight[m] = pGaussian[pVariance[m]] / weightSum;
            }

            double tmp = 0.0;
            uchar* pOrigin = pad.data + pad.step * (tR + i) + (tR + j);
            for (int m = searchWindowSize, cnt = 0; m > 0; m--)
            {
                for (int n = searchWindowSize; n > 0; n--)
                {
                    tmp += *(pOrigin++) * pWeight[cnt++];
                }
                pOrigin += searchWindowStep;
            }
            *(pDst++) = (uchar)tmp;

            delete pWeight;
            delete pVariance;
        }
    }
    return dst;
}
