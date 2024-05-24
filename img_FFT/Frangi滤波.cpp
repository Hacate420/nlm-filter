#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <vector>
#include "frangi.h"
using namespace std;
using namespace cv;


//int main(int argc, char* argv[]) {
//int main() {
//	//set default frangi opts
//	frangi2d_opts_t opts;
//	frangi2d_createopts(&opts);
//
//
//	string filename = "F:\\偏振成像\\测试数据\\数据采集\\组合\\前后弧形\\3.bmp";
//	//read image file, run frangi, output to output file
//	Mat input_img = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
//	Mat input_img_fl;
//	input_img.convertTo(input_img_fl, CV_32FC1);
//	Mat vesselness, scale, angles;
//	frangi2d(input_img_fl, vesselness, scale, angles, opts);
//	//imwrite(outFilename + ".png", vesselness * 255);
//
//	namedWindow("src", WINDOW_NORMAL);
//	namedWindow("dst", WINDOW_NORMAL);
//	imshow("src", input_img);
//	imshow("dst", vesselness * 255);
//	waitKey(0);
//}
