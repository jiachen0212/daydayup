
/*
#include <stdio.h>
#include <iostream>

int main(int argc, const char * argv[]) {
    std::cout << "Hello, World!\n";
    return 0;
}*/


#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
 
using namespace std;
using namespace cv;

Mat src, src_gray;

char* window_name = "Demo";

int main( int argc, char** argv )
{

    // 读取一副图片，不改变图片本身的颜色类型（该读取方式为DOS运行模式）
    // src = imread(argv[1], 1 );
    // std::cout << src.size << endl;

    src = imread(argv[1], 1 );
    // 将图片转换成灰度图片
    cvtColor(src, src_gray, CV_RGB2GRAY);
    namedWindow(window_name, CV_WINDOW_AUTOSIZE);

    imshow(window_name, src_gray);
    while(true) {
        int c;
        c = waitKey(0);
        // esc键 
        if (27 == (char) c) {
            break;
        }
    }
    
    return 0;


}
