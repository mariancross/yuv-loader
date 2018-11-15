#include <vector>
#include <opencv2/core/core.hpp>

#include "io/YUV.h"

int main(int argc, char* argv[])
{
    YUV yuv;
    yuv.read("../resources/BasketballPass_416x240_50.yuv", 416, 240, 50);
    yuv.write("../resources/output.yuv", yuv.getY(), yuv.getU(), yuv.getV());
    return 0;
}
