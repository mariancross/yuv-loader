/*
 * Copyright (C) 2018 Maria Santamaria
 *
 * This file is part of yuv-loader.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "YUV.h"
using namespace std;
using namespace cv;

YUV::YUV()
{

}

YUV::~YUV()
{

}

void YUV::read(const char* filename, int width, int height, int nFrames)
{
    FILE *stream = fopen(filename, "rb");
    int lumaSize = width * height;
    int halfWidth = width / 2;
    int halfHeight = height / 2;
    int chromaSize = halfWidth * halfHeight;

    y = vector<Mat>(nFrames);
    u = vector<Mat>(nFrames);
    v = vector<Mat>(nFrames);

    auto *yRaw = (unsigned char *) malloc(sizeof(unsigned char) * lumaSize);
    auto *uRaw = (unsigned char *) malloc(sizeof(unsigned char) * chromaSize);
    auto *vRaw = (unsigned char *) malloc(sizeof(unsigned char) * chromaSize);

    for (int i = 0; i < nFrames; i++)
    {
        fread(yRaw, sizeof(unsigned char), lumaSize, stream);
        fread(uRaw, sizeof(unsigned char), chromaSize, stream);
        fread(vRaw, sizeof(unsigned char), chromaSize, stream);

        y[i] = Mat(height, width, CV_8UC1, yRaw).clone();
        u[i] = Mat(halfHeight, halfWidth, CV_8UC1, uRaw).clone();
        v[i] = Mat(halfHeight, halfWidth, CV_8UC1, vRaw).clone();
    }

    free(yRaw);
    free(uRaw);
    free(vRaw);

    fclose(stream);
}

void YUV::write(const char* filename, const vector<Mat>& y, const vector<Mat>& u, const vector<Mat>& v)
{
    FILE* stream = fopen(filename, "wb");
    auto nFrames = (int)y.size();
    int lumaSize = y[0].cols * y[0].rows;
    int chromaSize = u[0].cols * u[0].rows;

    for (int i = 0; i < nFrames; i++)
    {
        fwrite(y[i].data, sizeof(unsigned char), lumaSize, stream);
        fwrite(u[i].data, sizeof(unsigned char), chromaSize, stream);
        fwrite(v[i].data, sizeof(unsigned char), chromaSize, stream);
    }

    fclose(stream);
}

vector<Mat> YUV::getY()
{
    return y;
}

vector<Mat> YUV::getU()
{
    return u;
}

vector<Mat> YUV::getV()
{
    return v;
}
