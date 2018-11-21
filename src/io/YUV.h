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

#ifndef YUV_H
#define YUV_H

#include <cstdio>
#include <fstream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class YUV
{
private:
  std::vector<cv::Mat> y;
  std::vector<cv::Mat> u;
  std::vector<cv::Mat> v;

protected:

public:
  YUV();
  ~YUV();

  void read(const char* filename, int width, int height, int nFrames);

  void write(
    const char* filename, const std::vector<cv::Mat>& y, const std::vector<cv::Mat>& u, const std::vector<cv::Mat>& v);

  std::vector<cv::Mat> getY();
  std::vector<cv::Mat> getU();
  std::vector<cv::Mat> getV();
};

#endif
