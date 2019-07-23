/*
 * Copyright (C) 2018 Maria Santamaria
 *
 * This file is part of YUVLoader.
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

#ifndef MODEL_H
#define MODEL_H

#include <ctime>
#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>

typedef std::vector<std::pair<std::string, tensorflow::Tensor>> tensor_dict;

class Model
{
private:
  tensorflow::Session* session;
  tensorflow::MetaGraphDef graphDef;
  std::vector<tensorflow::Tensor> outputs;

  void clearSession();

protected:

public:
  Model();
  ~Model();

  void load(const std::string& pathToGraph, const std::string checkpointPath);

  void apply(const std::vector<cv::Mat>& video);
};


#endif
