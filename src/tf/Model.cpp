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

#include "Model.h"
using namespace std;
using namespace cv;
using namespace tensorflow;

Model::Model()
{
  session = 0;
}

Model::~Model()
{
  clearSession();
}

void Model::clearSession()
{
  if (session != 0)
  {
    delete session;
    session = 0;
  }
}

void Model::load(const string& pathToGraph, const string checkpointPath)
{
  clearSession();
  TF_CHECK_OK(NewSession(SessionOptions(), &session));
  TF_CHECK_OK(ReadBinaryProto(Env::Default(), pathToGraph, &graphDef));
  TF_CHECK_OK(session->Create(graphDef.graph_def()));

  const string restoreOpName = graphDef.saver_def().restore_op_name();
  const string filenameTensorName = graphDef.saver_def().filename_tensor_name();

  Tensor checkpointPathTensor(DT_STRING, TensorShape());
  checkpointPathTensor.scalar<string>()() = checkpointPath;

  tensor_dict feed_dict = {{filenameTensorName, checkpointPathTensor}};
  TF_CHECK_OK(session->Run(feed_dict, {}, {restoreOpName}, nullptr));
}

void Model::apply(const vector<Mat>& frame)
{
  const int patchSize = 128;
  const int channels = 1;
  const int outputSize = 4;
  Tensor input(DT_FLOAT, TensorShape({1, patchSize, patchSize, channels}));
  auto input_mapped = input.tensor<float, outputSize>();

  int width = patchSize * (frame[0].cols / patchSize);
  int height = patchSize * (frame[0].rows / patchSize);
  int dataSize = (width * height) / (patchSize * patchSize);

  for (int i = 0; i < frame.size(); i++)
  {
    vector<double> singleOutput(outputSize, 0);
    time_t begin;
    time(&begin);

    for (int y = 0; y < height; y += patchSize)
    {
      for (int x = 0; x < width; x += patchSize)
      {
        for (int dy = 0; dy < patchSize; dy++)
        {
          for (int dx = 0; dx < patchSize; dx++)
          {
            input_mapped(0, dy, dx, 0) = frame[i].at<uchar>(y + dy, x + dx) / 255.0;
          }
        }
        tensor_dict feed_dict = {{"input_batch", input}};
        TF_CHECK_OK(session->Run(feed_dict, {"BiasAdd_12"}, {}, &outputs));
        auto output_mapped = outputs[0].tensor<float, 2>();

        for (int j = 0; j < outputSize; j++)
        {
          singleOutput[j] += output_mapped(j);
        }
      }
    }

    time_t end;
    time(&end);
    double seconds = difftime(end, begin);

    for (int j = 0; j < outputSize; j++)
    {
      singleOutput[j] /= dataSize;
      cout << singleOutput[j] << " ";
    }
    cout << seconds << endl;
  }
}
