#include <sys/time.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
// #include "paddle_inference_api.h"
#include "trt_helper.h"

// using paddle_infer::Config;
// using paddle_infer::Predictor;
// using paddle_infer::CreatePredictor;

static const int MAX_SEQ = 128;

// struct sample{
//     std::string qid;
//     std::string label;
//     std::vector<int> shape_info_0;
//     std::vector<int64_t> i0;
//     std::vector<int> shape_info_1;
//     std::vector<int64_t> i1;
//     std::vector<int> shape_info_2;
//     std::vector<int64_t> i2;
//     std::vector<int> shape_info_3;
//     std::vector<float> i3;
//     std::vector<int> shape_info_4;
//     std::vector<int64_t> i4;
//     std::vector<int> shape_info_5;
//     std::vector<int64_t> i5;
//     std::vector<int> shape_info_6;
//     std::vector<int64_t> i6;
//     std::vector<int> shape_info_7;
//     std::vector<int64_t> i7;
//     std::vector<int> shape_info_8;
//     std::vector<int64_t> i8;
//     std::vector<int> shape_info_9;
//     std::vector<int64_t> i9;
//     std::vector<int> shape_info_10;
//     std::vector<int64_t> i10;
//     std::vector<int> shape_info_11;
//     std::vector<int64_t> i11;
//     std::vector<float> out_data;
//     uint64_t timestamp;
// };

void split_string(const std::string& str,
                  const std::string& delimiter,
                  std::vector<std::string>& fields) {
    size_t pos = 0;
    size_t start = 0;
    size_t length = str.length();
    std::string token;
    while ((pos = str.find(delimiter, start)) != std::string::npos && start < length) {
        token = str.substr(start, pos - start);
        fields.push_back(token);
        start += delimiter.length() + token.length();
    }
    if (start <= length - 1) {
        token = str.substr(start);
        fields.push_back(token);
    }
}

void field2vec(const std::string& input_str,
               bool padding,
               std::vector<int>* shape_info,
               std::vector<int>* i64_vec,
               std::vector<float>* f_vec = nullptr) {
    std::vector<std::string> i_f;
    split_string(input_str, ":", i_f);
    std::vector<std::string> i_v;
    split_string(i_f[1], " ", i_v);
    std::vector<std::string> s_f;
    split_string(i_f[0], " ", s_f);
    for (auto& f : s_f) {
        shape_info->push_back(std::stoi(f));
    }
    int batch_size = shape_info->at(0);
    int seq_len = shape_info->at(1);
    if (i64_vec) {
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                i64_vec->push_back(std::stoll(i_v[i * seq_len + j]));
            }
            // padding to MAX_SEQ_LEN
            for (int j = 0; padding && j < MAX_SEQ - seq_len; ++j) {
                i64_vec->push_back(0);
            }
        }
    } else {
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                f_vec->push_back(std::stof(i_v[i * seq_len + j]));
            }
            // padding to MAX_SEQ_LEN
            for (int j = 0; padding && j < MAX_SEQ - seq_len; ++ j) {
                f_vec->push_back(0);
            }
        }
    }
    if (padding) {
        (*shape_info)[1] = MAX_SEQ;
    }
}

void line2sample(const std::string& line, sample* sout) {
    std::vector<std::string> fields;
    split_string(line, ";", fields);
    assert(fields.size() == 14);
    // parse qid
    std::vector<std::string> qid_f;
    split_string(fields[0], ":", qid_f);
    sout->qid = qid_f[1];
    // Parse label
    std::vector<std::string> label_f;
    split_string(fields[1], ":", label_f);
    sout->label = label_f[1];
    // Parse input field
    field2vec(fields[2], true, &(sout->shape_info_0), &(sout->i0));
    field2vec(fields[3], true, &(sout->shape_info_1), &(sout->i1));
    field2vec(fields[4], true, &(sout->shape_info_2), &(sout->i2));
    field2vec(fields[5], true, &(sout->shape_info_3), nullptr, &(sout->i3));
    field2vec(fields[6], false, &(sout->shape_info_4), &(sout->i4));
    field2vec(fields[7], false, &(sout->shape_info_5), &(sout->i5));
    field2vec(fields[8], false, &(sout->shape_info_6), &(sout->i6));
    field2vec(fields[9], false, &(sout->shape_info_7), &(sout->i7));
    field2vec(fields[10], false, &(sout->shape_info_8), &(sout->i8));
    field2vec(fields[11], false, &(sout->shape_info_9), &(sout->i9));
    field2vec(fields[12], false, &(sout->shape_info_10), &(sout->i10));
    field2vec(fields[13], false, &(sout->shape_info_11), &(sout->i11));

    sout->out_data.resize(sout->shape_info_11[0]);
    return;
}

int main(int argc, char *argv[]) {
  // init
  std::string model_para_file = argv[1];
  std::cout << model_para_file << std::endl;
  auto trt_helper = new TrtHepler(model_para_file, 0);
  // preprocess
  std::string aline;
  std::ifstream ifs;
  ifs.open(argv[2], std::ios::in);
  std::ofstream ofs;
  ofs.open(argv[3], std::ios::out);
  std::vector<sample> sample_vec;
  while (std::getline(ifs, aline)) {
      sample s;
      line2sample(aline, &s);
      sample_vec.push_back(s);
  }

  // inference
  for (auto& s : sample_vec) {
      // //run(predictor.get(), s);
      trt_helper->Forward(s);
  }

  // postprocess
  for (auto& s : sample_vec) {
      std::ostringstream oss;
      oss << s.qid << "\t";
      oss << s.label << "\t";
      for (int i = 0; i < s.out_data.size(); ++i) {
          oss << s.out_data[i];
          if (i == s.out_data.size() - 1) {
              oss << "\t";
          } else {
              oss << ",";
          }
      }
      oss << s.timestamp << "\n";
      ofs.write(oss.str().c_str(), oss.str().length());
  }
  ofs.close();
  ifs.close();
  return 0;
}
