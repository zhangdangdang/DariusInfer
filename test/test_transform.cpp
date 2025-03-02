
#include "data/tensor.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

float MinusOne(float value) { return value - 1.f; }

TEST(test_transform, transform1) {
    using namespace darius_infer;
    Tensor<float> f1(2, 3, 4);
    f1.Rand();
    LOG(INFO) << "-------------------before transform-------------------";
    f1.Show();
    f1.Transform(MinusOne);
    LOG(INFO) << "-------------------after transform-------------------";
    f1.Show();
}