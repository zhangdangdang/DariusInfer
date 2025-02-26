//
// Created by fss on 23-5-27.
//

#include <gtest/gtest.h>
#include <glog/logging.h>
#include "data/tensor.hpp"

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging("darius");
  FLAGS_log_dir = "../log";
  FLAGS_alsologtostderr = true;

  LOG(INFO) << "Start test...\n";
  return RUN_ALL_TESTS();

//    std::vector<float> values(2 * 3 * 2);
//    // 将1到12填充到values中
//    for (int i = 0; i < 12; ++i) {
//        values.at(i) = float(i + 1);
//    }
//    arma::fcube cube(2, 3, 2);  // 2 行 x 3 列 x 2 切片
//
//    // 初始化第一个切片
//    cube.slice(0) = {{1, 2, 3}, {4, 5, 6}};
//
//    // 初始化第二个切片
//    cube.slice(1) = {{7, 8, 9}, {10, 11, 12}};
//    const float *cube_data = cube.memptr();
//    for(int i=0;i<cube.size();i++){
//        printf("%f ",cube_data[i]);
//    }
//    printf("\n end\n");
//    // 打印 cube
//    std::cout << "Cube:\n" << cube << std::endl;
//    for (uint32_t i = 0; i < cube.n_slices; ++i) {
//        auto &channel_data = cube.slice(i);
//        const arma::fmat &channel_data_t =
//                arma::fmat(values.data() + i * 2*3, cube.n_cols, cube.n_rows);
//        const float *cube_data = channel_data_t.memptr();
//        printf("\n mat\n");
//        for(int i=0;i<channel_data_t.size();i++){
//            printf("%f ",cube_data[i]);
//        }
//        printf("\n end\n");
//        printf("\n mat\n");
//        std::cout << "Channel " << i << ":\n" << channel_data_t << std::endl;
//        printf("\n end \n");
//        printf("\n mat t \n");
//        channel_data = channel_data_t.t();
//        const float *cube_data1 = channel_data.memptr();
//        printf("\n mat\n");
//        for(int i=0;i<channel_data_t.size();i++){
//            printf("%f ",cube_data1[i]);
//        }
//        std::cout << "Channel " << i << ":\n" << channel_data << std::endl;
//        printf("\n end t\n");
//
//    }
//    const float *cube_data1 = cube.memptr();
//    for(int i=0;i<cube.size();i++){
//        printf("%f ",cube_data1[i]);
//    }
//    printf("\n end\n");
//
//
//    return 0;
}