#include "../include/data/tensor.hpp"
#include <glog/logging.h>
#include <memory>
#include <numeric>

namespace darius_infer
{
    Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols)
    {
        data_ = arma::fcube(rows, cols, channels);
        if (channels == 1 && rows == 1)
        {
            this->raw_shapes_ = std::vector<u_int32_t>{cols};
        }
        else if (channels == 1)
        {
            this->raw_shapes_ = std::vector<u_int32_t>{rows, cols};
        }
        else
        {
            this->raw_shapes_ = std::vector<u_int32_t>{channels, rows, cols};
        }
    }

    Tensor<float>::Tensor(uint32_t size)
    {
        data_ = arma::fcube(1, size, 1);
        this->raw_shapes_ = std::vector<uint32_t>{size};
    }

    Tensor<float>::Tensor(uint32_t rows, uint32_t cols)
    {
        data_ = arma::fcube(rows, cols, 1);
        this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
    }

    // 使用shape创建tensor
    Tensor<float>::Tensor(const std::vector<uint32_t> &shapes)
    {
        CHECK(!shapes.empty() && shapes.size() <= 3);

        uint32_t remaining = 3 - shapes.size();
        std::vector<uint32_t> shapes_(3, 1);
        std::copy(shapes.begin(), shapes.end(), shapes_.begin() + remaining);

        uint32_t channels = shapes_.at(0);
        uint32_t rows = shapes_.at(1);
        uint32_t cols = shapes_.at(2);

        data_ = arma::fcube(rows, cols, channels);
        if (channels == 1 && rows == 1)
        {
            this->raw_shapes_ = std::vector<uint32_t>{cols};
        }
        else if (channels == 1)
        {
            this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
        }
        else
        {
            this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
        }
    }

    // 拷贝构造函数
    Tensor<float>::Tensor(const Tensor &tensor)
    {
        if (this != &tensor)
        {
            this->data_ = tensor.data_;
            this->raw_shapes_ = tensor.raw_shapes_;
        }
    }

    // 移动构造函数
    Tensor<float>::Tensor(Tensor<float> &&tensor) noexcept
    {
        if (this != &tensor)
        {
            this->data_ = std::move(tensor.data_);
            this->raw_shapes_ = tensor.raw_shapes_;
        }
    }

    // 赋值运算符重载
    Tensor<float> &Tensor<float>::operator=(Tensor<float> &&tensor) noexcept
    {
        if (this == &tensor)
        {
            return *this;
        }
        else
        {
            this->data_ = std::move(tensor.data_);
            this->raw_shapes_ = tensor.raw_shapes_;
            return *this;
        }
    }

    Tensor<float> &Tensor<float>::operator=(const Tensor<float> &tensor)
    {
        if (this == &tensor)
        {
            return *this;
        }
        else
        {
            this->data_ = tensor.data_;
            this->raw_shapes_ = tensor.raw_shapes_;
            return *this;
        }
    }

    uint32_t Tensor<float>::rows() const
    {
        CHECK(!this->data_.empty());
        return this->data_.n_rows;
    }

    uint32_t Tensor<float>::cols() const
    {
        CHECK(!this->data_.empty());
        return this->data_.n_cols;
    }

    uint32_t Tensor<float>::channels() const
    {
        CHECK(!this->data_.empty());
        return this->data_.n_slices;
    }

    uint32_t Tensor<float>::size() const
    {
        CHECK(!this->data_.empty());
        return this->data_.size();
    }

    void Tensor<float>::set_data(const arma::fcube &data)
    {
        CHECK(data.n_rows == this->data_.n_rows)
            << data.n_rows << " != " << this->data_.n_rows;
        CHECK(data.n_cols == this->data_.n_cols)
            << data.n_cols << " != " << this->data_.n_cols;
        CHECK(data.n_slices == this->data_.n_slices)
            << data.n_slices << " != " << this->data_.n_slices;
        this->data_ = data;
    }

    bool Tensor<float>::empty() const { return this->data_.empty(); }

    float Tensor<float>::index(uint32_t offset) const
    {
        CHECK(offset < this->data_.size()) << "Tensor index out of bound!";
        return this->data_.at(offset);
    }

    float &Tensor<float>::index(uint32_t offset)
    {
        CHECK(offset < this->data_.size()) << "Tensor index out of bound!";
        return this->data_.at(offset);
    }

    std::vector<uint32_t> Tensor<float>::shapes() const
    {
        CHECK(!this->data_.empty());
        return {this->channels(), this->rows(), this->cols()};
    }

    arma::fcube &Tensor<float>::data() { return this->data_; }

    const arma::fcube &Tensor<float>::data() const { return this->data_; }

    arma::fmat &Tensor<float>::slice(uint32_t channel)
    {
        CHECK_LT(channel, this->channels());
        return this->data_.slice(channel);
    }

    const arma::fmat &Tensor<float>::slice(uint32_t channel) const
    {
        CHECK_LT(channel, this->channels());
        return this->data_.slice(channel);
    }

    float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const
    {
        CHECK_LT(row, this->rows());
        CHECK_LT(col, this->cols());
        CHECK_LT(channel, this->channels());
        return this->data_.at(row, col, channel);
    }

    float &Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col)
    {
        CHECK_LT(row, this->rows());
        CHECK_LT(col, this->cols());
        CHECK_LT(channel, this->channels());
        return this->data_.at(row, col, channel);
    }

    void Tensor<float>::Padding(const std::vector<uint32_t> &pads,
                                float padding_value)
    {
        CHECK(!this->data_.empty());
        CHECK_EQ(pads.size(), 4);

        // 请补充代码    #include <vector>

        if (this->data_.empty())
        {
            throw std::invalid_argument("Tensor data is empty.");
        }

        // 检查 pads 的大小是否为 4
        if (pads.size() != 4)
        {
            throw std::invalid_argument("Pads vector must contain exactly 4 elements.");
        }

        // 四周填充的维度
        uint32_t pad_rows1 = pads.at(0); // up
        uint32_t pad_rows2 = pads.at(1); // bottom
        uint32_t pad_cols1 = pads.at(2); // left
        uint32_t pad_cols2 = pads.at(3); // right;

        // 获取原始张量的尺寸
        uint32_t original_rows = this->rows();
        uint32_t original_cols = this->cols();
        uint32_t channels = this->channels();

        // 计算新的尺寸
        uint32_t new_rows = original_rows + pad_rows1 + pad_rows2;
        uint32_t new_cols = original_cols + pad_cols1 + pad_cols2;

        // 创建一个新的数据容器
        // std::vector<float> new_data(new_rows * new_cols, padding_value);
        // 创建一个新的填充后的张量
        arma::fcube new_data(new_rows, new_cols, channels, arma::fill::value(padding_value));
        // 将原始数据复制到新张量的中心位置
        for (uint32_t c = 0; c < channels; ++c)
        {
            for (uint32_t i = 0; i < original_rows; ++i)
            {
                for (uint32_t j = 0; j < original_cols; ++j)
                {
                    new_data.at(i + pad_rows1, j + pad_cols1, c) = this->data_.at(i, j, c);
                }
            }
        }

        // 更新 data_ 和 raw_shapes_
        this->data_ = new_data;
        this->raw_shapes_ = {channels, new_rows, new_cols};
        // for (uint32_t i = 0; i < new_data.n_slices; ++i)
        // {
        //     LOG(INFO) << "1111111111111111Channel: " << i;
        //     LOG(INFO) << "\n"
        //               << new_data.slice(i);
        // }

    }

    void Tensor<float>::Fill(float value)
    {
        int a = this->data_.empty();
        CHECK(!this->data_.empty());
        this->data_.fill(value);
        int b = this->data_.empty();
    }

    void Tensor<float>::Fill(const std::vector<float> &values, bool row_major)
    { // fill的时候把外部传入行主序的值转成列主序的
        CHECK(!this->data_.empty());
        const uint32_t total_elems = this->data_.size();
        CHECK_EQ(values.size(), total_elems);
        if (row_major)
        {
            const uint32_t rows = this->rows();
            const uint32_t cols = this->cols();
            const uint32_t planes = rows * cols;
            const uint32_t channels = this->data_.n_slices;

            for (uint32_t i = 0; i < channels; ++i)
            {
                auto &channel_data = this->data_.slice(i);
                const arma::fmat &channel_data_t =
                    arma::fmat(values.data() + i * planes, this->cols(), this->rows());
                // arma::fmat(values.data() + i * planes, this->rows(), this->cols());
                // channel_data = channel_data_t; //.t();
                channel_data = channel_data_t.t();
            }
        }
        else
        {
            std::copy(values.begin(), values.end(), this->data_.memptr());
        }
    }

    void Tensor<float>::Show()
    {
        for (uint32_t i = 0; i < this->channels(); ++i)
        {
            LOG(INFO) << "Channel: " << i;
            LOG(INFO) << "\n"
                      << this->data_.slice(i);
        }
    }

    void Tensor<float>::Flatten(bool row_major)
    {
        CHECK(!this->data_.empty());
        const uint32_t rows = this->rows();
        const uint32_t cols = this->cols();
        const uint32_t channels = this->data_.n_slices;
        const uint32_t total_size = rows * cols * channels;
        std::vector<float> flattened_data(total_size); // std::vector<float> values;  //
        if (row_major)
        {
            flattened_data = this->values(true);
        }
        else
        {
            flattened_data = this->values(false);
        }
        this->raw_shapes_ = {total_size};
        // this->Show();
        // const float *cube_data = data_.memptr();
        // for (int i = 0; i < total_size; i++)
        // {
        //     printf("%0.0f ", cube_data[i]);
        // }
        // printf("\n end \n");
        this->data_.reshape(1, total_size, 1);
        // this->Show();
        if (row_major)
        {
            this->Fill(flattened_data, true);
        }
        else
        {
            this->Fill(flattened_data, false);
        }
    }

    void Tensor<float>::Rand()
    {
        CHECK(!this->data_.empty());
        this->data_.randn();
    }

    void Tensor<float>::Ones()
    {
        CHECK(!this->data_.empty());
        this->Fill(1.f);
    }

    void Tensor<float>::Transform(const std::function<float(float)> &filter)
    {
        CHECK(!this->data_.empty());
        this->data_.transform(filter);
    }

    const std::vector<uint32_t> &Tensor<float>::raw_shapes() const
    {
        CHECK(!this->raw_shapes_.empty());
        CHECK_LE(this->raw_shapes_.size(), 3);
        CHECK_GE(this->raw_shapes_.size(), 1);
        return this->raw_shapes_;
    }

    void Tensor<float>::Reshape(const std::vector<uint32_t> &shapes,
                                bool row_major)
    {
        CHECK(!this->data_.empty());
        CHECK(!shapes.empty());
        const uint32_t origin_size = this->size();
        const uint32_t current_size =
            std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies<uint32_t>());
        CHECK(shapes.size() <= 3);
        CHECK(current_size == origin_size);

        std::vector<float> values;
        if (row_major)
        { // cube reshapede 数据不会自动变成对应的顺序数据，需要手动提取出来，然后给再赋值进去。todo 有点转换太多了
            values = this->values(true);
        }
        if (shapes.size() == 3)
        {
            this->data_.reshape(shapes.at(1), shapes.at(2), shapes.at(0));
            this->raw_shapes_ = {shapes.at(0), shapes.at(1), shapes.at(2)}; // todo
        }
        else if (shapes.size() == 2)
        {
            this->data_.reshape(shapes.at(0), shapes.at(1), 1);
            this->raw_shapes_ = {shapes.at(0), shapes.at(1)};
        }
        else
        {
            this->data_.reshape(1, shapes.at(0), 1);
            this->raw_shapes_ = {shapes.at(0)};
        }

        if (row_major)
        {
            this->Fill(values, true);
        }
    }

    float *Tensor<float>::raw_ptr()
    {
        CHECK(!this->data_.empty());
        return this->data_.memptr();
    }

    float *Tensor<float>::raw_ptr(uint32_t offset)
    {
        const uint32_t size = this->size();
        CHECK(!this->data_.empty());
        CHECK_LT(offset, size);
        return this->data_.memptr() + offset;
    }

    std::vector<float> Tensor<float>::values(bool row_major)
    {
        CHECK_EQ(this->data_.empty(), false);
        std::vector<float> values(this->data_.size());

        if (!row_major)
        {
            std::copy(this->data_.mem, this->data_.mem + this->data_.size(),
                      values.begin());
        }
        else
        {
            uint32_t index = 0;
            for (uint32_t c = 0; c < this->data_.n_slices; ++c)
            {
                const arma::fmat &channel = this->data_.slice(c).t();
                std::copy(channel.begin(), channel.end(), values.begin() + index);
                index += channel.size();
            }
            CHECK_EQ(index, values.size());
        }
        return values;
    }

    float *Tensor<float>::matrix_raw_ptr(uint32_t index)
    {
        CHECK_LT(index, this->channels());
        uint32_t offset = index * this->rows() * this->cols();
        CHECK_LE(offset, this->size());
        float *mem_ptr = this->raw_ptr() + offset;
        return mem_ptr;
    }
} // namespace darius_infer