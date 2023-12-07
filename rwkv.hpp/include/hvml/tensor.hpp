#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "intrinsics.hpp"
#include <vector>
#include <iostream>
#include <algorithm>
#include <cassert>

#include "nlohmann/json.hpp"

#define ALIGNMENT 64

#define bfloat16 short

enum TENSORTYPE
{
    /// Boolean type
    kBOOL,
    /// Unsigned byte
    kUINT_8,
    /// Signed byte
    kINT_8,
    /// Signed integer (16-bit)
    kINT_16,
    /// Unsigned integer (16-bit)
    kUINT_16,
    /// Half-precision floating point
    kFLOAT_16,
    /// Brain floating point
    kBFLOAT_16,
    /// Signed integer (32-bit)
    kINT_32,
    /// Unsigned integer (32-bit)
    kUINT_32,
    /// Floating point (32-bit)
    kFLOAT_32,
    /// Floating point (64-bit)
    kFLOAT_64,
    /// Signed integer (64-bit)
    kINT_64,
    /// Unsigned integer (64-bit)
    kUINT_64,

};

NLOHMANN_JSON_SERIALIZE_ENUM(TENSORTYPE, {
                                             {kBOOL, "BOOL"},
                                             {kUINT_8, "U8"},
                                             {kINT_8, "I8"},
                                             {kINT_16, "I16"},
                                             {kUINT_16, "U16"},
                                             {kFLOAT_16, "F16"},
                                             {kBFLOAT_16, "BF16"},
                                             {kINT_32, "I32"},
                                             {kUINT_32, "U32"},
                                             {kFLOAT_32, "F32"},
                                             {kFLOAT_64, "F64"},
                                             {kINT_64, "I64"},
                                             {kUINT_64, "U64"},
                                         })

// can be float, double, int, etc.
template <typename DataType>
class Tensor
{
public:
    DataType *data __attribute__((aligned(ALIGNMENT)));
    uint64_t data_size_in_elements;
    uint64_t data_size_in_bytes;

    std::vector<uint64_t> shape = {};

    Tensor()
    {
        this->data = nullptr;
        this->data_size_in_elements = 0;
        this->data_size_in_bytes = 0;
    }

    uint64_t get_data_size_in_elements(std::vector<uint64_t> shape)
    {
        uint64_t size = 1;
        for (size_t i = 0; i < shape.size(); i++)
        {
            size *= shape[i];
        }
        this->data_size_in_elements = size;
        return size;
    }

    uint64_t get_data_size_in_bytes()
    {
        return this->data_size_in_elements * sizeof(DataType);
    }

    void fill(DataType value)
    {
        auto simd_value = SET1(value);
#pragma omp parallel for schedule(static, 256)
        for (uint64_t i = 0; i < this->data_size_in_elements; i += SIMD_WIDTH)
        {
            STORE(this->data + i, simd_value);
        }
    }

    void clone(Tensor<DataType> tensor)
    {
        if (this->data_size_in_bytes != tensor.data_size_in_bytes)
        {
            // throw std::runtime_error("Tensor sizes do not match during clone");
            std::cout << "Tensor sizes do not match during clone" << std::endl;

        }
        this->shape.clear();
        for (size_t i = 0; i < tensor.shape.size(); i++)
        {
            this->shape.push_back(tensor.shape[i]);
        }
        this->data_size_in_elements = tensor.data_size_in_elements;
        this->data_size_in_bytes = tensor.data_size_in_bytes;
        for (uint64_t i = 0; i < this->data_size_in_elements; i += SIMD_WIDTH)
        {
            STORE(this->data + i, LOAD(tensor.data + i));
        }
    }

    Tensor(std::vector<uint64_t> shape)
    {

        // copy the shape
        this->shape.clear();
        for (size_t i = 0; i < shape.size(); i++)
        {
            this->shape.push_back(shape[i]);
        }
        this->data_size_in_elements = get_data_size_in_elements(this->shape);
        this->data_size_in_bytes = get_data_size_in_bytes();
        // make sure alignment is correct to 64 bytes
        // malloc
        this->data = (DataType *)aligned_alloc(ALIGNMENT, this->data_size_in_bytes);
    }

    Tensor(std::vector<uint64_t> shape, DataType value)
    {
        // call the other constructor

        this->shape.clear();
        for (size_t i = 0; i < shape.size(); i++)
        {
            // std::cout << "shape: " << shape[i] << std::endl;
            if (shape[i] > 59605)
            {
                // throw std::runtime_error("Tensor size is too large");
                std::cout << "Tensor size is too large" << std::endl;
            }
            this->shape.push_back(shape[i]);
        }
        this->data_size_in_elements = get_data_size_in_elements(this->shape);
        this->data_size_in_bytes = get_data_size_in_bytes();
        // make sure alignment is correct to 64 bytes
        // malloc
        this->data = (DataType *)aligned_alloc(ALIGNMENT, this->data_size_in_bytes);
        this->fill(value);
    }

    Tensor(std::vector<uint64_t> shape, DataType *data)
    {
        this->shape.clear();
        for (size_t i = 0; i < shape.size(); i++)
        {
            this->shape.push_back(shape[i]);
        }
        this->data_size_in_elements = get_data_size_in_elements(this->shape);
        this->data_size_in_bytes = get_data_size_in_bytes();
        this->data = data;
    }

    // copy constructor
    Tensor(const Tensor<DataType> &tensor)
    {
        this->shape.clear();
        for (size_t i = 0; i < tensor.shape.size(); i++)
        {
            this->shape.push_back(tensor.shape[i]);
        }
        this->data_size_in_elements = tensor.data_size_in_elements;
        this->data_size_in_bytes = tensor.data_size_in_bytes;
        this->data = tensor.data;
    }

    void add(Tensor<DataType> &tensor, Tensor<DataType> &result)
    {
#pragma omp parallel for
        for (uint64_t i = 0; i < this->data_size_in_elements; i += SIMD_WIDTH)
        {
            STORE(result.data + i, ADD(LOAD(this->data + i), LOAD(tensor.data + i)));
        }
    }

    void add(Tensor<DataType> &tensor)
    {
#pragma omp parallel for
        for (uint64_t i = 0; i < this->data_size_in_elements; i += SIMD_WIDTH)
        {
            STORE(this->data + i, ADD(LOAD(this->data + i), LOAD(tensor.data + i)));
        }
    }

    void multiply(Tensor<DataType> &tensor, Tensor<DataType> &result)
    {
#pragma omp parallel for
        for (uint64_t i = 0; i < this->data_size_in_elements; i += SIMD_WIDTH)
        {
            STORE(result.data + i, MULTIPLY(LOAD(this->data + i), LOAD(tensor.data + i)));
        }
    }

    void multiply(float input, Tensor<DataType> &result)
    {
#pragma omp parallel for
        for (uint64_t i = 0; i < this->data_size_in_elements; i += SIMD_WIDTH)
        {
            STORE(result.data + i, MULTIPLY(LOAD(this->data + i), SET1(input)));
        }
    }

    void multiply(Tensor<DataType> &tensor)
    {
#pragma omp parallel for
        for (uint64_t i = 0; i < this->data_size_in_elements; i += SIMD_WIDTH)
        {
            STORE(this->data + i, MULTIPLY(LOAD(this->data + i), LOAD(tensor.data + i)));
        }
    }

    void multadd(Tensor<DataType> &tensor1, Tensor<DataType> &tensor2, Tensor<DataType> &result)
    {
#pragma omp parallel for
        for (uint64_t i = 0; i < this->data_size_in_elements; i += SIMD_WIDTH)
        {
            STORE(result.data + i, MULTADD(LOAD(this->data + i), LOAD(tensor1.data + i), LOAD(tensor2.data + i)));
        }
    }

    void multadd(Tensor<DataType> &tensor1, Tensor<DataType> &tensor2)
    {
#pragma omp parallel for
        for (uint64_t i = 0; i < this->data_size_in_elements; i += SIMD_WIDTH)
        {
            STORE(this->data + i, MULTADD(LOAD(this->data + i), LOAD(tensor1.data + i), LOAD(tensor2.data + i)));
        }
    }

    void reshape(std::vector<uint64_t> new_shape)
    {

        this->shape.clear();
        ulong newsize = 1;
        for (size_t i = 0; i < new_shape.size(); i++)
        {
            this->shape.push_back(new_shape[i]);
            newsize *= new_shape[i];
        }
        assert(newsize == this->data_size_in_elements);
        this->data_size_in_elements = newsize;
        this->data_size_in_bytes = get_data_size_in_bytes();
    }

    void unsafereshape(std::vector<uint64_t> new_shape)
    {

        this->shape.clear();
        ulong newsize = 1;
        for (size_t i = 0; i < new_shape.size(); i++)
        {
            this->shape.push_back(new_shape[i]);
            newsize *= new_shape[i];
        }
        this->data_size_in_elements = newsize;
        this->data_size_in_bytes = get_data_size_in_bytes();
    }

    void matmul(Tensor<float> &in, Tensor<float> &result, bool thisisbf16 = false)
    {
        // Pointers to the data

        // std::cout << "Is this bf16? " << thisisbf16 << std::endl;
        // A is float* or bf16*,

        // A is either bf16* or float*, check if it is bf16*

        auto B = in.data;
        float *C = result.data;

        // Dimensions
        const long IN = in.shape[2];
        const long OUT = result.shape[2];

        long BBT = 1;

        if (in.shape.size() > 1)
        {
            for (int i = in.shape.size() - 2; i >= 0; i--)
            {
                BBT *= in.shape[i];
            }
        }

        // confirm result length
        // std::cout << "result.data_size_in_elements: " << result.data_size_in_elements << std::endl;
        // std::cout << "BBT: " << BBT << std::endl;
        assert(result.data_size_in_elements == BBT * OUT);

        if (thisisbf16)
        {
            bfloat16 *A = (bfloat16 *)this->data;
// Parallel computation
#pragma omp parallel for collapse(2) schedule(dynamic, 256) shared(A, B, C)
            for (long i = 0; i < OUT; i += 1)
            {

                for (long bbj = 0; bbj < BBT; bbj += 1)
                {

                    auto acc = SET1(0.0f);
#pragma unroll(16)
                    for (long k = 0; k < IN; k += 32) // let intrinsics handle the unrolling
                    {

                        acc = DOTBF16(
                            LOADBF16(&A[i * IN + k]),
                            LOADFP32BF16(B + bbj * IN + k),
                            acc);
                    }
                    C[bbj * OUT + i] = REDUCE(acc);
                }
            }
        }
        else
        {
            float *A = (float *)this->data;
// Parallel computation for float tensors
#pragma omp parallel for collapse(2) schedule(dynamic, 256) shared(A, B, C)
            for (long i = 0; i < OUT; i += 1)
            {

                for (long bbj = 0; bbj < BBT; bbj += 1)
                {

                    auto acc = SET1(0.0f);
#pragma unroll(16)
                    for (long k = 0; k < IN; k += 32) // let intrinsics handle the unrolling
                    {

                        acc = DOTBF16F32(
                            LOADFP32BF16(A + i * IN + k),
                            LOADFP32BF16(B + bbj * IN + k),
                            acc);
                    }
                    C[bbj * OUT + i] = REDUCE(acc);
                }
            }
        }
    }
    void matmul(Tensor<float> &Art, Tensor<float> &Aot,
                Tensor<float> &Bt, Tensor<float> &Ct)
    {
        // Pointers to the data
        u_char *A = (u_char *)this->data;
        auto Ar = Art.data;
        auto Ao = Aot.data;
        auto B = Bt.data;
        auto C = Ct.data;

        long BB = Bt.shape[0];
        long T = Bt.shape[1];
        long IN = Bt.shape[2];
        long OUT = Ct.shape[2];

// Parallel computation
#pragma omp parallel for collapse(2) schedule(dynamic, UINT8THREADALLOC) shared(A, Ar, Ao, B, C)

        for (long bbj = 0; bbj < BB * T; bbj += 1)
        {
            for (long i = 0; i < OUT; i += 8)
            {

                // __m128 testacc = _mm128_setzero_ps();
                auto acc = UINT8ACC;
                auto acc2 = UINT8ACC;
                auto acc3 = UINT8ACC;
                auto acc4 = UINT8ACC;
                auto acc5 = UINT8ACC;
                auto acc6 = UINT8ACC;
                auto acc7 = UINT8ACC;
                auto acc8 = UINT8ACC;
                auto scale = PREPROCESSFLOATPARAMSUINT8(Ar + i);
                auto offset = PREPROCESSFLOATPARAMSUINT8(Ao + i);

#pragma unroll(16)
                for (long k = 0; k < IN; k += UINT8SIMDWIDTH)
                {
                    u_int8_t *aink = A + i * IN + k;
                    auto bbjonk = PREPROCESSFLOATINPUTUINT8(B + bbj * IN + k);

                    acc = UINT8MULTADD(offset, scale, (aink),
                                       bbjonk, acc, 0);
                    acc2 = UINT8MULTADD(offset, scale, (aink + IN),
                                        bbjonk, acc2, 1);
                    acc3 = UINT8MULTADD(offset, scale, (aink + IN * 2),
                                        bbjonk, acc3, 2);
                    acc4 = UINT8MULTADD(offset, scale, (aink + IN * 3),
                                        bbjonk, acc4, 3);
                    acc5 = UINT8MULTADD(offset, scale, (aink + IN * 4),
                                        bbjonk, acc5, 4);
                    acc6 = UINT8MULTADD(offset, scale, (aink + IN * 5),
                                        bbjonk, acc6, 5);
                    acc7 = UINT8MULTADD(offset, scale, (aink + IN * 6),
                                        bbjonk, acc7, 6);
                    acc8 = UINT8MULTADD(offset, scale, (aink + IN * 7),
                                        bbjonk, acc8, 7);
                }

                *(C + bbj * OUT + i + 7) = UINT8POSTREDUCE(acc8);
                *(C + bbj * OUT + i + 6) = UINT8POSTREDUCE(acc7);
                *(C + bbj * OUT + i + 5) = UINT8POSTREDUCE(acc6);
                *(C + bbj * OUT + i + 4) = UINT8POSTREDUCE(acc5);
                *(C + bbj * OUT + i + 3) = UINT8POSTREDUCE(acc4);
                *(C + bbj * OUT + i + 2) = UINT8POSTREDUCE(acc3);
                *(C + bbj * OUT + i + 1) = UINT8POSTREDUCE(acc2);
                *(C + bbj * OUT + i + 0) = UINT8POSTREDUCE(acc);
            }
        }
    }
    DataType sum()
    {
        DataType sum = 0;
        for (int i = 0; i < this->data_size_in_elements; i++)
        {
            sum += this->data[i];
        }
        return sum;
    }

    DataType expsum()
    {
        DataType sum = 0;
        for (int i = 0; i < this->data_size_in_elements; i++)
        {
            sum += exp(this->data[i]);
        }
        return sum;
    }

    void softmax()
    {
        // Pointers to the data
        auto A = this->data;

        // Dimensions
        auto dim = this->data_size_in_elements;

        // Parallel computation

        auto sum = this->expsum();

        auto sumhold = SET1(sum);

        for (long i = 0; i < dim; i += SIMD_WIDTH)
        {
            STORE(A + i, DIVIDE(EXP(LOAD(A + i)), sumhold));
        }
    }

    void gather(std::vector<std::vector<ulong>> indicies, Tensor<DataType> &buffer)
    {
        // Assume copy

        // Pointers to the data
        auto A = this->data;

        // Dimensions
        auto BATCH = indicies.size();
        auto T = indicies[0].size();
        auto OUT = this->shape[1];

        // print initial shape
        // std::cout << "buffer.size: " << buffer.data_size_in_elements << std::endl;

        buffer.shape.clear();
        buffer.shape.push_back(BATCH);
        buffer.shape.push_back(T);
        buffer.shape.push_back(OUT);
        buffer.data_size_in_elements = get_data_size_in_elements(buffer.shape);
        buffer.data_size_in_bytes = buffer.get_data_size_in_bytes();

        // std::cout << "BATCH: " << BATCH << std::endl;
        // std::cout << "T: " << T << std::endl;
        // std::cout << "OUT: " << OUT << std::endl;
        // std::cout << "out.size: " << BATCH*T*OUT << std::endl;
        // std::cout << "buffer.data_size_in_elements: " << buffer.data_size_in_elements << std::endl;
        // std::cout << "buffer.data_size_in_bytes: " << buffer.data_size_in_bytes << std::endl;

        // Parallel computation
        // #pragma omp parallel for collapse(2) schedule(dynamic, 256) shared(A, B)
        for (u_int64_t i = 0; i < BATCH; i += 1)
        {

            for (u_int64_t j = 0; j < T; j += 1)
            {

                for (u_int64_t k = 0; k < OUT; k += SIMD_WIDTH)
                {
                    auto acc = LOAD(A + indicies[i][j] * OUT + k);
                    STORE(&buffer.data[i * T * OUT + j * OUT + k], acc);
                }
            }
        }
    }

    void layernorm(const Tensor<DataType> &weight, const Tensor<DataType> &bias, const Tensor<DataType> &result)
    {
        uint64_t BTT = 1;

        if (this->shape.size() > 1)
        {
            for (int i = this->shape.size() - 2; i >= 0; i--)
            {
                BTT *= this->shape[i];
            }
        }

        uint64_t OUT = this->shape[this->shape.size() - 1];

        // confirm result length
        // std::cout << "result.data_size_in_elements: " << result.data_size_in_elements << std::endl;
        // std::cout << "this->data_size_in_elements: " << this->data_size_in_elements << std::endl;
        // std::cout << "OUT:" << OUT << std::endl;
        // std::cout << "BTT:" << BTT << std::endl;
        assert(result.data_size_in_elements == this->data_size_in_elements);

        // Pointers to the data
        auto A = this->data;
        auto W = weight.data;
        auto B = bias.data;
        auto C = result.data;

        for (uint64_t i = 0; i < BTT; i += 1)
        {
            float mean = 0.0f;
            float var = 0.0f;
            for (uint64_t j = 0; j < OUT; j += SIMD_WIDTH)
            {
                mean += REDUCE(LOAD(A + i * OUT + j));
            }
            mean /= OUT;

            for (uint64_t j = 0; j < OUT; j += SIMD_WIDTH)
            {
                auto acc = ADD(LOAD(A + i * OUT + j), SET1(-1.0f * mean));
                acc = MULTIPLY(acc, acc);
                var += REDUCE(acc);
            }
            var /= OUT;

            for (uint64_t j = 0; j < OUT; j += SIMD_WIDTH)
            {
                // std::cout << "level1: " <<j<< std::endl;
                auto acc = ADD(LOAD(A + i * OUT + j), SET1(-1.0f * mean));
                // std::cout << "level1mm: " <<j<< std::endl;
                acc = MULTIPLY(acc, SET1(1.0f / sqrt(var + 1e-5)));
                // std::cout << "level1acc: " <<j<< std::endl;

                STORE(C + i * OUT + j, MULTADD(LOAD(W + j), acc, LOAD(B + j)));
                // std::cout << "level1store: " <<j<< std::endl;
            }
        }
    }

    void lerp(Tensor<DataType> &tensor1, Tensor<DataType> &tensor2, Tensor<DataType> &result)
    {
        // Pointers to the data, assume (this) is the weights, lerp across last dimension
        auto A = tensor1.data;
        auto B = tensor2.data;
        auto C = result.data;

        // Dimensions

        auto IN = this->shape[0];

#pragma omp parallel for schedule(static, 256)
        for (uint64_t i = 0; i < result.data_size_in_elements; i += SIMD_WIDTH)
        {
            STORE(C + i, MULTADD(LOAD(B + i), LOAD(this->data + (i % IN)), MULTIPLY(LOAD(A + i), SET1(1.0f) - LOAD(this->data + (i % IN)))));
        }

        // Parallel computation
    }

    void relu(Tensor<DataType> &result)
    {
        // Pointers to the data
        auto A = this->data;
        auto C = result.data;

        // Dimensions
        result.unsafereshape(this->shape);

#pragma omp parallel for schedule(static, 256)
        for (uint64_t i = 0; i < this->data_size_in_elements; i += SIMD_WIDTH)
        {
            STORE(C + i, MAX(LOAD(A + i), SET1(0.0f)));
        }

        // Parallel computation
    }

    void sigmoid(Tensor<DataType> &result)
    {
        // Pointers to the data
        auto A = this->data;
        auto C = result.data;

        // Dimensions
        result.unsafereshape(this->shape);

        // std::cout << "result.sigmoid.data_size_in_elements: " << result.data_size_in_elements << std::endl;

#pragma omp parallel for schedule(static, 256)
        for (uint64_t i = 0; i < this->data_size_in_elements; i += SIMD_WIDTH)
        {
            STORE(C + i, DIVIDE(SET1(1.0f), ADD(SET1(1.0f), EXP(MULTIPLY(SET1(-1.0f), LOAD(A + i))))));
        }
        // Parallel computation
    }

    void wkv5(Tensor<float> &r, Tensor<float> &k, Tensor<float> &v, Tensor<float> &w, Tensor<float> &u, Tensor<float> &y)
    {

        auto rr = r.data;
        auto kk = k.data;
        auto vv = v.data;
        auto ww = w.data;
        auto uu = u.data;
        auto ss = this->data;
        auto out = y.data;

        int64_t B = r.shape[0];
        int64_t T = r.shape[1];
        int64_t C = r.shape[2];
        int64_t H = this->shape[1];

        // 1d
        int64_t bsize = H * T * (C / H);
        // 2d
        // int64_t bbsize = H * T * (C / H) * (C / H);

        // 1d tensor
        int64_t tsize = H * (C / H);
        // 2d tensor
        int64_t ttsize = H * (C / H) * (C / H);

        // 1d
        int64_t hsize = (C / H);
        // 2d
        int64_t hhsize = (C / H) * (C / H);

#pragma omp parallel for collapse(2) schedule(guided, 64) shared(kk, vv, ww, uu, rr, ss, out)
        for (int64_t bb = 0; bb < B; bb++)
        {
            for (int64_t hh = 0; hh < H; hh++)
            {
                for (int64_t t = 0; t < T; t++)
                {
                    for (int64_t i = 0; i < C / H; i++)
                    {

                        auto btimeoffset = bb * bsize;
                        auto timeoffset = btimeoffset + t * tsize;
                        auto bbhsize = bb * ttsize;

                        auto hoffset = hh * hsize;
                        auto bhofseti = timeoffset + hoffset;
                        auto bbhhsize = bbhsize + hh * hhsize;

                        int64_t iind = bhofseti + i;
                        auto hoffseti = hoffset + i;
                        auto bbhhofseti = bbhhsize + i * hsize;

                        // auto kkk = kk[iind];
                        auto kkk = SET1(kk[iind]);
                        auto uuu = SET1(uu[hoffseti]);
                        auto rrr = SET1(rr[iind]);
                        auto www = SET1(ww[hoffseti]);

                        for (int64_t j = 0; j < C / H; j += SIMD_WIDTH)
                        {
                            int64_t jind = bhofseti + j;
                            int64_t sind = bbhhofseti + j;

                            // atu = k[t,bb,hh,i]*v[t,bb,hh,j]
                            auto vvv = LOAD(&vv[jind]);

                            // multiply kkk and vvv
                            auto atu = MULTIPLY(vvv, kkk);

                            auto sss = LOAD(&ss[sind]);

                            // out[t,bb,hh,j] += r[t,bb,hh,i]*(s[bb,hh,i,j] + atu*u[hh,i] )
                            auto sssatuuuu = MULTADD(atu, uuu, sss);

                            auto outtt = LOAD(&out[jind]);

                            STORE(&out[jind], MULTADD(sssatuuuu, rrr, outtt));

                            // s[bb,hh,i,j] = s[bb,hh,i,j] * w[hh,i] + atu
                            STORE(&ss[sind], MULTADD(sss, www, atu));
                        }
                    }
                }
            }
        }
    }

    // [] operator
    Tensor<DataType> operator[](const uint64_t index)
    {
        uint64_t stride = 1;
        std::vector<uint64_t> new_shape = {};
        for (size_t i = 1; i < this->shape.size(); i++)
        {
            stride *= this->shape[i];
            new_shape.push_back(this->shape[i]);
        }

        return Tensor<DataType>(new_shape, this->data + index * stride);
    }

    // << operator for printing
    friend std::ostream &operator<<(std::ostream &os, const Tensor<DataType> &tensor)
    {
        if (tensor.shape.size() == 0)
        {
            os << tensor.data[0];
            return os;
        }

        os << "Tensor(";
        for (int i = 0; i < std::min(tensor.data_size_in_elements, 8UL); i++)
        {
            os << tensor.data[i];
            if (i != tensor.data_size_in_elements - 1)
            {
                os << ", ";
            }
        }
        if (tensor.data_size_in_elements > 8)
        {
            os << ", ...";
        }
        os << ", shape=(";
        for (int i = 0; i < tensor.shape.size(); i++)
        {
            os << tensor.shape[i];
            if (i != tensor.shape.size() - 1)
            {
                os << ", ";
            }
        }
        os << "))";
        return os;
    }

    // here we make it so, that if shape is {} then it is treated as a scalar by the compiler
    // this is useful for the [] operator
    operator DataType()
    {
        if (this->shape.size() != 0)
        {
            throw std::runtime_error("Tensor is not a scalar");
        }
        return this->data[0];
    }

    
};

#endif
