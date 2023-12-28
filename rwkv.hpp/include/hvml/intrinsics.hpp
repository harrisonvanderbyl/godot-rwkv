// simd

#ifndef INTRINSICS_HPP
#define INTRINSICS_HPP
// get pow
#include <cmath>
#define UINT8THREADALLOC 64

#define ALIGNMENT 32
// windows
#if defined(_WIN32) || defined(_WIN64)
#define ALIGNDECL __declspec(align(ALIGNMENT))
#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
// include windows.h to get _aligned_malloc
#include <windows.h>
#include <malloc.h>
#include <intrin.h>
// no default int types
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef long long ulong;
// fix missing M_E
#ifndef M_E
#define M_E 2.71828182845904523536f
#endif

#else
#define ALIGNDECL __attribute__((aligned(ALIGNMENT)))
#endif

#ifndef ulong
#define ulong size_t
#endif

#if defined(__AVX512F__) && defined(HVMLUSEAVX512) // This macro is defined if AVX-512 is supported
#include <immintrin.h>

#define SIMD_WIDTH 16
#define LOAD(x) _mm512_load_ps(x)
#define STORE(x, y) _mm512_store_ps(x, y)
#define SET1(x) _mm512_set1_ps(x)
#define MULTIPLY(x, y) _mm512_mul_ps(x, y)
#define MULTADD(x, y, z) _mm512_fmadd_ps(x, y, z)
#define REDUCE(x) _mm512_reduce_add_ps(x)
#define ADD(x, y) _mm512_add_ps(x, y)
#define MAX(x, y) _mm512_max_ps(x, y)
#define SUBTRACT(x, y) _mm512_sub_ps(x, y)
#define SIMDTYPE __m512

// check if using intel compiler
#if defined(__INTEL_LLVM_COMPILER)
#pragma message("AVX-512 exp is supported")
#define EXP(x) _mm512_exp_ps(x)
#else
#pragma message("AVX-512 exp is not supported, use intel compiler to use avx512exp, doing fallback")
#define EXP(x) exp_ps_fill(x)
SIMDTYPE exp_ps_fill(SIMDTYPE x)
{
    SIMDTYPE result = SET1(0.0f);
    for (int i = 0; i < SIMD_WIDTH; i++)
    {
        result[i] = pow(M_E, x[i]);
    }
    return result;
}
#endif
// #endif

#define DIVIDE(x, y) _mm512_div_ps(x, y)

// check if bf16 is supported
#ifdef __AVX512BF16__
#pragma message("AVX-512-bf16 is supported")
#define LOADBF16(x) (__m512bh) _mm512_loadu_si512(x)
// load 2 continous fp32 values from memory and convert to bf16
#define LOADFP32BF16(x) (__m512bh) _mm512_cvtne2ps_pbh(LOAD(x), LOAD(x + 16))
// do dot product of 2 bf16 vectors
#define DOTBF16(x, y, acc) _mm512_dpbf16_ps(acc, x, y)
#define DOTBF16F32(x, y, acc) _mm512_dpbf16_ps(acc, x, y)

#else
#pragma message("AVX-512-bf16 is not supported, doing in place conversion")
#define LOADBF16(x) x
#define LOADFP32BF16(x) x

// convert bf16 to fp32 by going uint16 -> int32(uint16, zeros) -> cast to float
#define bf16_to_fp32(x) (__m512) _mm512_slli_epi32(_mm512_cvtepi16_epi32(*(__m256i *)(x)), 16)

#define DOTBF16(x, y, acc) (_mm512_fmadd_ps(bf16_to_fp32(x + 16), LOAD(y), _mm512_fmadd_ps(bf16_to_fp32(x), LOAD(y + 16), acc)))

#define DOTBF16F32(x, y, acc) (_mm512_fmadd_ps(LOAD(x), LOAD(y), _mm512_fmadd_ps(LOAD(x + 16), LOAD(y + 16), acc)))

#endif

#define UINT8SIMDWIDTH SIMD_WIDTH
#define PREPROCESSFLOATINPUTUINT8(inp) LOAD(inp)
#define PREPROCESSFLOATPARAMSUINT8(inp) inp
#define UINT8ACC SET1(0.0f);
#define UINT8TOFLOAT32(x) _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(*(__m128i *)(x)))
#define UINT8POSTREDUCE(x) REDUCE(x)
#define UINT8MULTADD(offset, scale, u8, inp, acc, lane) (MULTADD(offset[lane] + scale[lane] * UINT8TOFLOAT32(u8), inp, acc))
// print out the SIMD width
#pragma message("AVX-512 is supported")
#else
// Fallback to AVX2 if AVX-512 is not supported
#ifdef __AVX2__
#include <immintrin.h>
#define SIMD_WIDTH 8
#define LOAD(x) _mm256_load_ps(x)
#define STORE(x, y) _mm256_store_ps((float*)x, y)
#define SET1(x) _mm256_set1_ps(x)
#define MULTIPLY(x, y) _mm256_mul_ps(x, y)
#define SUBTRACT(x, y) _mm256_sub_ps(x, y)
#define MULTADD(x, y, z) _mm256_fmadd_ps(x, y, z)
#ifdef _mm256_reduce_add_ps
#define REDUCE(x) _mm256_reduce_add_ps(x)
#else
#define REDUCE(x) reducefunc(x)
float reducefunc(__m256 x)
{
    float* y = (float*) &x;
    return y[0] + y[1] + y[2] + y[3] + y[4] + y[5] + y[6] + y[7];
}
#endif
#define ADD(x, y) _mm256_add_ps(x, y)
#define MAX(x, y) _mm256_max_ps(x, y)
#define DIVIDE(x, y) _mm256_div_ps(x, y)
#define SIMDTYPE __m256
#if defined(__INTEL_LLVM_COMPILER)
#pragma message("AVX-2 exp is supported")
#define EXP(x) _mm256_exp_ps(x)
#else
#define EXP(x) exp_ps_fill(x)
SIMDTYPE exp_ps_fill(SIMDTYPE y)
{
    float* x = (float*) &y;
    return _mm256_set_ps(pow(M_E, x[7]), pow(M_E, x[6]), pow(M_E, x[5]), pow(M_E, x[4]), pow(M_E, x[3]), pow(M_E, x[2]), pow(M_E, x[1]), pow(M_E, x[0]));
}
#endif
// print out the SIMD width
#pragma message("AVX-2 is supported")

#define LOADBF16(x) x
#define LOADFP32BF16(x) x
#define bf16_to_fp32(x) (__m256) _mm256_slli_epi32(_mm256_cvtepi16_epi32(*(__m128i *)(x)), 16)

// convert bf16 to fp32 by going uint16 -> int32(uint16, zeros) -> cast to float
#if defined(__AVX512BF16__)

#define LOADBF16(x) (__m256bh) _mm256_loadu_si256((__m256i *)x)

#define fp32_to_bf16(x) (__m256bh) _mm256_cvtne2ps_pbh(LOAD(x), LOAD(x + 8))

#define DOTBF16(x, y, acc) (_mm256_dpbf16_ps(acc, x, y))

#define DOTBF16F32(x, y, acc) (_mm256_dpbf16_ps(acc, LOADBF16(x), fp32_to_bf16(y)))

#else

#define DOTBF16(x, y, acc) (_mm256_fmadd_ps(bf16_to_fp32(x + 16), LOAD(y), _mm256_fmadd_ps(bf16_to_fp32(x + 24), LOAD(y + 8), _mm256_fmadd_ps(bf16_to_fp32(x), LOAD(y + 16), _mm256_fmadd_ps(bf16_to_fp32(x + 8), LOAD(y + 24), acc)))))

#define DOTBF16F32(x, y, acc) (_mm256_fmadd_ps(LOAD(x), LOAD(y), _mm256_fmadd_ps(LOAD(x + 8), LOAD(y + 8), _mm256_fmadd_ps(LOAD(x + 16), LOAD(y + 16), _mm256_fmadd_ps(LOAD(x + 24), LOAD(y + 24), acc)))))

#endif

#define UINT8SIMDWIDTH SIMD_WIDTH
#define UINT8TOFLOAT32(x) _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si64(x)))
#define PREPROCESSFLOATINPUTUINT8(inp) LOAD(inp)
#define PREPROCESSFLOATPARAMSUINT8(inp) inp
#define UINT8ACC SET1(0.0f);
#define UINT8POSTREDUCE(x) REDUCE(x)
#define UINT8MULTADD(offset, scale, u8, inp, acc, lane) MULTADD(offset[lane] + scale[lane] * UINT8TOFLOAT32(u8), inp, acc)

#else
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define UINT8THREADALLOC 1
#define SIMD_WIDTH 4 // NEON typically operates on 128-bit registers (4 floats)
#define LOAD(x) vld1q_f32(x)
#define STORE(x, y) vst1q_f32((float*)x, y)
#define SET1(x) vdupq_n_f32(x)
#define SET1Q(x) vdupq_n_f32(x)
#define MULTIPLY(x, y) vmulq_f32(x, y)
#define MULTADD(x, y, z) vmlaq_f32(z, x, y)
#define REDUCE(x) x[0] + x[1] + x[2] + x[3]
#define ADD(x, y) vaddq_f32(x, y)
#define SUBTRACT(x, y) vsubq_f32(x, y)
#define MAX(x, y) vmaxq_f32(x, y)
#define DIVIDE(x, y) vdivq_f32(x, y)
#define SIMDTYPE float32x4_t
#define EXP(x) exp_ps_fill(x)
SIMDTYPE exp_ps_fill(SIMDTYPE x)
{
    SIMDTYPE result = SET1(0.0f);
    for (int i = 0; i < SIMD_WIDTH; i++)
    {
        result[i] = pow(M_E, x[i]);
    }
    return result;
}



#define LOADBF16(x) x
#define LOADFP32BF16(x) x

#define bf16_to_fp32(x) vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vld1_u16((unsigned short *)x)), 16))

#define DOTBF16(x, y, acc) (MULTADD(bf16_to_fp32(x + 16), LOAD(y), MULTADD(bf16_to_fp32(x + 20), LOAD(y + 4), MULTADD(bf16_to_fp32(x + 24), LOAD(y + 8), MULTADD(bf16_to_fp32(x + 28), LOAD(y + 12), MULTADD(bf16_to_fp32(x), LOAD(y + 16), MULTADD(bf16_to_fp32(x + 4), LOAD(y + 20), MULTADD(bf16_to_fp32(x + 8), LOAD(y + 24), MULTADD(bf16_to_fp32(x + 12), LOAD(y + 28), acc)))))))))

#define DOTBF16F32(x, y, acc) (MULTADD(LOAD(x), LOAD(y), MULTADD(LOAD(x + 4), LOAD(y + 4), MULTADD(LOAD(x + 8), LOAD(y + 8), MULTADD(LOAD(x + 12), LOAD(y + 12), MULTADD(LOAD(x + 16), LOAD(y + 16), MULTADD(LOAD(x + 20), LOAD(y + 20), MULTADD(LOAD(x + 24), LOAD(y + 24), MULTADD(LOAD(x + 28), LOAD(y + 28), acc)))))))))

#ifndef aligned_alloc
#ifdef __GNUC__
#define ALIGNMENT 16

inline void *aligned_alloc(size_t alignment, size_t size)
{
    void *ptr = nullptr;
    int result = posix_memalign(&ptr, alignment, size);
    return (result == 0) ? ptr : nullptr;
}
#else
#error "aligned_alloc is not supported by this compiler"
#endif
#endif

#define UINT8SIMDWIDTH 8
// #define UINT8POSTREDUCE(x) (float)(x)
// #define UINT8MULTADD(offset, scale, u8, inp, acc, lane) uint8dotproduct(offset, scale, u8, inp, acc, lane)
#define REDUCEQ(x) x[0]+x[1]+x[2]+x[3]
#if defined(__ARM_FP16_FORMAT_IEEE) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#define SET1Q(x) vdupq_n_f16(x)
#define UINT8SIMDWIDTH 16
#define REDUCEQ(x) x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7]
// #define PREPROCESSFLOATINPUTUINT8(inp) vcombine_f16(vcvt_f16_f32(vld1q_f32(inp)), vcvt_f16_f32(vld1q_f32(inp + 4)))
// #define PREPROCESSFLOATPARAMSUINT8(inp) vcombine_f16(vcvt_f16_f32(vld1q_f32(inp)), vcvt_f16_f32(vld1q_f32(inp + 4)))
// #define UINT8ACC (float16_t)0.0f;

// float uint8dotproduct(float16x8_t offset_vec, float16x8_t scale_vec, uint8_t *u8, float16x8_t inp, float16_t acc, int lane)
// {

//     // Apply scale and offset to the float16 vectors
//     auto mulp = vmulq_f16(inp, vaddq_f16(vmulq_laneq_f16(vcvtq_f16_u16(vmovl_u8(vld1_u8(u8))), scale_vec, lane), vdupq_laneq_f16(offset_vec, lane)));

//     return acc + mulp[0] + mulp[1] + mulp[2] + mulp[3] + mulp[4] + mulp[5] + mulp[6] + mulp[7];
// }
#pragma message("ARM NEON FP16 is supported")
#else
#pragma message("ARM NEON FP16 is not supported")
// #define PREPROCESSFLOATINPUTUINT8(inp) inp
// #define PREPROCESSFLOATPARAMSUINT8(inp) inp
// #define UINT8ACC 0.0f;

// float uint8dotproduct(float *offset, float *scale, uint8_t *u8, float *inp, float acc, int lane)
// {
//     // Load the uint8_t data into a vector
//     uint8x8_t u8_vec = vld1_u8(u8);

//     // Convert uint8_t values to float32x4_t
//     uint16x8_t u16_vec = vmovl_u8(u8_vec);                       // convert uint8_t to uint16_t
//     uint32x4_t u32_low_vec = vmovl_u16(vget_low_u16(u16_vec));   // Extract lower part and convert to uint32_t
//     uint32x4_t u32_high_vec = vmovl_u16(vget_high_u16(u16_vec)); // Extract upper part and convert to uint32_t

//     // Apply scale and offset to the float vectors
//     float32x4_t offset_vec = vdupq_n_f32(offset[lane]); // Create a vector of the offset
//     float32x4_t scale_vec = vdupq_n_f32(scale[lane]);   // Create a vector of the scale

//     // Load the input float vector
//     // Perform the multiplication with inp vector
//     float32x4_t float_low_vec = vmulq_f32(vmlaq_f32(offset_vec, vcvtq_f32_u32(u32_low_vec), scale_vec), vld1q_f32(inp));
//     float32x4_t float_high_vec = vmulq_f32(vmlaq_f32(offset_vec, vcvtq_f32_u32(u32_high_vec), scale_vec), vld1q_f32(inp + 4));

//     // Pairwise add the lower and upper parts

//     // Add to the accumulator

//     return acc + float_low_vec[0] + float_low_vec[1] + float_low_vec[2] + float_low_vec[3] + float_high_vec[0] + float_high_vec[1] + float_high_vec[2] + float_high_vec[3];
// }
#endif

// Print out the SIMD width
#pragma message("ARM NEON is supported")
#else
#pragma message("No SIMD is supported")
#define SIMD_WIDTH 1
#define LOAD(x) *(x)
#define STORE(x, y) *((float*)x) = y
#define SET1(x) x
#define MULTIPLY(x, y) (x * y)
#define MULTADD(x, y, z) (x * y + z)
#define ADD(x, y) (x + y)
#define REDUCE(x) x
#define SUBTRACT(x, y) (x - y)
#define MAX(x, y) (x > y ? x : y)
#define EXP(x) exp(x)
#define DIVIDE(x, y) (x / y)
#define SIMDTYPE float
#define EXP(x) exp(x)
#endif

#endif
#endif

#ifdef DEBUG
#define DEBUG_MESSAGE(x) std::cout << x << std::endl;
#else
#define DEBUG_MESSAGE(x)
#endif

#endif