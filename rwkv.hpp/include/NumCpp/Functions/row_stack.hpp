/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2023 David Pilger
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy of this
/// software and associated documentation files(the "Software"), to deal in the Software
/// without restriction, including without limitation the rights to use, copy, modify,
/// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
/// permit persons to whom the Software is furnished to do so, subject to the following
/// conditions :
///
/// The above copyright notice and this permission notice shall be included in all copies
/// or substantial portions of the Software.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
/// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
/// PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
/// FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
/// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
/// DEALINGS IN THE SOFTWARE.
///
/// Description
/// Functions for working with NdArrays
///
#pragma once

#include <initializer_list>
#include <string>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //============================================================================
    // Method Description:
    /// Stack arrays in sequence vertically (row wise).
    ///
    /// @param inArrayList: {list} of arrays to stack
    ///
    /// @return NdArray
    ///
    template<typename dtype>
    NdArray<dtype> row_stack(const std::initializer_list<NdArray<dtype>>& inArrayList)
    {
        // first loop through to calculate the final size of the array
        Shape finalShape;
        for (auto& ndarray : inArrayList)
        {
            if (finalShape.isnull())
            {
                finalShape = ndarray.shape();
            }
            else if (ndarray.shape().cols != finalShape.cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("input arrays must have the same number of columns.");
            }
            else
            {
                finalShape.rows += ndarray.shape().rows;
            }
        }

        // now that we know the final size, contruct the output array
        NdArray<dtype> returnArray(finalShape);
        uint32         rowStart = 0;
        for (auto& ndarray : inArrayList)
        {
            const Shape theShape = ndarray.shape();
            for (uint32 row = 0; row < theShape.rows; ++row)
            {
                for (uint32 col = 0; col < theShape.cols; ++col)
                {
                    returnArray(rowStart + row, col) = ndarray(row, col);
                }
            }
            rowStart += theShape.rows;
        }

        return returnArray;
    }
} // namespace nc
