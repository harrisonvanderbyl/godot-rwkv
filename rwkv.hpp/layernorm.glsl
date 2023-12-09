#version 450
#pragma shader_stage(compute)

#extension GL_EXT_shader_atomic_float : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
//

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1 ) in;

layout( constant_id = 0 ) const uint BBT = 1;
layout( constant_id = 1 ) const uint OUT = 1;
layout( constant_id = 2 ) const uint BLOCKSIZE = 1;

layout(set = 0, binding = 0) readonly buffer xy_buffer {
    float xy[];
};

layout(set = 0, binding = 1) readonly buffer w_buffer {
    float w[];
};

layout(set = 0, binding = 2) readonly buffer b_buffer {
    float b[];
};

layout(set = 0, binding = 3) buffer y_buffer {
    float y[];
};


void main() {
    uint bbt = gl_GlobalInvocationID.x;
    // uint outx0 = gl_GlobalInvocationID.y * BLOCKSIZE;
    // uint outx1 = outx0 + BLOCKSIZE;
    // uint inxglobal = gl_GlobalInvocationID.z;

    float mean = 0.0;

    for (uint outx = 0; outx < OUT; outx++) {
        
        mean += xy[bbt * OUT + outx];
        
    }

    mean /= OUT;

    float var = 0.0;

    for (uint outx = 0; outx < OUT; outx++) {
        
        var += (xy[bbt * OUT + outx] - mean) * (xy[bbt * OUT + outx] - mean);
        
    }

    var /= OUT;

    float std = sqrt(var);

    for (uint outx = 0; outx < OUT; outx++) {
        
        y[bbt * OUT + outx] = ((xy[bbt * OUT + outx] - mean) / std) * w[outx] + b[outx];
        
    }

}