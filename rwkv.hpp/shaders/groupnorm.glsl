#version 450
#pragma shader_stage(compute)

#extension GL_EXT_shader_atomic_float : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
//

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1 ) in;

layout( constant_id = 0 ) const uint B = 1;
layout( constant_id = 1 ) const uint T = 1;
layout( constant_id = 2 ) const uint C = 1;
layout( constant_id = 3 ) const uint H = 1;

layout(set = 0, binding = 0) readonly buffer input_buffer {
    float inp[];
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
    uint bb = gl_GlobalInvocationID.x;
    uint tt = gl_GlobalInvocationID.y;
    uint hh = gl_GlobalInvocationID.z;

    float mean = 0.0;

    uint OUT = C / H;
    uint OFFSET = hh * OUT;
    uint bbt = bb * T + tt;


    for (uint outx = OFFSET; outx < OFFSET+OUT; outx++) {
        
        mean += inp[bbt * OUT + outx];
        
    }

    mean /= OUT;

    float var = 0.0;

    for (uint outx = OFFSET; outx < OFFSET+OUT; outx++) {
        
        var += (inp[bbt * OUT + outx] - mean) * (inp[bbt * OUT + outx] - mean);
        
    }

    var /= OUT;

    float std = sqrt(var + 1e-5);

    for (uint outx = OFFSET; outx < OFFSET+OUT; outx++) {
        
        y[bbt * OUT + outx] = ((inp[bbt * OUT + outx] - mean) / std) * w[outx] + b[outx];
        
    }

}