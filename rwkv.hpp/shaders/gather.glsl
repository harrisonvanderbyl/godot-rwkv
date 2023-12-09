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

layout(set = 0, binding = 1) readonly buffer indices_buffer {
    int w[];
};

layout(set = 0, binding = 2) buffer y_buffer {
    float y[];
};


void main() {
    uint bbt = gl_GlobalInvocationID.x;


    for (uint outx = 0; outx < OUT; outx++) {
        
        y[bbt * OUT + outx] = xy[w[bbt] * OUT + outx];
        
    }

}