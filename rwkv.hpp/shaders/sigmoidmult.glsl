#version 450
#pragma shader_stage(compute)

#extension GL_EXT_shader_atomic_float : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
//

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1 ) in;

layout( constant_id = 0 ) const uint ELEMENTS = 1;
layout( constant_id = 1 ) const uint BLOCKSIZE = 1;

layout(set = 0, binding = 0) readonly buffer xy_buffer {
    float xy[];
};

layout(set = 0, binding = 1) buffer x_buffer {
    float x[];
};

layout(set = 0, binding = 2) buffer y_buffer {
    float y[];
};


void main() {
    uint BLOCKID = gl_GlobalInvocationID.x;
    uint bbt = BLOCKID * BLOCKSIZE;
    uint bbt2 = bbt + BLOCKSIZE;

    for (uint i = bbt; i < bbt2; i++) {
        // sigmoid
        y[i] = x[i] / (1.0 + exp(-xy[i]));
    }

}