#version 450
#pragma shader_stage(compute)

#extension GL_EXT_shader_atomic_float : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
//

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1 ) in;

layout( constant_id = 0 ) const uint ELEMENTS = 1;

layout(set = 0, binding = 0) readonly buffer xy_buffer {
    mat4 xy[];
};

layout(set = 0, binding = 1) readonly buffer indices_buffer {
    mat4 w[];
};

layout(set = 0, binding = 2) writeonly buffer y_buffer {
    mat4 y[];
};


void main() {
    uint i = gl_GlobalInvocationID.x;
    

    y[i] = xy[i] + w[i];
    

}