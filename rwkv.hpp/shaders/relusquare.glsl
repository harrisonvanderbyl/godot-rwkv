#version 450
#pragma shader_stage(compute)

#extension GL_EXT_shader_atomic_float : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
//

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1 ) in;

layout( constant_id = 0 ) const uint B = 1;
layout( constant_id = 1 ) const uint T = 1;
layout( constant_id = 2 ) const uint C = 1;

layout(set = 0, binding = 0) readonly buffer xy_buffer {
    float xy[];
};

layout(set = 0, binding = 1) buffer y_buffer {
    float y[];
};


void main() {
    uint b = gl_GlobalInvocationID.x;
    uint t = gl_GlobalInvocationID.y;
   
   
    for (uint c = 0; c < C*T*B; c++) {
        uint i = b * T*C + t * C + c;
        if (xy[i] < 0.0) {
            y[i] = 0.0;
        }
        else {
            y[i] = xy[i]*xy[i];
        }
    }

}