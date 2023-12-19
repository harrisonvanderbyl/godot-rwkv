#version 450
#pragma shader_stage(compute)

#extension GL_EXT_shader_atomic_float : require
#extension GL_EXT_shared_memory_block: require 
// #extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
// atomic add vec4
//

layout(local_size_x = 1, local_size_y = 8, local_size_z = 8) in;

layout( constant_id = 0 ) const uint BBT = 1;
layout( constant_id = 1 ) const uint IN = 1;
layout( constant_id = 2 ) const uint OUT = 1;
layout( constant_id = 3 ) const uint BLOCKSIZE = 1;

layout(set = 0, binding = 0) readonly buffer xy_buffer {
    readonly vec4 xy[];
};

layout(set = 0, binding = 1) readonly buffer w_buffer {
    readonly uint8_t w[];
};

layout(set = 0, binding = 2) readonly buffer w_range_buffer {
    readonly vec4 wr[];
};
layout(set = 0, binding = 3) readonly buffer wo_buffer {
    readonly vec4 wo[];
};

layout(set = 0, binding = 4) writeonly buffer y_buffer {
    writeonly vec4 y[];
};

const uint in4 = IN/4;
const uint out4 = OUT/4;

// shared int sharep;

void main() {
    uint bbt = gl_GlobalInvocationID.x;
    uint outx = gl_GlobalInvocationID.y;
    uint i = gl_GlobalInvocationID.z;

    uint lid = gl_LocalInvocationID.z;
    uint gid = gl_GlobalInvocationID.z;
    

    // if (lid == 0) {
    //     atomicAdd(sharep, 1);
    //     // sharep = 0;
    // }

    // barrier();

    

    const vec4 wrr = wr[outx];
    const vec4 woo = wo[outx];

    const vec4 mata1 = vec4((w[(outx*4+0) * IN  + i*4+0]), (w[(outx*4+0) * IN  + i*4+1]), (w[(outx*4+0) * IN  + i*4+2]), (w[(outx*4+0) * IN  + i*4+3]))*wrr.x + woo.x;
    const vec4 mata2 = vec4((w[(outx*4+1) * IN  + i*4+0]), (w[(outx*4+1) * IN  + i*4+1]), (w[(outx*4+1) * IN  + i*4+2]), (w[(outx*4+1) * IN  + i*4+3]))*wrr.y + woo.y;
    const vec4 mata3 = vec4((w[(outx*4+2) * IN  + i*4+0]), (w[(outx*4+2) * IN  + i*4+1]), (w[(outx*4+2) * IN  + i*4+2]), (w[(outx*4+2) * IN  + i*4+3]))*wrr.z + woo.z;
    const vec4 mata4 = vec4((w[(outx*4+3) * IN  + i*4+0]), (w[(outx*4+3) * IN  + i*4+1]), (w[(outx*4+3) * IN  + i*4+2]), (w[(outx*4+3) * IN  + i*4+3]))*wrr.w + woo.w;
    const mat4 matccc0 = mat4(mata1,mata2,mata3,mata4);
                                        
    const vec4 inp = xy[bbt * in4 + i];
    vec4 sum = inp*matccc0;  

    
    atomicAdd(y[bbt * out4 + outx].x, sum.x);
    atomicAdd(y[bbt * out4 + outx].y, sum.y);
    atomicAdd(y[bbt * out4 + outx].z, sum.z);
    atomicAdd(y[bbt * out4 + outx].w, sum.w);
 
    // barrier();

    // if (lid == 0) {
    //     y[bbt * out4 + outx] = shared_y[bbt * out4 + outx];
    // }


    
}