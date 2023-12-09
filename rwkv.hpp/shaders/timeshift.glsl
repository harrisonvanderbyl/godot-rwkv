#version 450
#pragma shader_stage(compute)

#extension GL_EXT_shader_atomic_float : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
//

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1 ) in;

layout( constant_id = 0 ) const uint BATCH = 1;
layout( constant_id = 1 ) const uint SEQ = 1;
layout( constant_id = 2 ) const uint OUT = 1;
layout( constant_id = 3 ) const uint BLOCKSIZE = 1;

layout(set = 0, binding = 0) buffer state_buffer {
    float state[];
};

layout(set = 0, binding = 1) readonly buffer input_buffer {
    float inp[];
};

layout(set = 0, binding = 2) buffer output_buffer {
    float o[];
};




void main() {
    uint b = gl_GlobalInvocationID.x;
    
    // this->buffer[i][0].clone(this->state[i][0]);
    for (uint i = 0; i < OUT; i++){
        o[b*OUT+i] = state[b*OUT+i];
    }

    for (uint j = 0; j < SEQ; j++){
        if (j > 0){
            // this->buffer[i][j].clone(input[i][j-1]);
            for (uint i = 0; i < OUT; i++){
                o[b*OUT*SEQ+j*OUT+i] = inp[b*OUT*SEQ+i+(j-1)*OUT];
            }
        }
        else{
            // this->state[i][0].clone(input[i][seq-1]);
            for (uint i = 0; i < OUT; i++){
                state[b*OUT+i] = inp[b*OUT*SEQ+i+(SEQ-1)*OUT];
            }
        }
    }

    // for (size_t j = 0; j < seq; j++){
    //     if (j > 0){
    //         this->buffer[i][j].clone(input[i][j-1]);
    //     }
    //     else{
    //         this->state[i][0].clone(input[i][seq-1]);
    //     }
    // }


}