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

layout(set = 0, binding = 0) readonly buffer r_buffer {
    float r[];
};

layout(set = 0, binding = 1) readonly buffer k_buffer {
    float k[];
};


layout(set = 0, binding = 2) readonly buffer v_buffer {
    float v[];
};

layout(set = 0, binding = 3) readonly buffer w_buffer {
    float w[];
};

layout(set = 0, binding = 5) readonly buffer u_buffer {
    float u[];
};

layout(set = 0, binding = 6) buffer s_buffer {
    float s[];
};

layout(set = 0, binding = 7) buffer o_buffer {
    float o[];
};

void main() {
     uint bsize = H * T * (C / H);
      
    // 1d tensor
    uint tsize = H * (C / H);
    // 2d tensor
    uint ttsize = H * (C / H) * (C / H);

    // 1d
    uint hsize = (C / H);
    // 2d
    uint hhsize = (C / H) * (C / H);

    uint bb = gl_GlobalInvocationID.x;
    uint hh = gl_GlobalInvocationID.y;

    for (uint t = 0; t < T; t++)
                {
                    for (uint i = 0; i < C / H; i++)
                    {

                        uint btimeoffset = bb * bsize;
                        uint timeoffset = btimeoffset + t * tsize;
                        uint bbhsize = bb * ttsize;

                        uint hoffset = hh * hsize;
                        uint bhofseti = timeoffset + hoffset;
                        uint bbhhsize = bbhsize + hh * hhsize;

                        uint iind = bhofseti + i;
                        uint hoffseti = hoffset + i;
                        uint bbhhofseti = bbhhsize + i * hsize;

                        // auto kkk = kk[iind];
                        float kkk = (k[iind]);
                        float uuu = (u[hoffseti]);
                        float rrr = (r[iind]);
                        float www = (w[hoffseti]);

                        for (uint j = 0; j < C / H; j += 1)
                        {
                            uint jind = bhofseti + j;
                            uint sind = bbhhofseti + j;

                            // atu = k[t,bb,hh,i]*v[t,bb,hh,j]
                            float vvv = v[jind];

                            // multiply kkk and vvv
                            float atu = (vvv * kkk);

                            float sss = s[sind];

                            // out[t,bb,hh,j] += r[t,bb,hh,i]*(s[bb,hh,i,j] + atu*u[hh,i] )
                            float sssatuuuu = (atu * uuu) + sss;

                            float outtt = o[jind];

                            o[jind] = (sssatuuuu * rrr + outtt);

                            // s[bb,hh,i,j] = s[bb,hh,i,j] * w[hh,i] + atu
                            s[sind] = (sss * www) + atu;
                        }
                    }
                }
    
}