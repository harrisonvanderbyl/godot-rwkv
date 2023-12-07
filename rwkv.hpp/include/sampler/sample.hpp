#include "NumCpp.hpp"
int typical(float* _logits, float _temp = 0.9, float _tau = 0.8)
{
    int len = pow(2,16);
    // choose top token
    nc::NdArray<double> logits = nc::NdArray<double>(1,len);
    for (int i = 0; i < len; i++) {
        logits[i] = _logits[i];
    }

    nc::NdArray<double> probs = nc::special::softmax(logits); 
    logits = -nc::log(probs);
    nc::NdArray<double> ent = nc::nansum(logits * probs);
    nc::NdArray<double> shifted_logits = nc::abs(logits - ent);
    nc::NdArray<uint32_t> sorted_ids = nc::argsort(shifted_logits);
    nc::NdArray<double> sorted_logits = shifted_logits[sorted_ids];
    nc::NdArray<double> sorted_probs = probs[sorted_ids];
    nc::NdArray<double> cumulative_probs = nc::cumsum(sorted_probs);
    nc::NdArray<double> tau = nc::NdArray<double>(1,1);
    tau[0] = _tau;
    auto mask = (cumulative_probs < tau);
    // convert mask to int
    nc::NdArray<int> mask_int = nc::NdArray<int>(1,mask.size());
    for (uint64_t i = 0; i < mask.size(); i++) {
        mask_int[i] = mask[i];
    }

    // get cutoff
    auto cutoff = nc::sum(mask_int);
    // set probs to 0
    probs[shifted_logits > sorted_logits[cutoff]] = 0;
    if (_temp != 1.0) {
        probs = nc::power(probs, 1.0 / _temp);
    }

    // get random token
    auto out = nc::random::discrete<int>(nc::shape(tau),probs);
    return out[0];  
}

std::vector<int> typical(int batchsize, float* _logits, float _temp = 0.9, float _tau = 0.8){
    std::vector<int> out;
    for(int i = 0; i < batchsize; i++){
        out.push_back(typical(&_logits[i*int(pow(2,16))], _temp, _tau));
    }
    return out;
}