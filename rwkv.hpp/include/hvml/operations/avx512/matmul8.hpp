#ifndef HVMLAVX512MAT8_CPP
#define HVMLAVX512MAT8_CPP
#include "hvml/tensor.hpp"

// include threads
#include <thread>

// create 8 threads
// each thread does 1/8 of the work
// first, lets create a global variable to hold the threads
std::thread *t1;
std::thread *t2;
std::thread *t3;
std::thread *t4;
std::thread *t5;
std::thread *t6;
std::thread *t7;
std::thread *t8;

enum JOBTYPE {
	MATMUL,
	RWKV_ATT
};

// create a function that will be called by the thread
struct MatMulJob {
	const u_char *A = nullptr;
	const float *B;
	const float *C;
	const float *Ao;
	const float *Ar;
	const float *Bt;
	const float *Ct;
	const ulong bbt;
	const ulong ii;
	const ulong IN;
	const ulong OUT;
	JOBTYPE type = MATMUL;
    const float* ex = nullptr;
    const ulong H = 0;
    const ulong hh = 0;
};

// make compatible with compiler
std::atomic<ulong> jobs10 = 0;
std::atomic<ulong> jobs11 = 0;
std::atomic<ulong> jobs12 = 0;
std::atomic<ulong> jobs13 = 0;

std::atomic<ulong> jobs20 = 0;
std::atomic<ulong> jobs21 = 0;
std::atomic<ulong> jobs22 = 0;
std::atomic<ulong> jobs23 = 0;

std::atomic<ulong> jobs30 = 0;
std::atomic<ulong> jobs31 = 0;
std::atomic<ulong> jobs32 = 0;
std::atomic<ulong> jobs33 = 0;

std::atomic<ulong> jobs40 = 0;
std::atomic<ulong> jobs41 = 0;
std::atomic<ulong> jobs42 = 0;
std::atomic<ulong> jobs43 = 0;

void dopartial(MatMulJob *job) {
	// do the work
	auto A = job->A;
	auto B = job->B;
	auto C = job->C;
	auto IN = job->IN;
	auto OUT = job->OUT;
	auto Ao = job->Ao;
	auto Ar = job->Ar;
	auto Bt = job->Bt;
	auto Ct = job->Ct;
	auto bbt = job->bbt;
	auto ii = job->ii;

	const auto Ario = _mm512_load_ps(Ar + ii);
    const auto Aoioo = _mm512_div_ps( _mm512_load_ps(Ao + ii), Ario);
	__m512 zz = _mm512_setzero_ps();
	for (uint32_t i = ii; i < ii + 16; i += 1) {
		const float Aoio = Aoioo[i & 15];

		__m512 aa = _mm512_setzero_ps();
		const auto IAIN = A + i * IN;
		for (uint32_t k = 0; k < IN; k += 32) {
            
			const __m512 a00 = Aoio + _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(*((__m128i *)(IAIN + k))));
			const __m512 a01 = Aoio + _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(*((__m128i *)(IAIN + k + 16))));
		
			
			const __m512 b01 = _mm512_load_ps(B + bbt * IN + k + 16);
			const __m512 b00 = _mm512_load_ps(B + bbt * IN + k);

            

            
            // aa = _mm512_dpbf16_ps(aa, _mm512_cvtne2ps_pbh(a00, a01), _mm512_cvtne2ps_pbh(b00, b01)); 
            aa = _mm512_fmadd_ps(a00, b00, aa);
            aa = _mm512_fmadd_ps(a01, b01, aa);

		}
		zz[i & 15] = _mm512_reduce_add_ps( aa);
	}
	_mm512_store_ps(
			(void*)(C + bbt * OUT + ii),
			zz * Ario);
}

void dopartialwkv5att(MatMulJob *job) {
	auto T = job->bbt;
	auto B = job->ii;
	auto CH = job->IN;
	auto bb = job->OUT;
	auto kk = job->Ao;
    auto ww = job->C;
	auto vv = job->Ar;
	auto uu = job->Bt;
	auto rr = job->Ct;
	auto ss = job->ex;
	auto out = job->B;
    auto H = job->H;
    auto hh = job->hh;

	// 1d
	uint bsize = H * T * CH;

	// 1d tensor
	uint tsize = H * CH;
	// 2d tensor
	uint ttsize = H * CH * CH;

	// 1d
	uint hsize = CH;
	// 2d
	uint hhsize = CH * CH;

	for (uint t = 0; t < T; t++) {
		for (uint i = 0; i < CH; i++) {
			auto btimeoffset = bb * bsize;
			auto timeoffset = btimeoffset + t * tsize;
			auto bbhsize = bb * ttsize;

			auto hoffset = hh * hsize;
			auto bhofseti = timeoffset + hoffset;
			auto bbhhsize = bbhsize + hh * hhsize;

			uint iind = bhofseti + i;
			auto hoffseti = hoffset + i;
			auto bbhhofseti = bbhhsize + i * hsize;

			// auto kkk = kk[iind];
			auto kkk = SET1(kk[iind]);
			auto uuu = SET1(uu[hoffseti]);
			auto rrr = SET1(rr[iind]);
			auto www = SET1(ww[hoffseti]);

			for (uint j = 0; j < CH; j += SIMD_WIDTH) {
				uint jind = bhofseti + j;
				uint sind = bbhhofseti + j;

				// atu = k[t,bb,hh,i]*v[t,bb,hh,j]
				auto vvv = LOAD(&vv[jind]);

				// multiply kkk and vvv
				auto atu = MULTIPLY(vvv, kkk);

				auto sss = LOAD(&ss[sind]);

				// out[t,bb,hh,j] += r[t,bb,hh,i]*(s[bb,hh,i,j] + atu*u[hh,i] )
				auto sssatuuuu = MULTADD(atu, uuu, sss);

				auto outtt = LOAD(&out[jind]);

				STORE((void*)&out[jind], MULTADD(sssatuuuu, rrr, outtt));

				// s[bb,hh,i,j] = s[bb,hh,i,j] * w[hh,i] + atu
				STORE((void*)&ss[sind], MULTADD(sss, www, atu));
			}
		}
	}
}

void listen(std::atomic<ulong> *jobs1, std::atomic<ulong> *jobs2) {
	// wait for all jobs to be done
	while (true) {
		// check if all jobs are done

		// get last job
		if (jobs1[0] != 0) {
            if((*(MatMulJob **)jobs1)->type == JOBTYPE::RWKV_ATT){
                dopartialwkv5att(*(MatMulJob **)jobs1);
            }else{
			    dopartial(*(MatMulJob **)jobs1);
            }
			jobs1[0] = 0;
		}
		if (jobs2[0] != 0) {
			dopartial(*(MatMulJob **)jobs2);
			jobs2[0] = 0;
		}
	}
}

bool started = false;

void startWorkers() {
	// start the threads
	if (started) {
		return;
	}
	started = true;

	std::cout << "Starting workers" << std::endl;

	t1 = new std::thread(listen, &jobs10, &jobs11);
	t2 = new std::thread(listen, &jobs12, &jobs13);
	t3 = new std::thread(listen, &jobs20, &jobs21);
	t4 = new std::thread(listen, &jobs22, &jobs23);
	t5 = new std::thread(listen, &jobs30, &jobs31);
	t6 = new std::thread(listen, &jobs32, &jobs33);
	t7 = new std::thread(listen, &jobs40, &jobs41);
	t8 = new std::thread(listen, &jobs42, &jobs43);
	std::cout << "Started workers" << std::endl;
}

template <>
void Tensor<uint8_t, HVMLCPU>::matmul(Tensor<float, HVMLCPU> &Art, Tensor<float, HVMLCPU> &Aot,
		Tensor<float, HVMLCPU> &Bt, Tensor<float, HVMLCPU> &Ct) {
	// Pointers to the data
	const u_char *A = (u_char *)this->data;
	const auto Ar = Art.data;
	const auto Ao = Aot.data;
	const auto B = Bt.data;
	const auto C = Ct.data;

	const ulong BB = Bt.shape[0];
	const ulong T = Bt.shape[1];
	const ulong IN = Bt.shape[2];
	const ulong OUT = Ct.shape[2];

	startWorkers();

	for (uint32_t bbt = 0; bbt < BB * T; bbt += 1) {
		for (uint32_t ii = 0; ii < OUT; ii += (16 * 16)) {
			auto job1 = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, ii, IN, OUT };
			jobs10 = (uint64_t)&job1;
			auto job2 = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, ii + 16, IN, OUT };
			jobs11 = (uint64_t)&job2;
			auto job3 = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, ii + 32, IN, OUT };
			jobs12 = (uint64_t)&job3;
			auto job4 = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, ii + 48, IN, OUT };
			jobs13 = (uint64_t)&job4;

			auto job5 = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, ii + 64, IN, OUT };
			jobs20 = (uint64_t)&job5;
			auto job6 = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, ii + 80, IN, OUT };
			jobs21 = (uint64_t)&job6;
			auto job7 = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, ii + 96, IN, OUT };
			jobs22 = (uint64_t)&job7;
			auto job8 = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, ii + 112, IN, OUT };
			jobs23 = (uint64_t)&job8;

			auto job9 = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, ii + 128, IN, OUT };

			jobs30 = (uint64_t)&job9;
			auto job10 = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, ii + 144, IN, OUT };
			jobs31 = (uint64_t)&job10;
			auto job11 = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, ii + 160, IN, OUT };
			jobs32 = (uint64_t)&job11;
			auto job12 = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, ii + 176, IN, OUT };
			jobs33 = (uint64_t)&job12;

			auto job13 = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, ii + 192, IN, OUT };
			jobs40 = (uint64_t)&job13;
			auto job14 = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, ii + 208, IN, OUT };
			jobs41 = (uint64_t)&job14;
			auto job15 = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, ii + 224, IN, OUT };
			jobs42 = (uint64_t)&job15;
			auto job16 = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, ii + 240, IN, OUT };
			jobs43 = (uint64_t)&job16;

			// wait for all jobs to be done
			while (
					jobs10 != 0 || jobs11 != 0 || jobs12 != 0 || jobs13 != 0 ||
					jobs20 != 0 || jobs21 != 0 || jobs22 != 0 || jobs23 != 0 ||
					jobs30 != 0 || jobs31 != 0 || jobs32 != 0 || jobs33 != 0 ||
					jobs40 != 0 || jobs41 != 0 || jobs42 != 0 || jobs43 != 0) {
				// if so, sleep for 5ms
				// std::this_thread::sleep_for(std::chrono::microseconds(1));
				// continue;
			}
		}
		// float outo = _mm512_reduce_add_epi32(c);
	}
}

void matmul(
		Tensor<uint8_t, HVMLCPU> &AThis, Tensor<float, HVMLCPU> &Art, Tensor<float, HVMLCPU> &Aot, Tensor<float, HVMLCPU> &ABt, Tensor<float, HVMLCPU> &ACt,
		Tensor<uint8_t, HVMLCPU> &BThis, Tensor<float, HVMLCPU> &Brt, Tensor<float, HVMLCPU> &Bot, Tensor<float, HVMLCPU> &BBt, Tensor<float, HVMLCPU> &BCt,
		Tensor<uint8_t, HVMLCPU> &CThis, Tensor<float, HVMLCPU> &Crt, Tensor<float, HVMLCPU> &Cot, Tensor<float, HVMLCPU> &CBt, Tensor<float, HVMLCPU> &CCt,
		Tensor<uint8_t, HVMLCPU> &DThis, Tensor<float, HVMLCPU> &Drt, Tensor<float, HVMLCPU> &Dot, Tensor<float, HVMLCPU> &DBt, Tensor<float, HVMLCPU> &DCt) {
	// Pointers to the data
	const auto AT = AThis.data;
	const auto Ar = Art.data;
	const auto Ao = Aot.data;
	const auto AB = ABt.data;
	const auto AC = ACt.data;

	const auto BT = BThis.data;
	const auto Br = Brt.data;
	const auto Bo = Bot.data;
	const auto BB = BBt.data;
	const auto BC = BCt.data;

	const auto CT = CThis.data;
	const auto Cr = Crt.data;
	const auto Co = Cot.data;
	const auto CB = CBt.data;
	const auto CC = CCt.data;

	const auto DT = DThis.data;
	const auto Dr = Drt.data;
	const auto Do = Dot.data;
	const auto DB = DBt.data;
	const auto DC = DCt.data;

	const ulong BATCH = ABt.shape[0];
	const ulong TIME = ABt.shape[1];
	const ulong IN = ABt.shape[2];
	const ulong OUT = ACt.shape[2];

	startWorkers();

	// #pragma omp parallel for collapse(2)
	for (uint32_t BatchTime = 0; BatchTime < BATCH * TIME; BatchTime += 1) {
		for (uint32_t ii = 0; ii < OUT; ii += 16 * 4) {
			auto job1 = MatMulJob{ AT, AB, AC, Ao, Ar, ABt.data, ACt.data, BatchTime, ii, IN, OUT };
			jobs10 = (uint64_t)&job1;
			auto job2 = MatMulJob{ AT, AB, AC, Ao, Ar, ABt.data, ACt.data, BatchTime, ii + 16, IN, OUT };
			jobs11 = (uint64_t)&job2;
			auto job3 = MatMulJob{ AT, AB, AC, Ao, Ar, ABt.data, ACt.data, BatchTime, ii + 32, IN, OUT };
			jobs12 = (uint64_t)&job3;
			auto job4 = MatMulJob{ AT, AB, AC, Ao, Ar, ABt.data, ACt.data, BatchTime, ii + 48, IN, OUT };
			jobs13 = (uint64_t)&job4;

			auto job5 = MatMulJob{ BT, BB, BC, Bo, Br, BBt.data, BCt.data, BatchTime, ii, IN, OUT };
			jobs20 = (uint64_t)&job5;
			auto job6 = MatMulJob{ BT, BB, BC, Bo, Br, BBt.data, BCt.data, BatchTime, ii + 16, IN, OUT };
			jobs21 = (uint64_t)&job6;
			auto job7 = MatMulJob{ BT, BB, BC, Bo, Br, BBt.data, BCt.data, BatchTime, ii + 32, IN, OUT };
			jobs22 = (uint64_t)&job7;
			auto job8 = MatMulJob{ BT, BB, BC, Bo, Br, BBt.data, BCt.data, BatchTime, ii + 48, IN, OUT };
			jobs23 = (uint64_t)&job8;

			auto job9 = MatMulJob{ CT, CB, CC, Co, Cr, CBt.data, CCt.data, BatchTime, ii, IN, OUT };
			jobs30 = (uint64_t)&job9;
			auto job10 = MatMulJob{ CT, CB, CC, Co, Cr, CBt.data, CCt.data, BatchTime, ii + 16, IN, OUT };
			jobs31 = (uint64_t)&job10;
			auto job11 = MatMulJob{ CT, CB, CC, Co, Cr, CBt.data, CCt.data, BatchTime, ii + 32, IN, OUT };
			jobs32 = (uint64_t)&job11;
			auto job12 = MatMulJob{ CT, CB, CC, Co, Cr, CBt.data, CCt.data, BatchTime, ii + 48, IN, OUT };
			jobs33 = (uint64_t)&job12;

			auto job13 = MatMulJob{ DT, DB, DC, Do, Dr, DBt.data, DCt.data, BatchTime, ii, IN, OUT };
			jobs40 = (uint64_t)&job13;
			auto job14 = MatMulJob{ DT, DB, DC, Do, Dr, DBt.data, DCt.data, BatchTime, ii + 16, IN, OUT };
			jobs41 = (uint64_t)&job14;
			auto job15 = MatMulJob{ DT, DB, DC, Do, Dr, DBt.data, DCt.data, BatchTime, ii + 32, IN, OUT };
			jobs42 = (uint64_t)&job15;
			auto job16 = MatMulJob{ DT, DB, DC, Do, Dr, DBt.data, DCt.data, BatchTime, ii + 48, IN, OUT };
			jobs43 = (uint64_t)&job16;

			// wait for all jobs to be done
			while (
					jobs10 != 0 || jobs11 != 0 || jobs12 != 0 || jobs13 != 0 ||
					jobs20 != 0 || jobs21 != 0 || jobs22 != 0 || jobs23 != 0 ||
					jobs30 != 0 || jobs31 != 0 || jobs32 != 0 || jobs33 != 0 ||
					jobs40 != 0 || jobs41 != 0 || jobs42 != 0 || jobs43 != 0) {
				// if so, sleep for 5ms
				// std::this_thread::sleep_for(std::chrono::microseconds(1));
				// continue;
			}
		}
		// float outo = _mm512_reduce_add_epi32(c);
	}
}
// t

void matmul(
		Tensor<uint8_t, HVMLDYNAMIC> &AThis, Tensor<float, HVMLDYNAMIC> &Art, Tensor<float, HVMLDYNAMIC> &Aot, Tensor<float, HVMLDYNAMIC> &ABt, Tensor<float, HVMLDYNAMIC> &ACt,
		Tensor<uint8_t, HVMLDYNAMIC> &BThis, Tensor<float, HVMLDYNAMIC> &Brt, Tensor<float, HVMLDYNAMIC> &Bot, Tensor<float, HVMLDYNAMIC> &BBt, Tensor<float, HVMLDYNAMIC> &BCt,
		Tensor<uint8_t, HVMLDYNAMIC> &CThis, Tensor<float, HVMLDYNAMIC> &Crt, Tensor<float, HVMLDYNAMIC> &Cot, Tensor<float, HVMLDYNAMIC> &CBt, Tensor<float, HVMLDYNAMIC> &CCt,
		Tensor<uint8_t, HVMLDYNAMIC> &DThis, Tensor<float, HVMLDYNAMIC> &Drt, Tensor<float, HVMLDYNAMIC> &Dot, Tensor<float, HVMLDYNAMIC> &DBt, Tensor<float, HVMLDYNAMIC> &DCt) {
	// verify adding all devices = 0
	auto dcount = AThis.device.device_type.i + BThis.device.device_type.i + CThis.device.device_type.i + DThis.device.device_type.i +
			Art.device.device_type.i + Brt.device.device_type.i + Crt.device.device_type.i + Drt.device.device_type.i +
			Aot.device.device_type.i + Bot.device.device_type.i + Cot.device.device_type.i + Dot.device.device_type.i +
			ABt.device.device_type.i + BBt.device.device_type.i + CBt.device.device_type.i + DBt.device.device_type.i +
			ACt.device.device_type.i + BCt.device.device_type.i + CCt.device.device_type.i + DCt.device.device_type.i;
	if (dcount != 0) {
		std::cout << "ERROR: All tensors must be on the same device" << std::endl;
		exit(1);
	}
	if (AThis.device.device_type.i == KHVMLCPU.i) {
		matmul(*(Tensor<uint8_t, HVMLCPU> *)&AThis, *(Tensor<float, HVMLCPU> *)&Art, *(Tensor<float, HVMLCPU> *)&Aot, *(Tensor<float, HVMLCPU> *)&ABt, *(Tensor<float, HVMLCPU> *)&ACt,
				*(Tensor<uint8_t, HVMLCPU> *)&BThis, *(Tensor<float, HVMLCPU> *)&Brt, *(Tensor<float, HVMLCPU> *)&Bot, *(Tensor<float, HVMLCPU> *)&BBt, *(Tensor<float, HVMLCPU> *)&BCt,
				*(Tensor<uint8_t, HVMLCPU> *)&CThis, *(Tensor<float, HVMLCPU> *)&Crt, *(Tensor<float, HVMLCPU> *)&Cot, *(Tensor<float, HVMLCPU> *)&CBt, *(Tensor<float, HVMLCPU> *)&CCt,
				*(Tensor<uint8_t, HVMLCPU> *)&DThis, *(Tensor<float, HVMLCPU> *)&Drt, *(Tensor<float, HVMLCPU> *)&Dot, *(Tensor<float, HVMLCPU> *)&DBt, *(Tensor<float, HVMLCPU> *)&DCt);
	}
}

template <>
void Tensor<float, HVMLCPU>::wkv5(Tensor<float, HVMLCPU> &r, Tensor<float, HVMLCPU> &k, Tensor<float, HVMLCPU> &v, Tensor<float, HVMLCPU> &w, Tensor<float, HVMLCPU> &u, Tensor<float, HVMLCPU> &y) {
	auto rr = r.data;
	auto kk = k.data;
	auto vv = v.data;
	auto ww = w.data;
	auto uu = u.data;
	auto ss = this->data;
	auto out = y.data;

	uint B = r.shape[0];
	uint T = r.shape[1];
	uint C = r.shape[2];
	uint H = this->shape[1];

	// #pragma omp parallel for collapse(2) schedule(guided, 64) shared(kk, vv, ww, uu, rr, ss, out)
	for (uint bb = 0; bb < B; bb++) {
		// heads are divisable by 8 I think
		for (uint hh = 0; hh < H; hh+=8) {

    // const u_char *A = nullptr;
	// const float *B;
	// const float *C;
	// const float *Ao;
	// const float *Ar;
	// const float *Bt;
	// const float *Ct;
	// const ulong bbt;
	// const ulong ii;
	// const ulong IN;
	// const ulong OUT;
	// JOBTYPE type = MATMUL;
    // const float* ex = nullptr;
    // const ulong H = 0;
    // const ulong hh = 0;
    auto job1 = MatMulJob{ 
                        nullptr,
                        out,
                        ww,
                        kk,
                        vv,
                        uu,
                        rr,
                        T,
                        B,
                        C/H,
                        bb,
                        JOBTYPE::RWKV_ATT,
                        ss,
                        H,
                        hh
                        };

    jobs10 = (uint64_t)&job1;
    auto job2 = MatMulJob{ 
                        nullptr,
                        out,
                        ww,
                        kk,
                        vv,
                        uu,
                        rr,
                        T,
                        B,
                        C/H,
                        bb,
                        JOBTYPE::RWKV_ATT,
                        ss,
                        H,
                        hh+1
                        };

    jobs12 = (uint64_t)&job2;
    auto job3 = MatMulJob{ 
                        nullptr,
                        out,
                        ww,
                        kk,
                        vv,
                        uu,
                        rr,
                        T,
                        B,
                        C/H,
                        bb,
                        JOBTYPE::RWKV_ATT,
                        ss,
                        H,
                        hh+2
                        };

    jobs20 = (uint64_t)&job3;
    auto job4 = MatMulJob{ 
                        nullptr,
                        out,
                        ww,
                        kk,
                        vv,
                        uu,
                        rr,
                        T,
                        B,
                        C/H,
                        bb,
                        JOBTYPE::RWKV_ATT,
                        ss,
                        H,
                        hh+3
                        };

    jobs22 = (uint64_t)&job4;
    auto job5 = MatMulJob{ 
                        nullptr,
                        out,
                        ww,
                        kk,
                        vv,
                        uu,
                        rr,
                        T,
                        B,
                        C/H,
                        bb,
                        JOBTYPE::RWKV_ATT,
                        ss,
                        H,
                        hh+4
                        };

    jobs30 = (uint64_t)&job5;
    auto job6 = MatMulJob{ 
                        nullptr,
                        out,
                        ww,
                        kk,
                        vv,
                        uu,
                        rr,
                        T,
                        B,
                        C/H,
                        bb,
                        JOBTYPE::RWKV_ATT,
                        ss,
                        H,
                        hh+5
                        };

    jobs32 = (uint64_t)&job6;
    auto job7 = MatMulJob{ 
                        nullptr,
                        out,
                        ww,
                        kk,
                        vv,
                        uu,
                        rr,
                        T,
                        B,
                        C/H,
                        bb,
                        JOBTYPE::RWKV_ATT,
                        ss,
                        H,
                        hh+6
                        };

    jobs40 = (uint64_t)&job7;
    auto job8 = MatMulJob{ 
                        nullptr,
                        out,
                        ww,
                        kk,
                        vv,
                        uu,
                        rr,
                        T,
                        B,
                        C/H,
                        bb,
                        JOBTYPE::RWKV_ATT,
                        ss,
                        H,
                        hh+7
                        };

    jobs42 = (uint64_t)&job8;

    // wait for all jobs to be done
    while (
            jobs10 != 0 || jobs12 != 0 ||
            jobs20 != 0 || jobs22 != 0 ||
            jobs30 != 0 || jobs32 != 0 ||
            jobs40 != 0 || jobs42 != 0) {

        // if so, sleep for 5ms
        // std::this_thread::sleep_for(std::chrono::microseconds(1));
        // continue;
    }


    // auto T = job->bbt;
	// auto B = job->ii;
	// auto CH = job->IN;
	// auto bb = job->OUT;
	// auto kk = job->Ao;
    // auto ww = job->C;
	// auto vv = job->Ar;
	// auto uu = job->Bt;
	// auto rr = job->Ct;
	// auto ss = job->ex;
	// auto out = job->B;
    // auto H = job->H;=
    // auto hh = job->hh;
            
            


            
		}
	}
}

#endif // HVMLAVX512MAT8_CPP