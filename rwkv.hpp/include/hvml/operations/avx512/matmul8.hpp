#ifndef HVMLAVX512MAT8_CPP
#define HVMLAVX512MAT8_CPP
#include "hvml/tensor.hpp"
#include <atomic>
// include threads
#include <thread>


ulong ZeroLong = ulong(0);
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
	ulong bbt;
	ulong ii;
	const ulong INSHAPE;
	const ulong OUTSHAPE;
	JOBTYPE type = MATMUL;
	const float *ex = nullptr;
	const ulong H = 0;
	const ulong hh = 0;

	// if convert to ulong, then it will be its own address
	operator ulong() const {
		return (ulong)this;
	}

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

void dopartial(MatMulJob job) {
	// do the work
	auto A = job.A;
	auto B = job.B;
	auto C = job.C;
	auto INSHAPE = job.INSHAPE;
	auto OUTSHAPE = job.OUTSHAPE;
	auto Ao = job.Ao;
	auto Ar = job.Ar;
	auto Batch = job.bbt;
	auto ii = job.ii;
	#if defined(__AVX512F__) && defined(HVMLUSEAVX512)
		const auto Ario = _mm512_load_ps(Ar + ii);
		const auto Aoioo = _mm512_div_ps(_mm512_load_ps(Ao + ii), Ario);
		__m512 zz = _mm512_setzero_ps();
		for (uint32_t i = ii; i < ii + 16; i += 1) {
			const float Aoio = Aoioo[i & 15]/Ario[i&15];

			__m512 aa = _mm512_setzero_ps();
			const auto IAINSHAPE = A + i * INSHAPE;
			for (uint32_t k = 0; k < INSHAPE; k += 32) {
				const __m512 b01 = _mm512_load_ps(B + bbt * INSHAPE + k + 16);
				const __m512 b00 = _mm512_load_ps(B + bbt * INSHAPE + k);

				const __m512 a00 = Aoio + _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(*((__m128i *)(IAINSHAPE + k))));
				const __m512 a01 = Aoio + _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(*((__m128i *)(IAINSHAPE + k + 16))));

				// aa = _mm512_dpbf16_ps(aa, _mm512_cvtne2ps_pbh(a00, a01), _mm512_cvtne2ps_pbh(b00, b01));
				aa = _mm512_fmadd_ps(a00, b00, aa);
				aa = _mm512_fmadd_ps(a01, b01, aa);
			}
			zz[i & 15] = _mm512_reduce_add_ps(aa);
		}
		_mm512_store_ps(
				(void *)(C + bbt * OUTSHAPE + ii),
				zz * Ario);
	#elif defined(__AVX2__)
	for (ulong bbt = 0; bbt < Batch; bbt += 1) {
		for (ulong dii = ii; dii < OUTSHAPE; dii += 16*16){ 
		for (ulong b = 0; b < 16; b+= 8){
		const float* Ario1 = (Ar + dii + b);
		const float* Aoio1 = (Ao + dii + b);

		float zz1[8] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0};
		
		for (uint32_t i = dii+b; i < dii + b+8; i += 1) {
			auto Aoio = SET1(Aoio1[i&7]);
			auto Ario = SET1(Ario1[i&7]);

			const auto IAINSHAPE = A + i * INSHAPE;

			auto sum1 = SET1(0.0);
			auto sum2 = SET1(0.0);
			for (uint32_t k = 0; k < INSHAPE; k += 16) {
				// avx2
				auto w = _mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i *)(IAINSHAPE + k)));  // Load the input uint8_t vector
				// convert uint32_t to float32x8_t
				auto u = MULTADD(_mm256_cvtepi32_ps(w),Ario,Aoio);   // Convert uint32_t to float32_t
				// Load the input float vector
				// Perform the multiplication with inp vector
				sum1 = MULTADD(u, LOAD(B + bbt * INSHAPE + k),sum1);

				auto w1 = _mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i *)(IAINSHAPE + k + 8)));  // Load the input uint8_t vector

				auto u1 = MULTADD(_mm256_cvtepi32_ps(w1),Ario,Aoio);   // Convert uint32_t to float32_t

				sum2 = MULTADD(u1, LOAD(B + bbt * INSHAPE + k + 8),sum2);
				
			}

			sum1 = ADD(sum1,sum2);
			
			zz1[i&7]= REDUCE(sum1);
			

		}

		
			STORE(
					(void *)(C + bbt * OUTSHAPE + dii + b),
					LOAD(zz1) );
			}
		}
	}

	#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
	for (ulong bbt = 0; bbt < Batch; bbt += 1) {
	for (ulong dii = ii; dii < OUTSHAPE; dii += 16*16){ 
	 	for (ulong b = 0; b < 16; b+= 4){
		const auto Ario1 = LOAD(Ar + dii+b);
		const auto Aoio1 = DIVIDE(LOAD(Ao + dii + b),Ario1);

		auto zz1 = SET1(0.0);

		for (uint32_t i = dii+b; i < dii + b+4; i += 1) {
			auto Aoio = Aoio1[i&3];

			const auto IAINSHAPE = A + i * INSHAPE;

			auto sum1 = SET1(0.0);
			auto sum2 = SET1(0.0);
			
			for (uint32_t k = 0; k < INSHAPE; k += 8) {
				
				auto u16_vec = vmovl_u8(vld1_u8((IAINSHAPE + k)));  
				
					// Convert uint8_t values to float32x4_t
									// convert uint8_t to uint16_t
				auto u32_low_vec = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_vec)))+Aoio;   // Extract lower part and convert to uint32_t
				auto u32_high_vec = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_vec)))+Aoio; // Extract upper part and convert to uint32_t
				// Load the input float vector
				// Perform the multiplication with inp vector
				sum1 = MULTADD(u32_low_vec, vld1q_f32(B + bbt * INSHAPE + k),sum1);
				sum2 = MULTADD(u32_high_vec, vld1q_f32(B + bbt * INSHAPE + k + 4),sum2);

			}

			sum1 = ADD(sum1,sum2);
			
			zz1[i&3]= REDUCE(sum1);
			

		}

		
		STORE(
				(void *)(C + bbt * OUTSHAPE + dii + b),
				zz1 * Ario1);
		}
	}
	}
	#endif
}

void dopartialwkv5att(MatMulJob job) {
	auto T = job.bbt;
	auto CH = job.INSHAPE;
	auto bb = job.OUTSHAPE;
	auto kk = job.Ao;
	auto ww = job.C;
	auto vv = job.Ar;
	auto uu = job.Bt;
	auto rr = job.Ct;
	auto ss = job.ex;
	auto out = job.B;
	auto H = job.H;
	auto hh = job.hh;

	// 1d
	uint32_t bsize = H * T * CH;

	// 1d tensor
	uint32_t tsize = H * CH;
	// 2d tensor
	uint32_t ttsize = H * CH * CH;

	// 1d
	uint32_t hsize = CH;
	// 2d
	uint32_t hhsize = CH * CH;

	for (uint32_t t = 0; t < T; t++) {
		for (uint32_t i = 0; i < CH; i++) {
			auto btimeoffset = bb * bsize;
			auto timeoffset = btimeoffset + t * tsize;
			auto bbhsize = bb * ttsize;

			auto hoffset = hh * hsize;
			auto bhofseti = timeoffset + hoffset;
			auto bbhhsize = bbhsize + hh * hhsize;

			uint32_t iind = bhofseti + i;
			auto hoffseti = hoffset + i;
			auto bbhhofseti = bbhhsize + i * hsize;

			// auto kkk = kk[iind];
			auto kkk = SET1(kk[iind]);
			auto uuu = SET1(uu[hoffseti]);
			auto rrr = SET1(rr[iind]);
			auto www = SET1(ww[hoffseti]);

			for (uint32_t j = 0; j < CH; j += SIMD_WIDTH) {
				uint32_t jind = bhofseti + j;
				uint32_t sind = bbhhofseti + j;

				// atu = k[t,bb,hh,i]*v[t,bb,hh,j]
				auto vvv = LOAD(&vv[jind]);

				// multiply kkk and vvv
				auto atu = MULTIPLY(vvv, kkk);

				auto sss = LOAD(&ss[sind]);

				// out[t,bb,hh,j] += r[t,bb,hh,i]*(s[bb,hh,i,j] + atu*u[hh,i] )
				auto sssatuuuu = MULTADD(atu, uuu, sss);

				auto outtt = LOAD(&out[jind]);

				STORE((void *)&out[jind], MULTADD(sssatuuuu, rrr, outtt));

				// s[bb,hh,i,j] = s[bb,hh,i,j] * w[hh,i] + atu
				STORE((void *)&ss[sind], MULTADD(sss, www, atu));
			}
		}
	}
}

void listenfunc(std::atomic<ulong> *jobs1, std::atomic<ulong> *jobs2) {
	// wait for all jobs to be done
	while (true) {
		// check if all jobs are done

		// get last job
		const auto currenta = jobs1->load();
		if (currenta != 0) {
			const auto current = *(MatMulJob *)currenta;

			if (current.type == JOBTYPE::RWKV_ATT) {
				dopartialwkv5att(current);
			} else {
				dopartial(current);
			}

			jobs1->store(ZeroLong);
		}
		const auto current2 = jobs2->load();
		if (current2 != 0) {
			const auto current = *(MatMulJob *)current2;

			dopartial(current);
			jobs2->store(ZeroLong);
			
			
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

	t1 = new std::thread(listenfunc, &jobs10, &jobs11);
	t2 = new std::thread(listenfunc, &jobs12, &jobs13);
	t3 = new std::thread(listenfunc, &jobs20, &jobs21);
	t4 = new std::thread(listenfunc, &jobs22, &jobs23);
	t5 = new std::thread(listenfunc, &jobs30, &jobs31);
	t6 = new std::thread(listenfunc, &jobs32, &jobs33);
	t7 = new std::thread(listenfunc, &jobs40, &jobs41);
	t8 = new std::thread(listenfunc, &jobs42, &jobs43);
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
	const ulong INSHAPE = Bt.shape[2];
	const ulong OUTSHAPE = Ct.shape[2];

	startWorkers();

	auto job10job = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, BB*T, 0, INSHAPE, OUTSHAPE };
	auto job11job = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, BB*T, 16, INSHAPE, OUTSHAPE };
	auto job12job = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, BB*T, 32, INSHAPE, OUTSHAPE };
	auto job13job = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, BB*T, 48, INSHAPE, OUTSHAPE };
	auto job20job = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, BB*T, 64, INSHAPE, OUTSHAPE };
	auto job21job = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, BB*T, 80, INSHAPE, OUTSHAPE };
	auto job22job = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, BB*T, 96, INSHAPE, OUTSHAPE };
	auto job23job = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, BB*T, 112, INSHAPE, OUTSHAPE };
	auto job30job = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, BB*T, 128, INSHAPE, OUTSHAPE };
	auto job31job = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, BB*T, 144, INSHAPE, OUTSHAPE };
	auto job32job = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, BB*T, 160, INSHAPE, OUTSHAPE };
	auto job33job = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, BB*T, 176, INSHAPE, OUTSHAPE };
	auto job40job = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, BB*T, 192, INSHAPE, OUTSHAPE };
	auto job41job = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, BB*T, 208, INSHAPE, OUTSHAPE };
	auto job42job = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, BB*T, 224, INSHAPE, OUTSHAPE };
	auto job43job = MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, BB*T, 240, INSHAPE, OUTSHAPE };


		jobs10 = job10job;
		jobs11 = job11job;
		jobs12 = job12job;
		jobs13 = job13job;
		jobs20 = job20job;
		jobs21 = job21job;
		jobs22 = job22job;
		jobs23 = job23job;
		jobs30 = job30job;
		jobs31 = job31job;
		jobs32 = job32job;
		jobs33 = job33job;
		jobs40 = job40job;
		jobs41 = job41job;
		jobs42 = job42job;
		jobs43 = job43job;

		while (
				jobs10 != 0 | jobs11 != 0 | jobs12 != 0 | jobs13 != 0 |
				jobs20 != 0 | jobs21 != 0 | jobs22 != 0 | jobs23 != 0 |
				jobs30 != 0 | jobs31 != 0 | jobs32 != 0 | jobs33 != 0 |
				jobs40 != 0 | jobs41 != 0 | jobs42 != 0 | jobs43 != 0) {
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

	startWorkers();

	uint32_t B = r.shape[0];
	uint32_t T = r.shape[1];
	uint32_t C = r.shape[2];
	uint32_t H = this->shape[1];

	// #pragma omp parallel for collapse(2) schedule(guided, 64) shared(kk, vv, ww, uu, rr, ss, out)
	for (uint32_t bb = 0; bb < B; bb++) {
		// heads are divisable by 8 I think
		for (uint32_t hh = 0; hh < H; hh += 8) {
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
				C / H,
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
				C / H,
				bb,
				JOBTYPE::RWKV_ATT,
				ss,
				H,
				hh + 1
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
				C / H,
				bb,
				JOBTYPE::RWKV_ATT,
				ss,
				H,
				hh + 2
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
				C / H,
				bb,
				JOBTYPE::RWKV_ATT,
				ss,
				H,
				hh + 3
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
				C / H,
				bb,
				JOBTYPE::RWKV_ATT,
				ss,
				H,
				hh + 4
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
				C / H,
				bb,
				JOBTYPE::RWKV_ATT,
				ss,
				H,
				hh + 5
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
				C / H,
				bb,
				JOBTYPE::RWKV_ATT,
				ss,
				H,
				hh + 6
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
				C / H,
				bb,
				JOBTYPE::RWKV_ATT,
				ss,
				H,
				hh + 7
			};

			jobs42 = (uint64_t)&job8;

			// wait for all jobs to be done
			while (
					jobs10 != 0 || jobs12 != 0 ||
					jobs20 != 0 || jobs22 != 0 ||
					jobs30 != 0 || jobs32 != 0 ||
					jobs40 != 0 || jobs42 != 0) {
						
			}
		}
	}
}

#endif // HVMLAVX512MAT8_CPP