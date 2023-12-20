#ifndef HVMLAVX512MAT8_CPP
#define HVMLAVX512MAT8_CPP
#include "hvml/tensor.hpp"
#include <atomic>
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
	const float *ex = nullptr;
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

std::atomic<bool> jobs10done = false;
std::atomic<bool> jobs11done = false;
std::atomic<bool> jobs12done = false;
std::atomic<bool> jobs13done = false;

std::atomic<bool> jobs20done = false;
std::atomic<bool> jobs21done = false;
std::atomic<bool> jobs22done = false;
std::atomic<bool> jobs23done = false;

std::atomic<bool> jobs30done = false;
std::atomic<bool> jobs31done = false;
std::atomic<bool> jobs32done = false;
std::atomic<bool> jobs33done = false;

std::atomic<bool> jobs40done = false;
std::atomic<bool> jobs41done = false;
std::atomic<bool> jobs42done = false;
std::atomic<bool> jobs43done = false;

void dopartial(MatMulJob job) {
	// do the work
	auto A = job.A;
	auto B = job.B;
	auto C = job.C;
	auto IN = job.IN;
	auto OUT = job.OUT;
	auto Ao = job.Ao;
	auto Ar = job.Ar;
	auto bbt = job.bbt;
	auto ii = job.ii;
	#ifdef __AVX512F__ 
		const auto Ario = _mm512_load_ps(Ar + ii);
		const auto Aoioo = _mm512_div_ps(_mm512_load_ps(Ao + ii), Ario);
		__m512 zz = _mm512_setzero_ps();
		for (uint32_t i = ii; i < ii + 16; i += 1) {
			const float Aoio = Aoioo[i & 15];

			__m512 aa = _mm512_setzero_ps();
			const auto IAIN = A + i * IN;
			for (uint32_t k = 0; k < IN; k += 32) {
				const __m512 b01 = _mm512_load_ps(B + bbt * IN + k + 16);
				const __m512 b00 = _mm512_load_ps(B + bbt * IN + k);

				const __m512 a00 = Aoio + _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(*((__m128i *)(IAIN + k))));
				const __m512 a01 = Aoio + _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(*((__m128i *)(IAIN + k + 16))));

				// aa = _mm512_dpbf16_ps(aa, _mm512_cvtne2ps_pbh(a00, a01), _mm512_cvtne2ps_pbh(b00, b01));
				aa = _mm512_fmadd_ps(a00, b00, aa);
				aa = _mm512_fmadd_ps(a01, b01, aa);
			}
			zz[i & 15] = _mm512_reduce_add_ps(aa);
		}
		_mm512_store_ps(
				(void *)(C + bbt * OUT + ii),
				zz * Ario);
	#endif

	#if defined(__ARM_NEON) || defined(__ARM_NEON__)
	 	for (ulong b = 0; b < 16; b+= 4){
		const auto Ario1 = LOAD(Ar + ii+b);
		const auto Aoio1 = DIVIDE(LOAD(Ao + ii + b),Ario1);

		auto zz1 = SET1(0.0);

		for (uint32_t i = ii+b; i < ii + b+4; i += 1) {
			auto Aoio = Aoio1[i&3];

			const auto IAIN = A + i * IN;

			auto sum1 = SET1(0.0);
			auto sum2 = SET1(0.0);
			
			for (uint32_t k = 0; k < IN; k += 8) {
				
				auto u16_vec = vmovl_u8(vld1_u8((IAIN + k)));  
				
					// Convert uint8_t values to float32x4_t
									// convert uint8_t to uint16_t
				auto u32_low_vec = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_vec)))+Aoio;   // Extract lower part and convert to uint32_t
				auto u32_high_vec = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_vec)))+Aoio; // Extract upper part and convert to uint32_t
				// Load the input float vector
				// Perform the multiplication with inp vector
				sum1 = MULTADD(u32_low_vec, vld1q_f32(B + bbt * IN + k),sum1);
				sum2 = MULTADD(u32_high_vec, vld1q_f32(B + bbt * IN + k + 4),sum2);

			}

			sum1 = sum1+sum2;
			
			zz1[i&3]= REDUCE(sum1);
			

		}

		
		STORE(
				(void *)(C + bbt * OUT + ii + b),
				zz1 * Ario1);
		}
	#endif
}

void dopartialwkv5att(MatMulJob job) {
	auto T = job.bbt;
	auto CH = job.IN;
	auto bb = job.OUT;
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

				STORE((void *)&out[jind], MULTADD(sssatuuuu, rrr, outtt));

				// s[bb,hh,i,j] = s[bb,hh,i,j] * w[hh,i] + atu
				STORE((void *)&ss[sind], MULTADD(sss, www, atu));
			}
		}
	}
}

void listen(std::atomic<ulong> *jobs1, std::atomic<bool> *jobs1done, std::atomic<ulong> *jobs2, std::atomic<bool> *jobs2done) {
	// wait for all jobs to be done
	while (true) {
		// check if all jobs are done

		// get last job
		const auto currenta = jobs1->load();
		if (currenta != 0) {
			const auto current = *(MatMulJob *)currenta;

			if (current.type == JOBTYPE::RWKV_ATT) {
				dopartialwkv5att(current);
				jobs1->store(0UL);
			} else {
				dopartial(current);
				jobs1->store(0UL);
				if (current.ii + 16 * 16 >= (current.OUT)) {
					jobs1done[0] = true;
				}
			}
		}
		const auto current2 = jobs2->load();
		if (current2 != 0) {
			const auto current = *(MatMulJob *)current2;

			dopartial(current);
			jobs2->store(0);
			if ((current).ii + 16 * 16 >= (current).OUT) {
				jobs2done[0] = true;
			}
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

	t1 = new std::thread(listen, &jobs10, &jobs10done, &jobs11, &jobs11done);
	t2 = new std::thread(listen, &jobs12, &jobs12done, &jobs13, &jobs13done);
	t3 = new std::thread(listen, &jobs20, &jobs20done, &jobs21, &jobs21done);
	t4 = new std::thread(listen, &jobs22, &jobs22done, &jobs23, &jobs23done);
	t5 = new std::thread(listen, &jobs30, &jobs30done, &jobs31, &jobs31done);
	t6 = new std::thread(listen, &jobs32, &jobs32done, &jobs33, &jobs33done);
	t7 = new std::thread(listen, &jobs40, &jobs40done, &jobs41, &jobs41done);
	t8 = new std::thread(listen, &jobs42, &jobs42done, &jobs43, &jobs43done);
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
		uint32_t outlayer[16] = {
			0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0
		};
		jobs10done = false;
		jobs11done = false;
		jobs12done = false;
		jobs13done = false;

		jobs20done = false;
		jobs21done = false;
		jobs22done = false;
		jobs23done = false;

		jobs30done = false;
		jobs31done = false;
		jobs32done = false;
		jobs33done = false;

		jobs40done = false;
		jobs41done = false;
		jobs42done = false;
		jobs43done = false;

		while ((outlayer[0] <= OUT) || (outlayer[1] <= OUT) || (outlayer[2] <= OUT) || (outlayer[3] <= OUT) ||
				(outlayer[4] <= OUT) || (outlayer[5] <= OUT) || (outlayer[6] <= OUT) || (outlayer[7] <= OUT) ||
				(outlayer[8] <= OUT) || (outlayer[9] <= OUT) || (outlayer[10] <= OUT) || (outlayer[11] <= OUT) ||
				(outlayer[12] <= OUT) || (outlayer[13] <= OUT) || (outlayer[14] <= OUT) || (outlayer[15] <= OUT)) {
			
			if (outlayer[0] < OUT) {
				auto cmp = 0UL;
				if (jobs10.compare_exchange_strong(cmp, (ulong) new MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, outlayer[0], IN, OUT })) {
					outlayer[0] += 16 * 16;
				}
			} else {
				if (jobs10done) {
					outlayer[0] += 16 * 16;
				}
			}

			if (outlayer[1] < OUT) {
				auto cmp = 0UL;
				if (jobs11.compare_exchange_strong(cmp, (ulong) new MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, outlayer[1] + 16, IN, OUT })) {
					outlayer[1] += 16 * 16;
				}
			} else {
				if (jobs11done) {
					outlayer[1] += 16 * 16;
				}
			}

			if (outlayer[2] < OUT) {
				auto cmp = 0UL;
				if (jobs12.compare_exchange_strong(cmp, (ulong) new MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, outlayer[2] + 32, IN, OUT })) {
					outlayer[2] += 16 * 16;
				}
			} else {
				if (jobs12done) {
					outlayer[2] += 16 * 16;
				}
			}

			if (outlayer[3] < OUT) {
				auto cmp = 0UL;
				if (jobs13.compare_exchange_strong(cmp, (ulong) new MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, outlayer[3] + 48, IN, OUT })) {
					outlayer[3] += 16 * 16;
				}
			} else {
				if (jobs13done) {
					outlayer[3] += 16 * 16;
				}
			}

			if (outlayer[4] < OUT) {
				auto cmp = 0UL;
				if (jobs20.compare_exchange_strong(cmp, (ulong) new MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, outlayer[4] + 64, IN, OUT })) {
					outlayer[4] += 16 * 16;
				}
			} else {
				if (jobs20done) {
					outlayer[4] += 16 * 16;
				}
			}

			if (outlayer[5] < OUT) {
				auto cmp = 0UL;
				if (jobs21.compare_exchange_strong(cmp, (ulong) new MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, outlayer[5] + 80, IN, OUT })) {
					outlayer[5] += 16 * 16;
				}
			} else {
				if (jobs21done) {
					outlayer[5] += 16 * 16;
				}
			}

			if (outlayer[6] < OUT) {
				auto cmp = 0UL;
				if (jobs22.compare_exchange_strong(cmp, (ulong) new MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, outlayer[6] + 96, IN, OUT })) {
					outlayer[6] += 16 * 16;
				}
			} else {
				if (jobs22done) {
					outlayer[6] += 16 * 16;
				}
			}

			if (outlayer[7] < OUT) {
				auto cmp = 0UL;
				if (jobs23.compare_exchange_strong(cmp, (ulong) new MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, outlayer[7] + 112, IN, OUT })) {
					outlayer[7] += 16 * 16;
				}
			} else {
				if (jobs23done) {
					outlayer[7] += 16 * 16;
				}
			}

			if (outlayer[8] < OUT) {
				auto cmp = 0UL;
				if (jobs30.compare_exchange_strong(cmp, (ulong) new MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, outlayer[8] + 128, IN, OUT })) {
					outlayer[8] += 16 * 16;
				}
			} else {
				if (jobs30done) {
					outlayer[8] += 16 * 16;
				}
			}

			if (outlayer[9] < OUT) {
				auto cmp = 0UL;
				if (jobs31.compare_exchange_strong(cmp, (ulong) new MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, outlayer[9] + 144, IN, OUT })) {
					outlayer[9] += 16 * 16;
				}
			} else {
				if (jobs31done) {
					outlayer[9] += 16 * 16;
				}
			}

			if (outlayer[10] < OUT) {
				auto cmp = 0UL;
				if (jobs32.compare_exchange_strong(cmp, (ulong) new MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, outlayer[10] + 160, IN, OUT })) {
					outlayer[10] += 16 * 16;
				}
			} else {
				if (jobs32done) {
					outlayer[10] += 16 * 16;
				}
			}

			if (outlayer[11] < OUT) {
				auto cmp = 0UL;
				if (jobs33.compare_exchange_strong(cmp, (ulong) new MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, outlayer[11] + 176, IN, OUT })) {
					outlayer[11] += 16 * 16;
				}
			} else {
				if (jobs33done) {
					outlayer[11] += 16 * 16;
				}
			}

			if (outlayer[12] < OUT) {
				auto cmp = 0UL;
				if (jobs40.compare_exchange_strong(cmp, (ulong) new MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, outlayer[12] + 192, IN, OUT })) {
					outlayer[12] += 16 * 16;
				}
			} else {
				if (jobs40done) {
					outlayer[12] += 16 * 16;
				}
			}

			if (outlayer[13] < OUT) {
				auto cmp = 0UL;
				if (jobs41.compare_exchange_strong(cmp, (ulong) new MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, outlayer[13] + 208, IN, OUT })) {
					outlayer[13] += 16 * 16;
				}
			} else {
				if (jobs41done) {
					outlayer[13] += 16 * 16;
				}
			}

			if (outlayer[14] < OUT) {
				auto cmp = 0UL;
				if (jobs42.compare_exchange_strong(cmp, (ulong) new MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, outlayer[14] + 224, IN, OUT })) {
					outlayer[14] += 16 * 16;
				}
			} else {
				if (jobs42done) {
					outlayer[14] += 16 * 16;
				}
			}

			if (outlayer[15] < OUT) {
				auto cmp = 0UL;
				if (jobs43.compare_exchange_strong(cmp, (ulong) new MatMulJob{ A, B, C, Ao, Ar, Bt.data, Ct.data, bbt, outlayer[15] + 240, IN, OUT })) {
					outlayer[15] += 16 * 16;
				}
			} else {
				if (jobs43done) {
					outlayer[15] += 16 * 16;
				}
			}
		}
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
		for (uint hh = 0; hh < H; hh += 8) {
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