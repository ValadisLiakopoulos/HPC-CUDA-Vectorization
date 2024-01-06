all: force1d_sse force1d_avx force1d_serial force1d_manual_sse force1d_manual_avx# force1d_simd_sse force1d_simd_avx

force1d_sse: force1d.c
	gcc -Ofast -msse4.2  -o force1d_sse force1d.c -fopt-info-vec

force1d_avx: force1d.c
	gcc -Ofast -mavx  -o force1d_avx force1d.c -fopt-info-vec

force1d_serial: force1d.c
	gcc -o force1d_serial force1d.c -Ofast -fopt-info-vec

force1d_manual_sse: force1d_manual_sse.c
	gcc -pg -o force1d_manual_sse force1d_manual_sse.c -O3 -fopt-info-vec

force1d_manual_avx: force1d_manual_avx.c
	gcc -pg -o force1d_manual_avx force1d_manual_avx.c -O3 -march=native -fopt-info-vec


.PHONY:
clean:
	rm -f force1d_sse force1d_avx force1d_serial force1d_manual_sse force1d_manual_avx


#gcc -msse4 -dM -E - < /dev/null | egrep "SSE|AVX" | sort
#gcc -mavx -dM -E - < /dev/null | egrep "SSE|AVX" | sort