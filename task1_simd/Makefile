all: force1d_sse force1d_avx force1d_serial force1d_manual_sse# force1d_simd_sse force1d_simd_avx

force1d_sse: force1d.c
	gcc -Ofast -msse4.2  -o force1d_sse force1d.c

force1d_avx: force1d.c
	gcc -Ofast -mavx  -o force1d_avx force1d.c

force1d_serial: force1d.c
	gcc -o force1d_serial force1d.c -Ofast

force1d_manual_sse: force1d_manual_sse.c
	gcc -o force1d_manual_sse force1d_manual_sse.c -Ofast

.PHONY:
clean:
	rm -f force1d_sse force1d_avx force1d_serial force1d_manual_sse
