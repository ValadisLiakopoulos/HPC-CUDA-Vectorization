#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <xmmintrin.h>

//so i dont get the vectorization from the compiler
#pragma GCC optimize("no-tree-vectorize")

double get_wtime(void) {
  struct timeval t;
  gettimeofday(&t, NULL);
  return (double)t.tv_sec + (double)t.tv_usec*1.0e-6;
}

///parameters
const size_t N  = 1<<16;    // system size
const float eps = 5.0;      // Lenard-Jones, eps
const float rm  = 0.1;      // Lenard-Jones, r_m

///compute the Lennard-Jones force particle at position x0
float compute_force(float *positions, float x0)
{
    float rm2 = rm * rm;
    __m128 force_vec = _mm_set1_ps(0);
    __m128 x0_vector = _mm_set1_ps(x0);
    __m128 twelve_eps = _mm_set1_ps(12*eps);

    //we will be loading them 4 by 4 like SSE because 16 byte vectors
    //unroll the loop so the are being brought 8 by 8
    for (size_t i = 0; i < N; i += 16) {
        
        __m128 positions_vector0 = _mm_loadu_ps(&positions[i]);
        __m128 positions_vector1 = _mm_loadu_ps(&positions[i+4]);
        __m128 positions_vector2 = _mm_loadu_ps(&positions[i+8]);
        __m128 positions_vector3 = _mm_loadu_ps(&positions[i+12]);

        //float r = x0 - positions[i];
        __m128 r_vector0 = _mm_sub_ps(x0_vector, positions_vector0);
        __m128 r_vector1 = _mm_sub_ps(x0_vector, positions_vector1);
        __m128 r_vector2 = _mm_sub_ps(x0_vector, positions_vector2);
        __m128 r_vector3 = _mm_sub_ps(x0_vector, positions_vector3);

        //float r2 = r * r;
        __m128 r2_vector0 = _mm_mul_ps(r_vector0, r_vector0);
        __m128 r2_vector1 = _mm_mul_ps(r_vector1, r_vector1);
        __m128 r2_vector2 = _mm_mul_ps(r_vector2, r_vector2);
        __m128 r2_vector3 = _mm_mul_ps(r_vector3, r_vector3);

        //float s2 = rm2 / r2;
        __m128 s2_vector0 = _mm_div_ps(_mm_set1_ps(rm2), r2_vector0);
        __m128 s2_vector1 = _mm_div_ps(_mm_set1_ps(rm2), r2_vector1);
        __m128 s2_vector2 = _mm_div_ps(_mm_set1_ps(rm2), r2_vector2);
        __m128 s2_vector3 = _mm_div_ps(_mm_set1_ps(rm2), r2_vector3);

        //float s6 = s2*s2*s2;
        __m128 s6_vector0 = _mm_mul_ps(_mm_mul_ps(s2_vector0, s2_vector0), s2_vector0);
        __m128 s6_vector1 = _mm_mul_ps(_mm_mul_ps(s2_vector1, s2_vector1), s2_vector1);
        __m128 s6_vector2 = _mm_mul_ps(_mm_mul_ps(s2_vector2, s2_vector2), s2_vector2);
        __m128 s6_vector3 = _mm_mul_ps(_mm_mul_ps(s2_vector3, s2_vector3), s2_vector3);

        //force += 12 * eps * (s6*s6 - s6) / r;
        __m128 term_vector0 = _mm_div_ps(_mm_mul_ps(twelve_eps, _mm_sub_ps(_mm_mul_ps(s6_vector0, s6_vector0), s6_vector0)), r_vector0);
        __m128 term_vector1 = _mm_div_ps(_mm_mul_ps(twelve_eps, _mm_sub_ps(_mm_mul_ps(s6_vector1, s6_vector1), s6_vector1)), r_vector1);
        __m128 term_vector2 = _mm_div_ps(_mm_mul_ps(twelve_eps, _mm_sub_ps(_mm_mul_ps(s6_vector2, s6_vector2), s6_vector2)), r_vector2);
        __m128 term_vector3 = _mm_div_ps(_mm_mul_ps(twelve_eps, _mm_sub_ps(_mm_mul_ps(s6_vector3, s6_vector3), s6_vector3)), r_vector3);

        force_vec = _mm_add_ps(force_vec, term_vector0);
        force_vec = _mm_add_ps(force_vec, term_vector1);
        force_vec = _mm_add_ps(force_vec, term_vector2);
        force_vec = _mm_add_ps(force_vec, term_vector3);
       //force += term_vector[0] + term_vector[1] + term_vector[2] + term_vector[3];
    }

    //get the values from the force_vec, add them and send them back
    float force_array[4];
    _mm_storeu_ps(force_array, force_vec);
    float force=force_array[0]+force_array[1]+force_array[2]+force_array[3];

    return force;
}

int main(int argc, const char** argv)
{
    ///init random number generator
	srand48(1);

	//declare,malloc && init positions
    //allign memory
	float *positions = (float*)_mm_malloc(N * sizeof(float), 16);

	for (size_t i=0; i<N; i++)
		positions[i] = drand48()+0.1;


    ///timings
	double start, end;

	//declare and init x0 && f0
    float x0[] = { 0., -0.1, -0.2 };
    float f0[] = { 0, 0, 0 };

    //start the loop
    const size_t repetitions = 1000;
    start = get_wtime();
    for (size_t i = 0; i < repetitions; ++i )
    {
        for( size_t j = 0; j < 3; ++j )
            f0[j] += compute_force(positions, x0[j]);
    }
    end = get_wtime();

    //print results && elapsed time
    for(size_t j = 0; j < 3; ++j)
        printf("Force acting at x_0=%lf : %lf\n", x0[j], f0[j]/repetitions);

    printf("elapsed time: %lf mus\n", 1e6*(end-start));
	
    _mm_free(positions);
    return 0;
}
