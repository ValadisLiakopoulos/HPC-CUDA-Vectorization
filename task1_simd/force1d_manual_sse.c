#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <xmmintrin.h>

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
    float force = 0.0;

    __m128 x0_vector = _mm_set1_ps(x0);

    //we will be loading them 4 by 4 like SSE because 16 byte vectors
    for (size_t i = 0; i < N; i += 4) {
        
        __m128 positions_vector = _mm_loadu_ps(&positions[i]);

        //float r = x0 - positions[i];
        __m128 r_vector = _mm_sub_ps(x0_vector, positions_vector);
        //float r2 = r * r;
        __m128 r2_vector = _mm_mul_ps(r_vector, r_vector);
        //float s2 = rm2 / r2;
        __m128 s2_vector = _mm_div_ps(_mm_set1_ps(rm2), r2_vector);
        //float s6 = s2*s2*s2;
        __m128 s6_vector = _mm_mul_ps(_mm_mul_ps(s2_vector, s2_vector), s2_vector);


        //force += 12 * eps * (s6*s6 - s6) / r;
       __m128 term_vector = _mm_div_ps(_mm_mul_ps(_mm_set1_ps(12.0 * eps), _mm_sub_ps(_mm_mul_ps(s6_vector, s6_vector), s6_vector)), r_vector);
        force += term_vector[0] + term_vector[1] + term_vector[2] + term_vector[3];

        float r_values[4], r2_values[4],s2_values[4], s6_values[4],term_values[4];
        _mm_storeu_ps(r_values, r_vector);
        _mm_storeu_ps(r2_values, r2_vector);
        _mm_storeu_ps(s2_values, s2_vector);
        _mm_storeu_ps(s6_values, s6_vector);
        _mm_storeu_ps(term_values, term_vector);
        printf("r_vector values: %f %f %f %f\n", r_values[0], r_values[1], r_values[2], r_values[3]);
        printf("r2_vector values: %f %f %f %f\n", r2_values[0], r2_values[1], r2_values[2], r2_values[3]);
        printf("s2_vector values: %f %f %f %f\n", s2_values[0], s2_values[1], s2_values[2], s2_values[3]);
        printf("s6_vector values: %f %f %f %f\n", s6_values[0], s6_values[1], s6_values[2], s6_values[3]);
        printf("term_vector values: %f %f %f %f\n", term_values[0], term_values[1], term_values[2], term_values[3]);

        //printf("force: %f\n", force);
    }

    return force;
}

int main(int argc, const char** argv)
{
    ///init random number generator
	srand48(1);

	//declare,malloc && init positions
    //float *positions;
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
    for(size_t j = 0; j < 3; ++j )
        printf("Force acting at x_0=%lf : %lf\n", x0[j], f0[j]/repetitions);

    printf("elapsed time: %lf mus\n", 1e6*(end-start));
	return 0;
    _mm_free(positions);
}

