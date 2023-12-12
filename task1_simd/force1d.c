#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

double get_wtime(void) {
  struct timeval t;
  gettimeofday(&t, NULL);
  return (double)t.tv_sec + (double)t.tv_usec*1.0e-6;
}

/// parameters
const size_t N  = 1<<16; // system size
const float eps = 5.0;    // Lenard-Jones, eps
const float rm  = 0.1;   // Lenard-Jones, r_m


/// compute the Lennard-Jones force particle at position x0
float compute_force(float *positions, float x0)
{
	float rm2 = rm * rm;
	float force = 0.;
	for (size_t i=0; i<N; ++i) {
		float r = x0 - positions[i];
		float r2 = r * r; // r^2
		float s2 = rm2 / r2; // (rm/r)^2
		float s6 = s2*s2*s2; // (rm/r)^6
		force += 12 * eps * (s6*s6 - s6) / r;
	}
	return force;
}

int main(int argc, const char** argv)
{
    /// init random number generator
		srand48(1);

    float *positions;
		positions = malloc(N*sizeof(float));

		for (size_t i=0; i<N; i++)
			positions[i] = drand48()+0.1;

    /// timings
		double start, end;

    float x0[] = { 0., -0.1, -0.2 };
    float f0[] = { 0, 0, 0 };

    const size_t repetitions = 1000;
    start = get_wtime();
    for (size_t i = 0; i < repetitions; ++i )
    {
        for( size_t j = 0; j < 3; ++j )
            f0[j] += compute_force(positions, x0[j]);
    }
    end = get_wtime();

    for(size_t j = 0; j < 3; ++j )
        printf("Force acting at x_0=%lf : %lf\n", x0[j], f0[j]/repetitions);

    printf("elapsed time: %lf mus\n", 1e6*(end-start));
		return 0;
}

