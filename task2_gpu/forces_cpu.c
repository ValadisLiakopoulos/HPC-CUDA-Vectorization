#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

//#define PRINTSTATS

double get_wtime(void) {
  struct timeval t;
  gettimeofday(&t, NULL);
  return (double)t.tv_sec + (double)t.tv_usec*1.0e-6;
}

typedef struct Particle_s
{
  double x, y, z, m;
  double fx, fy, fz;
} Particle_t;


void initParticles(Particle_t *particles, int n)
{
	srand48(10);
  for (int i=0; i<n; i++) {
    particles[i].x = 10*drand48();
    particles[i].y = 10*drand48();
    particles[i].z = 10*drand48();
    particles[i].m = 1e7 / sqrt((double)n) *drand48();
  }
}

/**
 * Compute the gravitational forces in the system of particles
 * Symmetry of the forces is NOT exploited
**/
void computeGravitationalForces(Particle_t *particles, int n)
{
	const double G = 6.67408e-11;

	for (int i=0; i<n; i++)
	{
		particles[i].fx = 0;
		particles[i].fy = 0;
		particles[i].fz = 0;

		for (int j=0; j<n; j++)
			if (i!=j)
			{
				double tmp = pow(particles[i].x-particles[j].x, 2.0) +
							 pow(particles[i].y-particles[j].y, 2.0) +
							 pow(particles[i].z-particles[j].z, 2.0);

				double magnitude = G * particles[i].m * particles[j].m / pow(tmp, 1.5);

				particles[i].fx += (particles[i].x-particles[j].x) * magnitude;
				particles[i].fy += (particles[i].y-particles[j].y) * magnitude;
				particles[i].fz += (particles[i].z-particles[j].z) * magnitude;
			}
	}
}


void printStatistics(Particle_t *particles, int n)
{
	double sfx = 0, sfy = 0, sfz = 0;
  double maxfx = particles[0].fx;
  double minfx = particles[0].fx;
  double maxfy = particles[0].fy;
  double minfy = particles[0].fy;
  double maxfz = particles[0].fz;
  double minfz = particles[0].fz;
  for (int i=0; i<n; i++) {
		if (minfx < particles[i].fx) minfx = particles[i].fx;
		if (maxfx > particles[i].fx) maxfx = particles[i].fx;
		if (minfy < particles[i].fy) minfy = particles[i].fy;
		if (maxfy > particles[i].fy) maxfy = particles[i].fy;
		if (minfz < particles[i].fz) minfz = particles[i].fz;
		if (maxfz > particles[i].fz) maxfz = particles[i].fz;
		sfx += particles[i].fx;
		sfy += particles[i].fy;
		sfz += particles[i].fz;
	}

	printf("%d particles: sfx=%e sfy=%e sfz=%e\n", n, sfx, sfy, sfz);
	printf("%d particles: minfx=%f maxfx=%f\n", n, minfx, maxfx);
	printf("%d particles: minfy=%f maxfy=%f\n", n, minfy, maxfy);
	printf("%d particles: minfz=%f maxfz=%f\n", n, minfz, maxfz);
}


int main(int argc, char *argv[])
{
	int n;

	if (argc == 2)
		n = (1 << atoi(argv[1]));
	else
		n = (1 << 14);

	Particle_t *particles = (Particle_t *)malloc(n*sizeof(Particle_t));

	initParticles(particles, n);

	double t0 = get_wtime();
	computeGravitationalForces(particles, n);
	double t1 = get_wtime();

#if defined(PRINTSTATS)
	printStatistics(particles, n);
#endif

	printf("Elapsed time=%lf seconds\n", t1-t0);

	return 0;
}
