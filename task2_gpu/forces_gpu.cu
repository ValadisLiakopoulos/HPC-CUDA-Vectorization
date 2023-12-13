#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define PRINTSTATS
#define BLOCK_SIZE 256

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

typedef struct results_s
{
  float sfx, sfy, sfz;
  double minfx, maxfx;
  double minfy, maxfy;
  double minfz, maxfz;
} results_t;


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
__global__ void computeGravitationalForces(Particle_t *particles, int n)
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
				double tmp = pow(particles[i].x - particles[j].x, 2.0) +
							 pow(particles[i].y - particles[j].y, 2.0) +
							 pow(particles[i].z - particles[j].z, 2.0);

				double magnitude = G * particles[i].m * particles[j].m / pow(tmp, 1.5);

				particles[i].fx += (particles[i].x - particles[j].x) * magnitude;
				particles[i].fy += (particles[i].y - particles[j].y) * magnitude;
				particles[i].fz += (particles[i].z - particles[j].z) * magnitude;
			}
	}
}


__global__ void printStatistics(Particle_t *particles, int n, results_t *results)
{
	results[0].sfx = 0, results[0].sfy = 0, results[0].sfz = 0;
	results[0].maxfx = particles[0].fx;
	results[0].minfx = particles[0].fx;
	results[0].maxfy = particles[0].fy;
	results[0].minfy = particles[0].fy;
	results[0].maxfz = particles[0].fz;
	results[0].minfz = particles[0].fz;
	int localIdx = threadIdx.x;
	__shared__ float sfx[BLOCK_SIZE];
	__shared__ float sfy[BLOCK_SIZE];
    __shared__ float sfz[BLOCK_SIZE];
  
  float local_sfx=0.;
  float local_sfy=0.;
  float local_sfz=0.;

  for (int i=0; i<n; i++) {
		if (results[0].minfx < particles[i].fx) results[0].minfx = particles[i].fx;
		if (results[0].maxfx > particles[i].fx) results[0].maxfx = particles[i].fx;
		if (results[0].minfy < particles[i].fy) results[0].minfy = particles[i].fy;
		if (results[0].maxfy > particles[i].fy) results[0].maxfy = particles[i].fy;
		if (results[0].minfz < particles[i].fz) results[0].minfz = particles[i].fz;
		if (results[0].maxfz > particles[i].fz) results[0].maxfz = particles[i].fz;
		
		//results[0].sfx += particles[i].fx;
		//results[0].sfy += particles[i].fy;
		//results[0].sfz += particles[i].fz;
		//atomicAdd(&local_sfx, (float)particles[i].fx);
		//atomicAdd(&local_sfy, (float)particles[i].fy);
		//atomicAdd(&local_sfz, (float)particles[i].fz);
		local_sfx += particles[i].fx;
		local_sfy += particles[i].fy;
		local_sfz += particles[i].fz;
	}
	
	sfx[localIdx] = local_sfx;
    sfy[localIdx] = local_sfy;
    sfz[localIdx] = local_sfz;


    __syncthreads();

  if(threadIdx.x == 0 && blockIdx.x == 0) {
    for (int index=1; index<threadIdx.x; index++) 
		{
            sfx[0] += sfx[index];
            sfy[0] += sfy[index];
            sfz[0] += sfz[index];
		}
  }

    if (localIdx == 0) {
        atomicAdd(&results[0].sfx, sfx[0]);
        atomicAdd(&results[0].sfy, sfy[0]);
        atomicAdd(&results[0].sfz, sfz[0]);
    }



	//results[0].sfx = local_sfx;
	//results[0].sfy = local_sfy;
	//results[0].sfz = local_sfz;
	
	//printf("%d particles: sfx=%e sfy=%e sfz=%e\n", n, sfx, sfy, sfz);
	//printf("%d particles: minfx=%f maxfx=%f\n", n, minfx, maxfx);
	//printf("%d particles: minfy=%f maxfy=%f\n", n, minfy, maxfy);
	//printf("%d particles: minfz=%f maxfz=%f\n", n, minfz, maxfz);
}


int main(int argc, char *argv[])
{
	int n;

	if (argc == 2)
		n = (1 << atoi(argv[1]));
	else
		n = (1 << 14);

	Particle_t *particles = (Particle_t *)malloc(n*sizeof(Particle_t));

	Particle_t *gpu_particles;
	initParticles(particles, n);
	cudaMalloc(&gpu_particles, n * sizeof(Particle_t));
	cudaMemcpy(gpu_particles, particles, n * sizeof(Particle_t), cudaMemcpyHostToDevice);

	//We need cudaEvent to measure time accurately
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//double t0 = get_wtime();
	cudaEventRecord(start);
	computeGravitationalForces<<<4,4>>>(gpu_particles, n);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop); //calculate elapsed time from GPU execution

	//double t1 = get_wtime();
	//cudaMemcpy(particles, gpu_particles, n * sizeof(Particle_t), cudaMemcpyDeviceToHost);

#if defined(PRINTSTATS)
	results_t results;
	results_t *gpu_results;

	cudaMalloc(&gpu_results, sizeof(results_t));
	printStatistics<<<1,1>>>(gpu_particles, n, gpu_results); //print statistics of forces through GPU calculation to minimize transfers
	cudaMemcpy(&results, gpu_results, sizeof(results_t), cudaMemcpyDeviceToHost);

	printf("%d particles: sfx=%e sfy=%e sfz=%e\n", n, results.sfx, results.sfy, results.sfz);
	printf("%d particles: minfx=%f maxfx=%f\n", n, results.minfx, results.maxfx);
	printf("%d particles: minfy=%f maxfy=%f\n", n, results.minfy, results.maxfy);
	printf("%d particles: minfz=%f maxfz=%f\n", n, results.minfz, results.maxfz);
#endif
	printf("Elapsed time=%lf seconds\n", milliseconds/1000.0);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(gpu_particles);
	cudaFree(gpu_results);
	return 0;
}
