all: forces_cpu  # forces_gpu

forces_cpu: forces_cpu.c
	gcc forces_cpu.c -g -O2 -Wall -o $@ -lm

.PHONY: clean
clean:
	rm -f forces_cpu
