#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int rank, size;
    long N = 10000000;  // Vector size
    double *A = NULL;
    double local_sum = 0.0, global_sum = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Compute counts for uneven distribution
    int *counts = (int*)malloc(size * sizeof(int));
    int *displs = (int*)malloc(size * sizeof(int));
    long base = N / size;
    long remainder = N % size;

    for (int i = 0; i < size; i++) {
        counts[i] = base + (i < remainder ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i-1] + counts[i-1];
    }

    // Allocate local array
    double *local_A = (double*)malloc(counts[rank] * sizeof(double));

    // Initialize vector only on root
    if (rank == 0) {
        A = (double*)malloc(N * sizeof(double));
        for (long i = 0; i < N; i++)
            A[i] = i + 1;
    }

    // Scatter vector to all processes (uneven)
    MPI_Scatterv(A, counts, displs, MPI_DOUBLE,
                 local_A, counts[rank], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Measure start time
    double start_time = MPI_Wtime();

    // Local computation
    for (int i = 0; i < counts[rank]; i++)
        local_sum += local_A[i];

    // Compute global sum and broadcast to all processes
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Measure end time
    double end_time = MPI_Wtime();

    if (rank == 0) {
        double expected = (N * (N + 1)) / 2.0;
        double average = global_sum / N;
        printf("Total Sum = %.0f | Expected = %.0f | Average = %.5f\n",
               global_sum, expected, average);
        printf("Execution Time = %.6f seconds\n", end_time - start_time);
        free(A);
    }

    free(local_A);
    free(counts);
    free(displs);
    MPI_Finalize();
    return 0;
}
