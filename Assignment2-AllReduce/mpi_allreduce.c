#include <stdio.h>
#include <stdlib.h>
#include <string.h> // For memcpy and memcmp
#include <mpi.h>
#include <math.h>   // For log2

// Helper function to check if a number is a power of two
int is_power_of_two(int n) {
    if (n <= 0) {
        return 0;
    }
    return (n & (n - 1)) == 0;
}

/*
Performs an all-reduce operation using a linear exchange-based algorithm.
*/
void linear_exchange_allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (op != MPI_SUM || datatype != MPI_INT) {
        if (rank == 0) fprintf(stderr, "This implementation only supports MPI_SUM for MPI_INT\n");
        MPI_Abort(comm, 1);
    }

    int chunk_size = count / size;
    int* int_sendbuf = (int*)sendbuf;
    int* int_recvbuf = (int*)recvbuf;

    int* pw_recvbuf = (int*)malloc(sizeof(int) * count);
    int* tempbuf = (int*)malloc(sizeof(int) * count);
    memset(pw_recvbuf, 0, sizeof(int) * count);
    
    // Phase 1: Reduce-Scatter 
    
    MPI_Request send_request[size], recv_request[size];
    for (int i = 0; i < size; i++) {
        int send_to = (rank + i) % size;
        int recv_from = (rank - i + size) % size;
        
        MPI_Isend(&int_sendbuf[send_to * chunk_size], chunk_size, MPI_INT, send_to, 0, comm, &send_request[i]);
        MPI_Irecv(&tempbuf[recv_from * chunk_size], chunk_size, MPI_INT, recv_from, 0, comm, &recv_request[i]);
    }

    // Use individual MPI_Wait calls in a loop
    for (int i = 0; i < size; i++) {
        MPI_Wait(&send_request[i], MPI_STATUS_IGNORE);
        MPI_Wait(&recv_request[i], MPI_STATUS_IGNORE);
    }

    // Perform the reduction into pw_recvbuf
    for(int j = 0; j < chunk_size; j++) {
       for(int i = 0; i < size; i++) {
          pw_recvbuf[rank * chunk_size + j] += tempbuf[i * chunk_size + j];
       }
    }
    // End of reduce-scatter phase
    // allgather phase TODO
    
    // Phase 2: Allgather (Completing the TODO part)
    
    // Copy the locally reduced chunk from pw_recvbuf to the final recvbuf
    memcpy(&int_recvbuf[rank * chunk_size], &pw_recvbuf[rank * chunk_size], chunk_size * sizeof(int));

    // Now, gather all the final chunks from other processes using a ring algorithm
    int right = (rank + 1) % size;
    int left = (rank - 1 + size) % size;

    for (int i = 0; i < size - 1; i++) {
        int send_chunk_idx = (rank - i + size) % size;
        int recv_chunk_idx = (rank - i - 1 + size) % size;

        MPI_Sendrecv(&int_recvbuf[send_chunk_idx * chunk_size], chunk_size, MPI_INT, right, 1,
                     &int_recvbuf[recv_chunk_idx * chunk_size], chunk_size, MPI_INT, left, 1,
                     comm, MPI_STATUS_IGNORE);
    }
    
    free(tempbuf);
    free(pw_recvbuf);
}


/*
Performs an all-reduce operation using a ring-based algorithm.
 */
void ring_allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (op != MPI_SUM || datatype != MPI_INT) {
        if (rank == 0) fprintf(stderr, "This implementation only supports MPI_SUM for MPI_INT\n");
        MPI_Abort(comm, 1);
    }

    int chunk_size = count / size;
    int* int_sendbuf = (int*)sendbuf;
    int* int_recvbuf = (int*)recvbuf;
    
    memcpy(recvbuf, sendbuf, count * sizeof(int));
    
    int* temp_chunk = (int*)malloc(chunk_size * sizeof(int));
    
    int right = (rank + 1) % size;
    int left = (rank - 1 + size) % size;

    // Phase 1: Reduce-Scatter
    for (int i = 0; i < size - 1; i++) {
        int send_chunk_idx = (rank - i + size) % size;
        int recv_chunk_idx = (rank - i - 1 + size) % size;

        MPI_Sendrecv(&int_recvbuf[send_chunk_idx * chunk_size], chunk_size, MPI_INT, right, 0,
                     temp_chunk, chunk_size, MPI_INT, left, 0,
                     comm, MPI_STATUS_IGNORE);

        for (int j = 0; j < chunk_size; j++) {
            int_recvbuf[recv_chunk_idx * chunk_size + j] += temp_chunk[j];
        }
    }

    // Phase 2: All-Gather
    for (int i = 0; i < size - 1; i++) {
        int send_chunk_idx = (rank - i + 1 + size) % size;
        int recv_chunk_idx = (rank - i + size) % size;

        MPI_Sendrecv(&int_recvbuf[send_chunk_idx * chunk_size], chunk_size, MPI_INT, right, 0,
                     &int_recvbuf[recv_chunk_idx * chunk_size], chunk_size, MPI_INT, left, 0,
                     comm, MPI_STATUS_IGNORE);
    }
    
    free(temp_chunk);
}


/*
Performs an all-reduce operation using Rabenseifner's algorithm.
 */
void rabenseifner_allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (op != MPI_SUM || datatype != MPI_INT) {
        if (rank == 0) fprintf(stderr, "This implementation only supports MPI_SUM for MPI_INT\n");
        MPI_Abort(comm, 1);
    }

    if (!is_power_of_two(size)) {
        if (rank == 0) fprintf(stderr, "Rabenseifner's algorithm implementation requires a power-of-two number of processes.\n");
        memcpy(recvbuf, sendbuf, count * sizeof(int)); 
        return;
    }

    int* int_sendbuf = (int*)sendbuf;
    int* int_recvbuf = (int*)recvbuf;
    int chunk_size = count / size;

    memcpy(recvbuf, sendbuf, count * sizeof(int));
    int* temp_buf = (int*)malloc(count * sizeof(int));

    // Phase 1: Reduce-Scatter (Recursive Halving)
    int current_chunk_count = count;
    for (int dist = size >> 1; dist > 0; dist >>= 1) {
        int partner = rank ^ dist;
        current_chunk_count /= 2;

        if (rank < partner) {
            MPI_Sendrecv(int_recvbuf + current_chunk_count, current_chunk_count, MPI_INT, partner, 0,
                         temp_buf, current_chunk_count, MPI_INT, partner, 0,
                         comm, MPI_STATUS_IGNORE);
            for (int i = 0; i < current_chunk_count; i++) int_recvbuf[i] += temp_buf[i];
        } else {
            MPI_Sendrecv(int_recvbuf, current_chunk_count, MPI_INT, partner, 0,
                         temp_buf, current_chunk_count, MPI_INT, partner, 0,
                         comm, MPI_STATUS_IGNORE);
            for (int i = 0; i < current_chunk_count; i++) temp_buf[i] += int_recvbuf[current_chunk_count + i];
            memcpy(int_recvbuf, temp_buf, current_chunk_count * sizeof(int));
        }
    }

    if (rank * chunk_size != 0) {
        memcpy(&int_recvbuf[rank * chunk_size], int_recvbuf, chunk_size * sizeof(int));
    }
    for(int i = 0; i < rank * chunk_size; i++) int_recvbuf[i] = 0;
    for(int i = (rank+1) * chunk_size; i < count; i++) int_recvbuf[i] = 0;

    // Phase 2: Allgather (Recursive Doubling)
    for (int dist = 1; dist < size; dist <<= 1) {
        int partner = rank ^ dist;
        int send_size = dist * chunk_size;
        int recv_size = send_size;
        int send_offset = (rank & ~(dist - 1)) * chunk_size;
        int recv_offset = (partner & ~(dist - 1)) * chunk_size;

        MPI_Sendrecv(int_recvbuf + send_offset, send_size, MPI_INT, partner, 0,
                     int_recvbuf + recv_offset, recv_size, MPI_INT, partner, 0,
                     comm, MPI_STATUS_IGNORE);
    }
    
    free(temp_buf);
}

int main(int argc, char** argv) {
    int rank, size;
    int count = 16;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // NEW: Check if the number of processes divides the total count.
    if (count % size != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: The number of elements (%d) must be divisible by the number of processes (%d).\n", count, size);
            fprintf(stderr, "Please use a process count that is a divisor of 16 (e.g., 1, 2, 4, 8, 16).\n");
        }
        MPI_Finalize();
        return 1;
    }

    int* sendbuf      = (int*)malloc(sizeof(int) * count);
    int* mpi_recvbuf  = (int*)malloc(sizeof(int) * count);
    int* linear_recvbuf = (int*)malloc(sizeof(int) * count);
    int* ring_recvbuf   = (int*)malloc(sizeof(int) * count);
    int* rab_recvbuf  = (int*)malloc(sizeof(int) * count);

    for (int i = 0; i < count; i++) {
        sendbuf[i] = rank * count + i + 1;
    }

    // Run all algorithms first
    MPI_Allreduce(sendbuf, mpi_recvbuf, count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    linear_exchange_allreduce(sendbuf, linear_recvbuf, count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    ring_allreduce(sendbuf, ring_recvbuf, count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    rabenseifner_allreduce(sendbuf, rab_recvbuf, count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    // Ensure all computations are finished before printing starts
    MPI_Barrier(MPI_COMM_WORLD);

    // Coordinated printing of Initial and Final data from All Ranks

    //Print Initial Data
    if (rank == 0) {
        printf("\n==================== INITIAL INPUT DATA ====================\n");
        fflush(stdout); // Flush the buffer to ensure the header prints first
    }
    MPI_Barrier(MPI_COMM_WORLD); // Wait for header to print

    for (int i = 0; i < size; i++) {
        if (rank == i) {
            printf("Rank %d Input:  ", rank);
            for (int j = 0; j < count; j++) printf("%d ", sendbuf[j]);
            printf("\n");
            fflush(stdout); // Ensure this rank's output is printed immediately
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    //Print Final Validated Results
    if (rank == 0) {
        printf("\n==================== FINAL VALIDATED RESULTS ===================\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < size; i++) {
        if (rank == i) {
            printf("\n-------------------- RANK %d FINAL RESULTS --------------------\n", rank);
            
            printf("MPI_Allreduce Result:   ");
            for (int j = 0; j < count; j++) printf("%d ", mpi_recvbuf[j]);
            printf("\n");

            printf("Linear Exchange Result: ");
            for (int j = 0; j < count; j++) printf("%d ", linear_recvbuf[j]);
            if (memcmp(mpi_recvbuf, linear_recvbuf, count * sizeof(int)) == 0) {
                printf("... [MEMCPY SUCCESS]\n");
            } else {
                printf("... [FAILURE]\n");
            }

            printf("Ring Allreduce Result:  ");
            for (int j = 0; j < count; j++) printf("%d ", ring_recvbuf[j]);
            if (memcmp(mpi_recvbuf, ring_recvbuf, count * sizeof(int)) == 0) {
                printf("... [MEMCPY SUCCESS]\n");
            } else {
                printf("... [FAILURE]\n");
            }

            printf("Rabenseifner Result:    ");
            if (!is_power_of_two(size)) {
                printf("... [SKIPPED: Process count %d is not a power of two]\n", size);
            } else {
                for (int j = 0; j < count; j++) printf("%d ", rab_recvbuf[j]);
                if (memcmp(mpi_recvbuf, rab_recvbuf, count * sizeof(int)) == 0) {
                    printf("... [MEMCPY SUCCESS]\n");
                } else {
                    printf("... [FAILURE]\n");
                }
            }
            fflush(stdout);
        }
        // This barrier is crucial. It ensures that Rank `i` finishes printing
        // its entire block before Rank `i+1` starts.
        MPI_Barrier(MPI_COMM_WORLD);
    }

    free(sendbuf);
    free(mpi_recvbuf);
    free(linear_recvbuf);
    free(ring_recvbuf);
    free(rab_recvbuf);
    
    MPI_Finalize();
    return 0;
}