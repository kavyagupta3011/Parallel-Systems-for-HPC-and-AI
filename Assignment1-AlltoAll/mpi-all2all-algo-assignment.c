//IMT2023016 Kavya Gupta
#include <stdio.h>
#include <stdlib.h>
#include <string.h> // For sprintf
#include <mpi.h>
#include <math.h>

// --- Helper Functions for Bandwidth-Optimal Algorithm (Unchanged) ---

int factorize(int n, int* factors) {
    int count = 0;
    int d = 2;
    while (n > 1) {
        while (n % d == 0) {
            factors[count++] = d;
            n /= d;
        }
        d++;
        if (d * d > n) {
            if (n > 1) {
                factors[count++] = n;
            }
            break;
        }
    }
    return count;
}

void to_variable_base(int n, int* digits, const int* bases, int num_bases) {
    int temp = n;
    for (int i = 0; i < num_bases; i++) {
        digits[i] = temp % bases[i];
        temp /= bases[i];
    }
}

int from_variable_base(const int* digits, const int* bases, int num_bases) {
    int n = 0;
    int product = 1;
    for (int i = 0; i < num_bases; i++) {
        n += digits[i] * product;
        product *= bases[i];
    }
    return n;
}

/**
 * @brief Prints the contents of a local buffer from each process, one by one.
 *
 * @param rank The rank of the current process.
 * @param size The total number of processes.
 * @param buf  The local buffer to print.
 * @param label A label for the printout (e.g., "sent", "received").
 */
void print_buffer_sequentially(int rank, int size, const int* buf, const char* label) {
    char print_str[4096] = {0};
    int len = 0;

    // Each rank waits its turn to print to avoid jumbled output.
    for (int i = 0; i < size; i++) {
        if (rank == i) {
            len += sprintf(print_str + len, "Rank %d %s: [", rank, label);
            for (int j = 0; j < size; j++) {
                len += sprintf(print_str + len, " %-4d", buf[j]);
            }
            printf("%s ]\n", print_str);
        }
        // A barrier here ensures strict print order, but can slow down execution.
        // For logging purposes, we'll place barriers between major sections instead.
    }
}


int main(int argc, char** argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int* sendbuf = (int*)malloc(size * sizeof(int));
    int* ref_recvbuf = (int*)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        sendbuf[i] = rank * 100 + i;
    }

    // --- Initial Data ---
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("\n--- Initial Sent Data ---\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    print_buffer_sequentially(rank, size, sendbuf, "sent");


    // --- Standard MPI_Alltoall (for reference) ---
    MPI_Alltoall(sendbuf, 1, MPI_INT, ref_recvbuf, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("\n---  MPI_Alltoall Results ---\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    print_buffer_sequentially(rank, size, ref_recvbuf, "[MPI_Alltoall]received");

    // --- All-to-all: Pairwise/XOR Exchange ---
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("\n--- Algorithm 1 All-to-all: Pairwise/XOR Exchange ---\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    int* xor_recvbuf = (int*)malloc(size * sizeof(int));
    if ((size > 0) && ((size & (size - 1)) == 0)) {
        for (int p = 0; p < size; p++) {
            int peer = rank ^ p;
            MPI_Sendrecv(&sendbuf[peer], 1, MPI_INT, peer, 0,
                         &xor_recvbuf[peer], 1, MPI_INT, peer, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        print_buffer_sequentially(rank, size, xor_recvbuf, "[Pairwise/XOR Exchange]received");
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) printf("SUCCESS: Pairwise/XOR exchange completed.\n");
    } else {
        if (rank == 0) {
            printf("SKIPPED: Pairwise/XOR exchange requires the number of processes to be a power of two. (Current size: %d)\n", size);
        }
    }
    free(xor_recvbuf);

    // --- All-to-all: Linear Exchange ---
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("\n--- Algorithm 2 All-to-all: Linear Exchange ---\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    int* linear_recvbuf = (int*)malloc(size * sizeof(int));
    for (int p = 0; p < size; p++) {
        int send_peer = (rank + p) % size;
        int recv_peer = (rank - p + size) % size;
        MPI_Sendrecv(&sendbuf[send_peer], 1, MPI_INT, send_peer, 0,
                     &linear_recvbuf[recv_peer], 1, MPI_INT, recv_peer, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    print_buffer_sequentially(rank, size, linear_recvbuf, "[Linear Exchange]received");
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("SUCCESS: Linear exchange completed.\n");
    free(linear_recvbuf);

    // --- All-to-all: Torsten Hoefler's Bandwidth-Optimal Algorithm ---
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("\n--- All-to-all: Bandwidth-Optimal Algorithm ---\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    int* optimal_recvbuf = (int*)malloc(size * sizeof(int));
    
    const int MAX_FACTORS = 64; 
    int factors[MAX_FACTORS];
    int num_factors = factorize(size, factors);

    int bases[MAX_FACTORS];
    for (int i = 0; i < num_factors; i++) {
        bases[i] = factors[num_factors - 1 - i];
    }
    
    int* theta_s = (int*)malloc(num_factors * sizeof(int));
    int* theta_p = (int*)malloc(num_factors * sizeof(int));
    int* theta_d = (int*)malloc(num_factors * sizeof(int));
    int* theta_rank = (int*)malloc(num_factors * sizeof(int));
    int* theta_r = (int*)malloc(num_factors * sizeof(int));

    for (int p = 0; p < size; p++) {
        to_variable_base(rank, theta_s, bases, num_factors);
        to_variable_base(p, theta_p, bases, num_factors);
        
        for(int k = 0; k < num_factors; k++){
            theta_d[k] = (theta_s[k] + theta_p[k]) % bases[k];
        }
        int send_peer = from_variable_base(theta_d, bases, num_factors);

        to_variable_base(rank, theta_rank, bases, num_factors);
        for(int k = 0; k < num_factors; k++){
            theta_r[k] = (theta_rank[k] - theta_p[k] + bases[k]) % bases[k];
        }
        int recv_peer = from_variable_base(theta_r, bases, num_factors);
        
        MPI_Sendrecv(&sendbuf[send_peer], 1, MPI_INT, send_peer, 0,
                     &optimal_recvbuf[recv_peer], 1, MPI_INT, recv_peer, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    free(theta_s);
    free(theta_p);
    free(theta_d);
    free(theta_rank);
    free(theta_r);
    
    print_buffer_sequentially(rank, size, optimal_recvbuf, "[Bandwidth-Optimal]received");
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("SUCCESS: Bandwidth-Optimal algorithm completed.\n");
    free(optimal_recvbuf);

    // --- Finalization ---
    free(sendbuf);
    free(ref_recvbuf);
    MPI_Finalize();
    return 0;
}
