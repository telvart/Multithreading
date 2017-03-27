// mpiMatrixMult_BcastScatterGather.c++ - Perform C = A * B, where:
//                     A is an MxP matrix,
//                     B is an PxK matrix,
//                     and hence C is an MxK matrix.
// This is a modified version of mpiMatrixMult that uses MPI_Bcast,
// MPI_Scatter, and MPI_Gather for optimized communication.

#include <mpi.h>
#include <math.h>
#include <iostream>
#include <stdlib.h>

static void multiply(double* A, double* B, double* C, int nRowsA, int P, int nColsB);

static double* buildMatrix(int nRows, int nCols)
{
	double* mat = new double[nRows*nCols];
	for (int r=0 ; r<nRows ; r++)
		for (int c=0 ; c<nCols ; c++)
			mat[r*nCols + c] = drand48();
	return mat;
}

static void checkAnswer(double* A, double* B, double* C, int M, int P, int K)
{
	std::cout << "Checking answer in C[" << M << "][" << K << "]: ";
	double *Ctest = new double[M*K];
	multiply(A, B, Ctest, M, P, K);
	bool ok = true;
	for (int r=0 ; r<M ; r++)
	{
		for (int c=0 ; c<K ; c++)
		{
			double diff = fabs(Ctest[r*K + c] - C[r*K + c]);
			if (diff != 0.0)
			{
				std::cerr << "Differ at [" << r << "][" << c << "]: " << diff << '\n';
				ok = false;
			}
		}
	}
	if (ok)
		std::cout << "Matrix multiplication checks!\n";
	delete [] Ctest;
}

static void do_rank_0_work(int communicatorSize, int M, int P, int K,
	int nRowsToCompute, int ArowsTag, int BTag)
{
	std::cout << "Preparing matrix A of size " << M << " x " << P << '\n';
	double* A = buildMatrix(M, P);
	std::cout << "Preparing matrix B of size " << P << " x " << K << std::endl;
	double* B = buildMatrix(P, K);

	MPI_Request sendReq; // we won't need to wait on this. We will do any
	                     // required waiting when we retrieve results.
	std::cout << "Distributing data to other processes..." << std::endl;
	// Must send entire matrix B:
	MPI_Ibcast(B, P*K, MPI_DOUBLE, 0, MPI_COMM_WORLD, &sendReq);
	// Scatter the matrix A to all processes:
	MPI_Iscatter(A, nRowsToCompute * P, MPI_DOUBLE,
	             MPI_IN_PLACE,       0, MPI_DOUBLE, // recvCount & type are ignored
	             0, MPI_COMM_WORLD, &sendReq); // rank 0 (me!) originated the scatter

	// Required data has been sent to everyone, so they should be able to start working.
	// Now I will get going on my piece:
	std::cout << "Starting to work on my part..." << std::endl;
	double* C = new double[M*K];
	multiply(A, B, C, nRowsToCompute, P, K);

	// My piece is done; let me get everyone else's
	std::cout << "Waiting for the others to send me their results..." << std::endl;
	MPI_Gather(MPI_IN_PLACE, 0, MPI_DOUBLE,
	           C, nRowsToCompute * K, MPI_DOUBLE,
	           0, MPI_COMM_WORLD);

	// Let's check to see if we got the correct answer. (This is just for testing/demonstration.)
	checkAnswer(A, B, C, M, P, K);

	// Clean up my dynamically allocated data:
	delete [] A;
	delete [] B;
	delete [] C;
}

static void do_rank_i_work(int P, int K, int nRowsToCompute, int ArowsTag, int BTag)
{
	double* Arows = new double[nRowsToCompute * P]; // Just for the assigned rows
	double* B = new double[P*K]; // Need entire matrix B
	MPI_Request dataReq[2];
	MPI_Ibcast(B, P*K, MPI_DOUBLE, 0, MPI_COMM_WORLD, &dataReq[0]);
	MPI_Iscatter(nullptr, 0, MPI_DOUBLE, // sendBuf, sendCount, sendType ignored
		Arows, nRowsToCompute * P, MPI_DOUBLE,
		0, MPI_COMM_WORLD, &dataReq[1]); // rank "0" originated the scatter

	double* Crows = new double[nRowsToCompute * K]; // Storage for just the assigned rows

	MPI_Status dataStatus[2];
	MPI_Waitall(2, dataReq, dataStatus);

	// Got the data - let's get multiplying:
	multiply(Arows, B, Crows, nRowsToCompute, P, K);

	// Now send the results back
	MPI_Gather(Crows, nRowsToCompute * K, MPI_DOUBLE,
	           nullptr, 0, MPI_DOUBLE,
	           0, MPI_COMM_WORLD);

	// Clean up my dynamically allocated data
	delete [] Arows;
	delete [] B;
	delete [] Crows;
}

static void multiply(double* A, double* B, double* C, int nRowsA, int P, int nColsB)
{
	// A is nRowsA x P; B is P x nColsB ==> C is nRowsA x nColsB
	for (int r=0 ; r<nRowsA ; r++)
	{
		for (int c=0 ; c<nColsB ; c++)
		{
			double sum = 0.0;
			for (int p=0 ; p<P ; p++)
				sum += A[r*P + p] * B[p*nColsB + c];
			C[r*nColsB + c] = sum;
		}
	}
}

static void process(int rank, int communicatorSize, int M, int P, int K)
{
	// Each process will compute M/communicatorSize rows of C.
	// It will be considered an error if communicatorSize does not
	// evenly divide M.
	if ((M % communicatorSize) != 0)
	{
		if (rank == 0)
			// Only report the error once; this is rank 0's responsibility:
			std::cerr << "communicatorSize " << communicatorSize
			          << " does not evenly divide M = " << M << '\n';
		// else: other ranks will just quietly exit
	}
	else
	{
		int nRowsToCompute = M / communicatorSize;
		int ArowsTag = 1;
		int BTag = 2;
		if (rank == 0)
			do_rank_0_work(communicatorSize, M, P, K, nRowsToCompute, ArowsTag, BTag);
		else
			// I am rank > 0 ==> do my assigned set of rows. Note that I don't
			// need to know my rank. The slices of data I have were extracted
			// according to my rank, and when I send my results back, the rank 0
			// process will correctly place my results into the global solution
			// based on my rank. But for me to do my work, I don't need to know
			// my rank.
			do_rank_i_work(P, K, nRowsToCompute, ArowsTag, BTag);
	}
}

int main (int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	int rank, communicatorSize;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &communicatorSize);

	// Get desired matrix sizes:
	if (argc < 4)
	{
		if (rank == 0) // only report error once
			std::cerr << "Specify M, P, and K on the command line.\n";
	}
	else
	{
		int M = atoi(argv[1]);
		int P = atoi(argv[2]);
		int K = atoi(argv[3]);
		process(rank, communicatorSize, M, P, K);
	}

	MPI_Finalize();
	return 0;
}
