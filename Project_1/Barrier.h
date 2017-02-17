// A final "best" implementation based on creating a class whose sole purpose
// is to provide a barrier.  To use this class, place the class definition
// below into "Barrier.h", do a "#include "Barrier.h" in your program, then
// declare:
//		Barrier aBarrier; // at some sort of global scope
// Then when you are in a function or method where you require the barrier:
//		aBarrier.barrier(numExpected);

// Barrier.h - A class that implements a Barrier

#ifndef BARRIER_H
#define BARRIER_H

#include <condition_variable>
#include <mutex>

/* Usage:
	1. Create an instance of a Barrier class (called, say, "b") that
	   is accessible to, but outside the scope of any thread code that
	   needs to use it.
	2. In the thread code where barrier synchronization is to occur,
	   each thread in the "barrier group" must execute:

	   b.barrier(num); // where "num" is the number of threads in
	                   // the "barrier group"
*/

class Barrier
{
public:
	Barrier();
	virtual ~Barrier();

	void barrier(int numExpectedAtBarrier);

private:
	int barrierCounter;
	std::mutex barrierMutex;
	std::condition_variable barrierCV;
};

#endif
