#include "Barrier.h"


Barrier::Barrier()
  : barrierCounter(0)
{

}
Barrier::~Barrier()
{

}

void Barrier::barrier(int numExpectedAtBarrier)
{
	std::unique_lock<std::mutex> ulbm(barrierMutex);

	barrierCounter++;
	if (barrierCounter != numExpectedAtBarrier)
		barrierCV.wait(ulbm);
	else
	{
		barrierCounter = 0;
		barrierCV.notify_all();
	}
}
