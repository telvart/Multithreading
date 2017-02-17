
#include <thread>
#include <mutex>
#include <iostream>
#include <fstream>
#include "Barrier.h"
#include "Train.h"

struct track
{
  int station1;
  int station2;
};

std::mutex coutMutex;
bool timetoExecute = false;
Barrier* b = new Barrier();

void trainThread(int trainNumber, Train* myTrain, int numTrains, int maxStops)
{
  int timeStep = 0;
  while(!timetoExecute); //wait until given the go signal
  while(timeStep != maxStops)//!myTrain->routeFinshed())
  {
    if(!myTrain->routeFinshed())
    {
      coutMutex.lock();
       std::cout<<"At timestep "<<timeStep<<", train " <<trainNumber
                <<" is going from "<<myTrain->getCurrentStation()
                <<" to station "<<myTrain->getNextStation()<<"\n";
      coutMutex.unlock();
      myTrain->advanceStation();
    }
    timeStep++;
    b->barrier(numTrains);
  }

}

int main(int argc, char** argv)
{

  int numTrains;
  int numStations;
  int numStops;
  int maxStops = 0;

  std::ifstream fileIn("schedules.txt");
  fileIn>>numTrains;
  fileIn>>numStations;
  Train** theTrains = new Train*[numTrains];

  for(int i=0; i<numTrains; i++)
  {
    fileIn>>numStops;
    theTrains[i] = new Train(numStops);
    for(int j=1; j<=numStops; j++)
    {
      fileIn>>theTrains[i]->m_schedule[j-1];
      if(j > maxStops)
      {
        maxStops = j;
      }
    }
  }
  fileIn.close();

  std::thread** t = new std::thread*[numTrains];
  for(int i=0; i<numTrains; i++)
  {
    t[i] = new std::thread(trainThread, i, theTrains[i], numTrains, maxStops);
  }

  timetoExecute = true;

  for(int i=0; i<numTrains; i++)
  {
    t[i]->join();
  }
  std::cout<<"Simulation complete. Exiting...\n\n";

  delete b;
  return 0;
}
