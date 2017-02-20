
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

std::mutex trainMutex;
bool timetoExecute = false;
Barrier* b = new Barrier();
track** tracks;
int numTrains;
int numStations;
int numDone;
char chars[] = {'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'};

bool compareTracks(track* t1, track* t2)
{
  return (t1->station1 + t1->station2) == (t2->station1 + t2->station2);
}

void clearTracks()
{
  for(int i=0; i<numTrains; i++)
  {
    if(tracks[i] != nullptr)
    {
      delete tracks[i];
      tracks[i] = nullptr;
    }
  }
}

bool safetoAdvance(track* t, int trainNumber)
{
  for(int i=0; i<numTrains; i++)
  {
    if(tracks[i] != nullptr && i != trainNumber)
    {
      if(compareTracks(tracks[i], t))
      {
        return false;
      }
    }
  }
  return true;
}

void trainThread(int trainNumber, Train* myTrain)
{
  int timeStep = 0;
  while(!timetoExecute);
  while(!myTrain->routeFinshed())
  {
     if(!myTrain->routeFinshed())
     {
      track* t = new track;
      t->station1=myTrain->getCurrentStation();
      t->station2=myTrain->getNextStation();
      trainMutex.lock();
      tracks[trainNumber]=t;
      if(safetoAdvance(t,trainNumber))
      {
        std::cout<<"At time step: "<<timeStep<<" train " <<chars[trainNumber]
                 <<" is going from station "<<myTrain->getCurrentStation()
                 <<" to station "<<myTrain->getNextStation()<<"\n";
                 myTrain->advanceStation();
      }
      else
      {
        std::cout<<"At time step: "<<timeStep<<" train " <<chars[trainNumber]
                 <<" must stay at station "<<myTrain->getCurrentStation()<<"\n";
      }
    }

    trainMutex.unlock();
    timeStep++;
    b->barrier(numTrains);
    clearTracks();
    b->barrier(numTrains);
  }

  trainMutex.lock();
  numDone++;
  trainMutex.unlock();
  // at this point, the thread is finished and will wait for the others
  if(numDone == numTrains)
  {
    b->releaseBarrier();
  }
  while(numDone < numTrains)
  {
    b->barrier(numTrains);
  }
}

int main(int argc, char** argv)
{
  if(argc < 2)
  {
    std::cout<<"Usage: ./project1 <fileName>\n";
    return 0;
  }

  std::ifstream fileIn(argv[1]);

  if(!fileIn.is_open())
  {
    std::cout<<"The file you entered was unable to be opened. Please try again\n";
    return 0;
  }

  int numStops;
  fileIn>>numTrains;
  fileIn>>numStations;
  Train** theTrains = new Train*[numTrains];
  tracks = new track*[numTrains];

  for(int i=0; i<numTrains; i++)
  {
    tracks[i]=nullptr;
  }

  for(int i=0; i<numTrains; i++)
  {
    fileIn>>numStops;
    theTrains[i] = new Train(numStops);
    for(int j=1; j<=numStops; j++)
    {
      fileIn>>theTrains[i]->m_schedule[j-1];
    }
  }
  fileIn.close();

  std::thread* t = new std::thread[numTrains];
  for(int i=0; i<numTrains; i++)
  {
    t[i] = std::thread(trainThread, i, theTrains[i]);
  }

  std::cout<<"Starting simulation...\n";
  timetoExecute = true;

  for(int i=0; i<numTrains; i++)
  {
    t[i].join();
  }

  std::cout<<"Simulation complete. Exiting...\n\n";

  for(int i=0; i<numTrains; i++)
  {
    delete theTrains[i];
  }

  delete b;
  delete[] tracks;
  delete[] t;
  delete[] theTrains;
  return 0;
}
