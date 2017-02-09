
#include <thread>
#include <mutex>
#include <iostream>
#include <fstream>
#include <string>
#include "Train.h"

std::mutex coutMutex;

void work(int trainNumber)
{
  coutMutex.lock();
  std::cout<<"Train "<<trainNumber<<": \n";
  // myTrain->printSchedule();
  // std::cout<<"\n";
  coutMutex.unlock();
}

int main(int argc, char** argv)
{

  int numTrains;
  int numStations;
  int numStops;
  //
  std::ifstream fileIn("schedules.txt");
  fileIn>>numTrains;
  fileIn>>numStations;
  // Train** theTrains = new Train*[numTrains];
  //
  // for(int i=0; i<numTrains; i++)
  // {
  //   fileIn>>numStops;
  //   theTrains[i] = new Train(numStops);
  //   for(int j=0; j<numStops; j++)
  //   {
  //     fileIn>>theTrains[i]->schedule[j];
  //   }
  // }

  // for(int i=0; i<numTrains; i++)
  // {
  //   std::cout<<"Train "<<i<<": ";
  //   theTrains[i]->printSchedule();
  //   std::cout<<"\n";
  // }

  std::thread** t = new std::thread*[numTrains];
  for(int i=0; i<numTrains; i++)
  {
    t[i] = new std::thread(work, i);
  }

  //std::cout<<"Hello, Multicore and GPGPU Programming!\n";

  for(int i=0; i<10; i++)
  {
    t[i]->join();
  }
  return 0;
}
