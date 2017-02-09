#include "Train.h"

Train::Train(int numStops)
{
  schedule = new int[numStops];
  m_numStops = numStops;
}

Train::~Train()
{
  delete[]schedule;
}

void Train::printSchedule()
{
  for(int i=0; i<m_numStops; i++)
  {
    std::cout<<schedule[i]<<" ";
  }
}
