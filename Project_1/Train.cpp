#include "Train.h"

Train::Train(int numStops)
{
  m_schedule = new int[numStops];
  m_numStops = numStops;
  m_scheduleIndex = 0;
  m_amDone=false;
}

Train::~Train()
{
  delete[] m_schedule;
}

void Train::printSchedule()
{
  for(int i=0; i<m_numStops; i++)
  {
    std::cout<<m_schedule[i]<<" ";
  }
}

bool Train::routeFinshed()
{
  return m_amDone;//m_scheduleIndex >= m_numStops;
}

int Train::getCurrentStation()
{
  if(!routeFinshed())
  {
    return m_schedule[m_scheduleIndex];
  }
  return -1;

}

void Train::advanceStation()
{
  m_scheduleIndex++;
  if(m_scheduleIndex == m_numStops - 1)
  {
    m_amDone = true;
  }
}

int Train::getLastStation()
{
  return m_schedule[m_numStops-1];
}

int Train::getNextStation()
{
  if(routeFinshed() || m_scheduleIndex >= m_numStops - 1)
  {
    return -1;
  }
  return m_schedule[m_scheduleIndex+1];
}
