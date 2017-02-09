

#ifndef TRAIN_H
#define TRAIN_H
#include <iostream>
class Train
{
public:

  Train(int numStops);
  ~Train();
  void printSchedule();
  int* schedule;

private:
  int m_numStops;


};

#endif
