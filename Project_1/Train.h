

#ifndef TRAIN_H
#define TRAIN_H
#include <iostream>
class Train
{
public:

  Train(int numStops);
  ~Train();
  void printSchedule();
  bool routeFinshed();
  void advanceStation();
  int getCurrentStation();
  int getNextStation();
  int getLastStation();

  int* m_schedule;

private:
  int m_numStops;
  int m_scheduleIndex;
  bool m_amDone;


};

#endif
