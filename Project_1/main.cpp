
#include <thread>
#include <mutex>
#include <iostream>

std::mutex coutMutex;

void work(int assignment)
{
  coutMutex.lock();
  std::cout<<"I am a thread given assignment "<<assignment<<"\n";
  coutMutex.unlock();
}

int main(int argc, char** argv)
{

  std::thread** t = new std::thread*[10];
  for(int i=0; i<10; i++)
  {
    t[i] = new std::thread(work, i);
  }

  std::cout<<"Hello, Multicore and GPGPU Programming!\n";

  for(int i=0; i<10; i++)
  {
    t[i]->join();
  }
  return 0;
}
