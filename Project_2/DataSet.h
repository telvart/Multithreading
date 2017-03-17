
#ifndef DATASET_H
#define DATASET_H
#include <fstream>
#include <string>
#include <iostream>

class DataSet
{
  public:
    DataSet();
    std::string** getData();

  private:

    std::string** theData;
};


#endif
