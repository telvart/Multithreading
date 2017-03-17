#include "DataSet.h"

DataSet::DataSet()
{
  theData = new std::string*[501];
  for(int i=0; i<501; i++)
  {
    theData[i] = new std::string[117];
  }



  std::ifstream fileIn("500_Cities__City-level_Data__GIS_Friendly_Format_.csv");
  std::string value;
  std::getline (fileIn, value);
  size_t pos=0;
  std::string token;
  std::string delimiter = ",";
  int curCol = 0;
  while((pos = value.find(delimiter)) != std::string::npos)
  {
    token = value.substr(0,pos);
    theData[0][curCol] = token;
    curCol++;
    value.erase(0,pos+delimiter.length());
  }
  theData[0][curCol]=value;

  std::cout<<"VALUES IN ARRAY:\n";
  for(int i=0; i<117; i++)
  {
    std::cout<<theData[0][i]<<"\n";
  }

  int currentRow = 1;
  curCol = 0;
  while(std::getline(fileIn, value))
  {
    std::cout<<value<<"\n";
  }



//  std::cout<<value<<"\n";

  //std::cout << value;
}

std::string** DataSet::getData()
{
  return theData;
}
