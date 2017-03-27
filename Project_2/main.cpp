
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <mpi.h>

std::string columnLetters[] =
{"A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
 "AA","AB","AC","AD","AE","AF","AG","AH","AI","AJ","AK","AL","AM","AN","AO","AP","AQ","AR","AS","AT","AU","AV","AW","AX","AY","AZ"
 "BA","BB","BC","BD","BE","BF","BG","BH","BI","BJ","BK","BL","BM","BN","BO","BP","BQ","BR","BS","BT","BU","BV","BW","BX","BY","BZ"
 "CA","CB","CC","CD","CE","CF","CG","CH","CI","CJ","CK","CL","CM","CN","CO","CP","BQ","CR","CS","CT","CU","CV","CW","CX","CY","CZ"
 "DA","DB","DC","DD","DE","DF","DG","DH","DI","DJ","DK","DL","DM"};

std::string** getDataFromFile(std::string fileName)
{
  std::string** theData = new std::string*[501];
  for(int i=0; i<501; i++)
  {
    theData[i] = new std::string[117];
  }
  std::ifstream fileIn(fileName);
  std::string value;

  size_t pos=0;
  std::string token;
  std::string delimiter = ",";
  std::string escape = "\"";

  int curCol, curRow = 0;
  while(std::getline(fileIn, value))
  {
    curCol = 0;
    bool lastEscaped=false;
    while(value.find(delimiter) != std::string::npos)
    {
      if((pos = value.find(escape)) < value.find(delimiter))
      {
        value.erase(value.begin());
        lastEscaped = true;
        pos = value.find(escape);
        token = value.substr(0,pos);
        theData[curRow][curCol] = token;
        curCol++;
        value.erase(0,pos+escape.length()+1);
      }
      else
      {
        lastEscaped = false;
        pos = value.find(delimiter);
        token = value.substr(0,pos);
        theData[curRow][curCol] = token;
        curCol++;
        value.erase(0,pos+delimiter.length());
      }
    }
    if(!lastEscaped)
    {
      theData[curRow][curCol] = value;
    }
    curRow++;
  }
  return theData;
}

std::string** theData = getDataFromFile("500_Cities__City-level_Data__GIS_Friendly_Format_.csv");

int convertLetter(std::string letter)
{
  for(int i=0; i<117; i++)
  {
    if(columnLetters[i] == letter)
    {
      return i;
    }
  }
  return -1;
}

std::string* columnByLetter(std::string colLetter)
{
  int index;
  if((index = convertLetter(colLetter)) != -1)
  {
    std::string* s = new std::string[500];
    for(int i=1; i<501; i++)
    {
      s[i-1] = theData[i][index];
    }
    return s;
  }
  return nullptr;
}

int main(int argc, char** argv)
{

  MPI_Init(&argc, &argv);
  int rank, communicatorSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &communicatorSize);

  if(rank == 0)
  {
    std::string* col = columnByLetter("F");
    for(int i=0; i<500; i++)
    {
      std::cout<<col[i]<<"\n";
      usleep(100000);
    }
  }
  else
  {

  }

  MPI_Finalize();
}



























//
