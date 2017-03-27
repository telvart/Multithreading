
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <stdlib.h>
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

double* columnByLetter(std::string colLetter)
{
  int index;
  if((index = convertLetter(colLetter)) != -1)
  {
    double* s = new double[500];
    for(int i=1; i<501; i++)
    {
      s[i-1] = atof(theData[i][index].c_str());
    }
    return s;
  }
  return nullptr;
}

std::string curMode, curOp, column;


bool validateParameters(int argc, char** argv, int processes)
{
  curMode = argv[1];
  curOp = argv[2];
  if(curMode == "sr")
  {
    if(500 % processes != 0)
    {
      std::cout<<"The number of processes did not divide evenly into 500!\n";
      return false;
    }
    column = argv[3];
  }
  else if (curMode == "bg")
  {
    int cols = 0;
    for(int i=3; i<argc; i++) {cols++;}

    if(cols != processes)
    {
      std::cout<<"The number of columns did not match the number of processes!\n";
      return false;
    }
  }
  else
  {
    std::cout<<"Invalid scheme selected!\n";
    return false;
  }
  return true;
}



int main(int argc, char** argv)
{

  MPI_Init(&argc, &argv);

  int rank, communicatorSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &communicatorSize);
  if(!validateParameters(argc, argv, communicatorSize)) {return 0;}

  if(curMode == "sr")
  {

    if (rank == 0)
    {
      double* workingColumn = columnByLetter(argv[3]);
      int numMessages = 500/communicatorSize;
      MPI_Scatter(workingColumn, numMessages, MPI_DOUBLE,
         MPI_IN_PLACE, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    }
    else
    {

    }
  }
  else
  {
    if(rank == 0)
    {

    }
    else
    {

    }
  }

  MPI_Finalize();
}



























//
