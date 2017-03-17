
#include <iostream>
#include <fstream>
#include <string>

std::string columnLetters[] =
{"A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
 "AA","AB","AC","AD","AE","AF","AG","AH","AI","AJ","AK","AL","AM","AN","AO","AP","AQ","AR","AS","AT","AU","AV","AW","AX","AY","AZ"
 "BA","BB","BC","BD","BE","BF","BG","BH","BI","BJ","BK","BL","BM","BN","BO","BP","BQ","BR","BS","BT","BU","BV","BW","BX","BY","BZ"
 "CA","CB","CC","CD","CE","CF","CG","CH","CI","CJ","CK","CL","CM","CN","CO","CP","BQ","CR","CS","CT","CU","CV","CW","CX","CY","CZ"
 "DA","DB","DC","DD","DE","DF","DG","DH","DI","DJ","DK","DL","DM","DN","DO","DP","DQ","DR","DS","DT","DU","DV","DW","DX","DY","DZ"};


int main(int argc, char** argv)
{
  std::ifstream fileIn("500_Cities__City-level_Data__GIS_Friendly_Format_.csv");
  std::string test;
  std::getline(fileIn,test);
  std::cout<<test<<"\n";
  std::cout<<4*26;
}
