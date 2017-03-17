project2: main.o DataSet.o
	g++ -g -Wall -std=c++11 -o project2 main.o DataSet.o
main.o: main.cpp
	g++ -g -Wall -std=c++11 -c main.cpp
DataSet.o: DataSet.cpp DataSet.h
	g++ -g -Wall -std=c++11 -c DataSet.cpp
clean:
	rm *.o project2
