project1: main.o Train.o Barrier.o
	g++ -g -Wall -std=c++11 -lpthread -o project1 main.o Train.o Barrier.o

main.o: main.cpp Train.h
	g++ -g -Wall -std=c++11 -lpthread -c main.cpp

Barrier.o: Barrier.cpp Barrier.h
	g++ -g -Wall -std=c++11 -c Barrier.cpp

Train.o: Train.h Train.cpp
	g++ -g -Wall -std=c++11 -c Train.cpp

clean:
	rm *.o project1
