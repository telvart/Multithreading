CPP = g++ -std=c++11
C_FLAGS = -fPIC -O -c

LINK = g++ -fPIC

OBJS = AffPoint.o AffVector.o Matrix3x3.o Matrix4x4.o ProjPoint.o

libcryph.so: $(OBJS)
	$(LINK) -shared -o libcryph.so $(OBJS)
	cp libcryph.so ../lib/

AffPoint.o: AffPoint.h AffPoint.c++
	$(CPP) $(C_FLAGS) AffPoint.c++

AffVector.o: AffVector.h AffVector.c++
	$(CPP) $(C_FLAGS) AffVector.c++

Matrix3x3.o: Matrix3x3.h Matrix3x3.c++
	$(CPP) $(C_FLAGS) Matrix3x3.c++

Matrix4x4.o: Matrix4x4.h Matrix4x4.c++
	$(CPP) $(C_FLAGS) Matrix4x4.c++

ProjPoint.o: ProjPoint.h ProjPoint.c++
	$(CPP) $(C_FLAGS) ProjPoint.c++
