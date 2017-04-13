CPP = g++ -std=c++11
C_FLAGS = -fPIC -O -c -DGL_GLEXT_PROTOTYPES -I../cryphutil -I../glslutil

LINK = g++ -fPIC

OBJS = Controller.o ModelView.o GLFWController.o BasicShape.o BasicShapeRenderer.o

libmvc.so: $(OBJS) ../lib/libcryph.so ../lib/libglsl.so
	$(LINK) -shared -o libmvc.so $(OBJS) ../lib/libcryph.so ../lib/libglsl.so
	cp libmvc.so ../lib/

../lib/libcryph.so: ../cryphutil/AffVector.h ../cryphutil/AffVector.c++ ../cryphutil/AffPoint.h ../cryphutil/AffPoint.c++
	(cd ../cryphutil; make)

../lib/libglsl.so: ../glslutil/ShaderIF.h ../glslutil/ShaderIF.c++
	(cd ../glslutil; make)

Controller.o: Controller.h Controller.c++
	$(CPP) $(C_FLAGS) Controller.c++

GLFWController.o: GLFWController.h GLFWController.c++
	$(CPP) $(C_FLAGS) GLFWController.c++

ModelView.o: ModelView.h ModelView.c++
	$(CPP) $(C_FLAGS) ModelView.c++

BasicShape.o: BasicShape.h BasicShape.c++
	$(CPP) $(C_FLAGS) BasicShape.c++

BasicShapeRenderer.o: BasicShapeRenderer.h BasicShapeRenderer.c++
	$(CPP) $(C_FLAGS) BasicShapeRenderer.c++
