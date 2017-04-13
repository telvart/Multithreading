CPP = g++ -std=c++11
INC = -I../cryphutil -I../glslutil -I../mvcutil -I../contourlines_GPU
C_FLAGS = -fPIC -g -c -DGL_GLEXT_PROTOTYPES $(INC)

LINK = g++ -fPIC -g
LOCAL_UTIL_LIBRARIES = ../lib/libcryph.so ../lib/libglsl.so ../lib/libmvc.so
GL_LIB_LOC = -L/usr/lib64/nvidia
OGL_LIBRARIES = $(GL_LIB_LOC) -lglfw -lGLU -lGL

OBJS = ContourLineController.o GL_LINES_Renderer.o

contourlines_GL.so: $(OBJS) $(LOCAL_UTIL_LIBRARIES)
	$(LINK) -shared -o contourlines_GL.so $(OBJS) $(LOCAL_UTIL_LIBRARIES) $(OGL_LIBRARIES)

../lib/libcryph.so: ../cryphutil/AffPoint.h ../cryphutil/AffPoint.c++
	(cd ../cryphutil; make)

../lib/libglsl.so: ../glslutil/ShaderIF.h ../glslutil/ShaderIF.c++
	(cd ../glslutil; make)

../lib/libmvc.so: ../mvcutil/Controller.h ../mvcutil/Controller.c++ ../mvcutil/ModelView.h ../mvcutil/ModelView.c++
	(cd ../mvcutil; make)

ContourLineController.o: ContourLineController.h ContourLineController.c++
	$(CPP) $(C_FLAGS) ContourLineController.c++

GL_LINES_Renderer.o: GL_LINES_Renderer.h GL_LINES_Renderer.c++
	$(CPP) $(C_FLAGS) GL_LINES_Renderer.c++
