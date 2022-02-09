LIBS= -lcudnn
all: cudnn_example

cudnn_example: example_cudnn.o
	nvcc $(LIBS) $^ -o $@ 

example_cudnn.o: example_cudnn.cpp
	nvcc -c example_cudnn.cpp

clean:
	-rm *.o cudnn_example
