CXX = nvcc
CXXFLAGS = -o

init:init.cpp
	g++ -o init init.cpp

cgsolver: main.cu
	$(CXX) $(CXXFLAGS) cgsolver main.cu

.PHONY: clean
clean:
	rm ./cgsolver
	rm ./init