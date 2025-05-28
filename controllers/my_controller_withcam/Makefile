WEBOTS_HOME = /usr/local/webots

CXX = g++
CXXFLAGS = -O2 -std=c++11 -I$(WEBOTS_HOME)/include/controller/c
LDFLAGS = -L$(WEBOTS_HOME)/lib/controller -lController

TARGET = my_controller_withcam
SRC = my_controller_withcam.cpp

all:
	$(CXX) $(CXXFLAGS) $(SRC) $(LDFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET)

