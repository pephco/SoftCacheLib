TARGET1 := 1_cache
TARGET2 := 2_nocache

CC = g++

# CL_DIR       = /usr
# CL_INCLUDE   = -I$(CL_DIR)/include/
# CL_LIBS      = -L$(CL_DIR)/lib/x86_64-linux-gnu/ -lOpenCL
CL_INCLUDE   = -I./OpenCL/include
CL_LIBS      = -L./OpenCL/lib -lOpenCL 

INCLUDE = -I../..

# -mcmodel=medium to avoid "relocation truncated to fit" error
# because of the large size of the data
CFLAGS1 = -std=c++11 -DCACHE_ENABLED=1 -g -mcmodel=medium
CFLAGS2 = -std=c++11 -DCACHE_ENABLED=0 -g -mcmodel=medium

SRC		= $(wildcard *.cpp) $(wildcard ./SoftCache/*.cpp) 
INCLUDE = -I./Utils -I./SoftCache

# Targets
.PHONY: t1 t2 clean all

t1:
	$(CC) $(SRC) $(CFLAGS1) $(INCLUDE) $(CL_INCLUDE) $(CL_LIBS) -o $(TARGET1)

t2:
	$(CC) $(SRC) $(CFLAGS2) $(INCLUDE) $(CL_INCLUDE) $(CL_LIBS) -o $(TARGET2)
	
all: t1 t2

clean:
	rm -rf $(TARGET1)
	rm -rf $(TARGET2)
