# Makefile for MLS Surface with NetCDF output
CC = gcc
CFLAGS = -O3 -march=native -fopenmp -Wall -std=c99
LDFLAGS = -lm -lnetcdf
TARGET = mls_surface
SRC = mls_surface.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o *.nc *.grd

.PHONY: all clean
