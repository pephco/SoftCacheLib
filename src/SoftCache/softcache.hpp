#ifndef SOFTCACHE_H
#define SOFTCACHE_H

#include <CL/cl.hpp>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include <time.h>
#include <assert.h>

#include <utils.hpp>

#include <iostream>
#include <fstream>
#include <iomanip>      // std::setw
#include <unordered_map>
#include <unordered_set>

// Settings
#define DEBUG           0
#define TIMING          1
#ifndef CACHE_ENABLED   // Can be overriden by compiler
#define CACHE_ENABLED   1
#endif

struct durations_t {
    unsigned long long hostToDevice;
    unsigned long long deviceToHost;
    unsigned long long kernel;
    unsigned int cacheHit;
    unsigned int cacheMiss;
    size_t bytesSaved;
    size_t bytesTotal;
    size_t bytesh2d_saved;
    size_t bytesh2d_total;
    size_t bytesd2h_saved;
    size_t bytesd2h_total;
};
#if TIMING
    #ifdef _WIN32
    // Source used for timing on windows:
    // https://learn.microsoft.com/en-us/windows/win32/sysinfo/acquiring-high-resolution-time-stamps
    #include <windows.h>
    #include <profileapi.h>
    #define START_TIMER                                                 \
        LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;    \
        LARGE_INTEGER Frequency;                                        \
        QueryPerformanceFrequency(&Frequency);                          \
        QueryPerformanceCounter(&StartingTime);

    #define STOP_TIMER(path)                                            \
        QueryPerformanceCounter(&EndingTime);                           \
        ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart; \
        ElapsedMicroseconds.QuadPart *= 1000000;                        \
        ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;             \
        path += ElapsedMicroseconds.QuadPart;
    #endif // _WIN32

    #ifdef __unix__
    #define START_TIMER         \
        clock_t start, end;     \
        start = clock();   

    #define STOP_TIMER(path)    \
        end = clock();          \
        path += end - start;
    #endif // __unix__
#else
    #define START_TIMER
    #define STOP_TIMER
#endif // TIMING

enum Flag {
    CPU,
    GPU,
    BOTH
};

struct CacheLine {
    Flag flag;
    int age;
    size_t size;
    void *tag;
    cl_mem deviceAddress;
};

enum Organisation {
    DIRECT_MAPPING,
    SET_ASSOCIATIVE,
    FULLY_ASSOCIATIVE
};

enum ReplacementPolicy {
    LRU,
    FIFO,
    RANDOM,
    SMALLEST
};

class Cache {
    private:
        std::unordered_map<cl_kernel, std::unordered_set<const void*>> kernelArguments; // < pointer to kernel, set of pointers to CL buffers >

        int nrOfSets;
        int nrOfLines;
        int nrOfLinesPerSet; // One of these is redundant

        enum Organisation organisation;
        enum ReplacementPolicy replacementPolicy;
        CacheLine *lines;
        vector<unsigned int> FIFO_index;

        cl_command_queue cache_command_queue;

        std::vector<unsigned int> lockedLines;

        bool isPrime(int n);
        
        int getTableSize(int n);
        int getSetIndex(const void *tag);

        CacheLine addToCache(const void *tag, size_t size, cl_mem deviceAddress, Flag flag, int idx = -1);
        CacheLine* getCacheLine(const void *tag);
        void replaceCacheLine(const void *tag, size_t size, cl_mem deviceAddress);

        // Helper functions for replacement policies
        int getRandomIndex(int setIndex);
        int getOldestIndex(int setIndex, bool increaseAge = false);
        int getSmallestDataLine(int setIndex);

        void initialise(Organisation organisation, ReplacementPolicy replacementPolicy, int cacheSize, int linesPerSet, bool write_back = false);

        durations_t duration; 
        

    public:
        Cache(Organisation organisation, ReplacementPolicy replacementPolicy, int cacheSize, int linesPerSet = 1, bool write_back = false);
        Cache(int argc, char** argv);
        ~Cache();

        cl_mem createBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret);
        cl_int enqueueWriteBuffer(
            cl_command_queue command_queue, 
            cl_mem *buffer, cl_bool blocking_write, 
            size_t offset, 
            size_t cb, 
            const void *ptr, 
            cl_uint num_events_in_wait_list, 
            const cl_event *event_wait_list, 
            cl_event *event
        );
        cl_int enqueueReadBuffer(
            cl_command_queue command_queue,
            cl_mem buffer, cl_bool blocking_read,
            size_t offset,
            size_t cb,
            void *ptr,
            cl_uint num_events_in_wait_list,
            const cl_event *event_wait_list,
            cl_event *event
        );

        cl_int setKernelArg(cl_kernel kernel, cl_uint index, size_t size, const void * value);

        cl_int enqueueNDRangeKernel(
            cl_command_queue command_queue,
            cl_kernel kernel,
            cl_uint work_dim,
            const size_t *global_work_offset,
            const size_t *global_work_size,
            const size_t *local_work_size,
            cl_uint num_events_in_wait_list,
            const cl_event *event_wait_list,
            cl_event *event
        );
        cl_int writeBack();                 // Write everything back to host
        cl_int writeBack(void *host_ptr);   // Write specific buffer back to host

        void setDirtyFlag(const void *tag, Flag flag = CPU);

        void printCache();
        void printTimeProfile();
        void writeTimeProfileToFile(vector<string> other_info) ;
        void resetCache();
        void resetTimers();

        bool write_back;
        unsigned int buffers;
};

#endif // SOFTCACHE_H