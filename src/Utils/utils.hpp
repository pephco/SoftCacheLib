#ifndef UTILS_HPP
#define UTILS_HPP

#include <CL/cl.h>
#include <iostream>
#include <math.h>

#include <algorithm>
#include <vector>
#include <time.h>

using namespace std;

#define OCL_CHECK(error,call)                                       \
    call;                                                           \
    if (error != CL_SUCCESS) {                                      \
      printf("%s:%d Error calling " #call ", error code is: %d\n",  \
              __FILE__,__LINE__, error);                            \
      exit(EXIT_FAILURE);                                           \
    }           

inline void printMatrix(int *arr, int n, int m) {
  int i, j;
  for (i = 0 ; i < n ; ++i ) {
    for (j = 0 ; j < m ; ++j ) {
        printf("%d ", arr[i*n+j]);
    }
    printf("\n" );
  }
  printf("\n" );
}

inline void printMatrix(float *arr, int n, int m) {
  int i, j;
  for (i = 0 ; i < n ; ++i ) {
    for (j = 0 ; j < m ; ++j ) {
        printf("%.2f\t", arr[i*n+j]);
    }
    printf("\n" );
  }
  printf("\n" );
}



inline string getErrorString(cl_int error)
{
switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}

inline void matrixMul(float * A, float * B, float * C, int w, int h) 
{
    for(int i = 0; i < w; ++i) 
    {
        for(int j = 0; j < h; j++) 
        {
            C[i*h + j] = 0;
            for(int k = 0; k < w; ++k) 
            {
                C[i*h + j] += A[i*h + k] * B[k*h + j];
            }
        }
    }
}

inline bool compareMatrices(float * A, float * B, unsigned int size) 
{
    for(unsigned int i = 0; i < size; ++i) 
    {
        if (fabs(A[i] - B[i]) > 0.1)
        {
            printf("Error at index %d: %f != %f \n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

inline long long probe_event_time(cl_event event, cl_command_queue command_queue) {
    cl_int error=0;
    cl_ulong eventStart,eventEnd;

    clWaitForEvents(1, &event);
    clFinish(command_queue);
    
    error = clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,
                                    sizeof(cl_ulong),&eventStart,NULL);
    if (error != CL_SUCCESS) {
        printf("ERROR (%d) in event start profiling.\n", error);
        return 0;
    }
    error = clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,
                                    sizeof(cl_ulong),&eventEnd,NULL);
    if (error != CL_SUCCESS) {
        printf("ERROR (%d) in event end profiling.\n", error);
        return 0;
    }

    // Convert from nanoseconds to microseconds
    return (long long)((eventEnd - eventStart) / 1000);
}



// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
inline const std::string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d %X", &tstruct);

    return buf;
}

class InputParser{
    private:
        std::vector <std::string> tokens;

    public:
        InputParser (int &argc, char **argv){
            for (int i=1; i < argc; ++i)
                this->tokens.push_back(std::string(argv[i]));
        }
        /// @author iain
        const std::string& getCmdOption(const std::string &option) const{
            std::vector<std::string>::const_iterator itr;
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){
                return *itr;
            }
            static const std::string empty_string("");
            return empty_string;
        }
        /// @author iain
        bool cmdOptionExists(const std::string &option) const{
            return std::find(this->tokens.begin(), this->tokens.end(), option)
                   != this->tokens.end();
        }    
};

template<typename CharT>
class DecimalSeparator : public std::numpunct<CharT>
{
public:
    DecimalSeparator(CharT Separator)
    : m_Separator(Separator)
    {}

protected:
    CharT do_decimal_point()const
    {
        return m_Separator;
    }

private:
    CharT m_Separator;
};

#endif // UTILS_HPP