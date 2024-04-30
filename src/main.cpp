#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <CL/cl.h>

#include <utils.hpp>
#include <softcache.hpp>

#include <fstream>
#include <sstream>

#include <map>
#include <iomanip>      // std::setw

#include <algorithm>
#include <random>

#define MEASURE_TIME 1

/* =====================================================
 * Remap native opencl functions to SoftCache functions
 */ 
#define clCreateBuffer                                  cache->createBuffer
#define clEnqueueWriteBuffer(a,b,c,d,e,f,g, h, i)       cache->enqueueWriteBuffer(a, &b, c, d, e, f, g, h, i)
#define clEnqueueReadBuffer                             cache->enqueueReadBuffer
#define clSetKernelArg                                  cache->setKernelArg
#define clEnqueueNDRangeKernel                          cache->enqueueNDRangeKernel
// Enable profiling of commands
#if TIMING
#define clCreateCommandQueue(a, b, c, d)                clCreateCommandQueue(a, b, CL_QUEUE_PROFILING_ENABLE, d)
#endif
// Disable freeing of memory because the destructor of SoftCache will do this
#if CACHE_ENABLED
#define clReleaseMemObject                              0 && clReleaseMemObject 
#endif
Cache *cache = nullptr;
/* ===================================================== */ 
using namespace std;

// OpenCL related declarations
cl_int err;
cl_platform_id platform;
cl_device_id device;
cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
cl_context ctx;
cl_program program;
cl_command_queue queue;
cl_event event = NULL;
cl_kernel matrix_mul_kernel;



char * loadKernelFile( const char * tempchar ) 
{
    int sourcesize = 1024*1024;
	char * source = (char *)calloc(sourcesize, sizeof(char));

	if(!source) 
    { 
        printf("ERROR: calloc(%d) failed\n", sourcesize); 
    }

    FILE * fp = fopen(tempchar, "rb");
	if(!fp) 
    { 
        printf("ERROR: unable to open '%s'\n", tempchar); 
    }

	fread(source + strlen(source), sourcesize, 1, fp);
	fclose(fp);

    return source;
}

void initialiseOpenCL()
{
        /* Setup OpenCL environment. */
    err = clGetPlatformIDs( 1, &platform, NULL );
    err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );

    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
    queue = clCreateCommandQueue( ctx, device, 0, &err );

    char *source = NULL;
    source = loadKernelFile("./kernel.cl");
    program = clCreateProgramWithSource(ctx, 1, (const char **) &source, NULL, &err);
    // program = cl_compileProgram( (char *) "src/kernel.cl", NULL);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    matrix_mul_kernel = clCreateKernel(program, "matrixMul", &err);

#if 0 // Kernel compile output    
    size_t ret_val_size;
    printf("Device: %p\n",device);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, 
        NULL, &ret_val_size);

    char *build_log = (char*)calloc(ret_val_size+1, sizeof(char));

    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
        ret_val_size+1, build_log, NULL);
    build_log[ret_val_size] = '\0';

    printf("Build log:\n %s...\n", build_log);
#endif
}

void matrixMulGPU(float * A, float * B, float * C, unsigned int w, unsigned int h)
{

    const unsigned int N = w * h; // Matrix vector size
    // cl_mem A_buffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(*A), A, &err);
    // cl_mem B_buffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(*B), B, &err);
    cl_mem A_buffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * sizeof(*A), NULL, &err);
    cl_mem B_buffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * sizeof(*B), NULL, &err);
    cl_mem C_buffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * sizeof(*C), NULL, &err);


    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }

    // cout << "A_buffer: " << A_buffer << endl;
    // cout << "B_buffer: " << B_buffer << endl;
    // cout << "C_buffer: " << C_buffer << endl;

    err  = clEnqueueWriteBuffer(queue, A_buffer, CL_TRUE, 0, N * sizeof(*A), A, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, B_buffer, CL_TRUE, 0, N * sizeof(*B), B, 0, NULL, NULL);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write buffer!\n");
        exit(1);
    }

    err  = clSetKernelArg(matrix_mul_kernel, 0, sizeof(cl_mem), (void *)&A_buffer);
    err |= clSetKernelArg(matrix_mul_kernel, 1, sizeof(cl_mem), (void *)&B_buffer);
    err |= clSetKernelArg(matrix_mul_kernel, 2, sizeof(cl_mem), (void *)&C_buffer);
    err |= clSetKernelArg(matrix_mul_kernel, 3, sizeof(int), &w);
    err |= clSetKernelArg(matrix_mul_kernel, 4, sizeof(int), &h);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %s\n", getErrorString(err));
        exit(1);
    }

    size_t global_work_size[2] = {w, h};
    size_t local_work_size[2] = {4, 4};

    err = clEnqueueNDRangeKernel(queue, matrix_mul_kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to execute kernel! %s\n", getErrorString(err));
        exit(1);
    }

    err = clEnqueueReadBuffer(queue, C_buffer, CL_TRUE, 0, N*sizeof(*C), C, 0, NULL, NULL);

    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %s\n", getErrorString(err));
        exit(1);
    }

    clReleaseMemObject(A_buffer);
    clReleaseMemObject(B_buffer);
    clReleaseMemObject(C_buffer);
}

int runTest(int argc, char** argv) 
{
    initialiseOpenCL();    
    
    const unsigned int w = 1024;
    const unsigned int h = 1024;
    const unsigned int N = w * h; // Matrix vector size
    
    float *A, *B, *C, *D, *E;        
    A = (float*) malloc(N * sizeof(*A));
    B = (float*) malloc(N * sizeof(*B));
    C = (float*) malloc(N * sizeof(*C));
    D = (float*) malloc(N * sizeof(*D));
    E = (float*) malloc(N * sizeof(*E));

    for (unsigned int i = 0; i < N; i++)
    {
        A[i] = rand() % 10 / 100.0;
        B[i] = rand() % 10 / 100.0;
    }

    // Matrix multiplication test:
    // C = A * B
    // D = B * C
    // E = C * D
    float * CPU_A, *CPU_B, *CPU_C, *CPU_D, *CPU_E;
    CPU_A = A;
    CPU_B = B;
    CPU_C = (float *) malloc(N * sizeof(*CPU_C));
    CPU_D = (float *) malloc(N * sizeof(*CPU_D));
    CPU_E = (float *) malloc(N * sizeof(*CPU_E));

    matrixMulGPU(A, B, C, w, h);
    matrixMul(CPU_A, CPU_B, CPU_C, w, h);    

    matrixMulGPU(B, C, D, w, h);
    matrixMul(CPU_B, CPU_C, CPU_D, w, h);

    matrixMulGPU(C, D, E, w, h);
    matrixMul(CPU_C, CPU_D, CPU_E, w, h);
    
    if (cache->write_back)
    {
        // Needed to retrieve the final results when write back is enabled. 
        // Does nothing if write back is disabled.
        cache->writeBack(E);
        cout << "C * D " << (compareMatrices(E, CPU_E, N) ? "correct" : "NOT correct") << endl;
    }
    else
    {
        cout << "A * B " << (compareMatrices(C, CPU_C, N) ? "correct" : "NOT correct") << endl;
        cout << "B * C " << (compareMatrices(D, CPU_D, N) ? "correct" : "NOT correct") << endl;
        cout << "C * D " << (compareMatrices(E, CPU_E, N) ? "correct" : "NOT correct") << endl;
    }

    free(CPU_C);
    free(CPU_D);
    free(CPU_E);

    /* Release OpenCL memory objects. */
    free(A);
    free(B);
    free(C);
    free(D);
    free(E);
    clReleaseCommandQueue( queue );
    clReleaseContext( ctx );

    return 0;
}


int main(int argc, char** argv) 
{
    if (argc > 2)
    {
        cache = new Cache(argc, argv);
    } 
    else 
    {
        cache = new Cache(FULLY_ASSOCIATIVE, FIFO, 12, 1, true);
    }
    

    // Run the main loop of the benchmark
    runTest(argc, argv);

    cache->printCache();
    cache->printTimeProfile();

    delete cache;

    // Wait for key input
    cout << "Program finished, press a key to exit..." << endl;

    return 0;
}
