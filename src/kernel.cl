// OpenCL Kernel
__kernel void
matrixMul(__global float* A, 
          __global float* B, 
          __global float* C, 
          int widthA, int widthB)
{
  
   int tx = get_global_id(0); 
   int ty = get_global_id(1);
 
   // value stores the element that is 
   // computed by the thread
   float value = 0;
   for (int k = 0; k < widthA; ++k)
   {
      float elementA = A[ty * widthA + k];
      float elementB = B[k * widthB + tx];
      value += elementA * elementB;
   }
 
   // Write the matrix to device memory each 
   // thread writes one element
   C[ty * widthA + tx] = value;
}