#include "mex.h"
#include "../src/test_gpu_access.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // input validation
    if (nrhs > 0 || nlhs != 1)
        mexErrMsgIdAndTxt("QuaDRiGa:intersect_mesh_mex", "Wrong number of input/output arguments.");

    // Create ouput array
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    double *cc = (double *)mxGetData(plhs[0]);

    cc[0] = 0; // Initialize output to 0
    test_gpu_access_CUDA(cc);
}