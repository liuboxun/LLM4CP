#include "mex.h"
#include "../src/qd_mesh_functions.h"
#include "../src/ray_mesh_intersect.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // input validation
    if (nlhs != 5 || nrhs < 3 )
        mexErrMsgIdAndTxt("QuaDRiGa:ray_mesh_intersect", "Wrong number of input/output arguments.");
    if (!mxIsSingle(prhs[0]) || !mxIsSingle(prhs[1]) || !mxIsSingle(prhs[2]))
        mexErrMsgIdAndTxt("QuaDRiGa:ray_mesh_intersect", "Inputs 'orig', 'dest', and 'mesh' must be provided in 'single' precision.");
    if (mxGetNumberOfElements(prhs[0]) != mxGetNumberOfElements(prhs[1]))
        mexErrMsgIdAndTxt("QuaDRiGa:ray_mesh_intersect", "Inputs 'orig' and 'dest' must have the same size.");
    if (mxGetN(prhs[0]) != 3 || mxGetN(prhs[1]) != 3)
        mexErrMsgIdAndTxt("QuaDRiGa:ray_mesh_intersect", "Inputs 'orig' and 'dest' must have 3 columns.");
    if (mxGetN(prhs[2]) != 9)
        mexErrMsgIdAndTxt("QuaDRiGa:ray_mesh_intersect", "Input 'mesh' must have 9 columns.");

    // Read number of hits to return in Wout
    unsigned int *no_hit_W;
    if (nrhs < 4)
    {
        no_hit_W = new unsigned int[0];
        no_hit_W[0] = 1;
    }
    else if (mxGetNumberOfElements(prhs[3]) != 1)
        mexErrMsgIdAndTxt("QuaDRiGa:ray_mesh_intersect", "Input 'no_hit_W' must be scalar.");
    else if (mxIsDouble(prhs[3]))
    {
        double *tmp = (double *)mxGetData(prhs[3]);
        no_hit_W = new unsigned int[0];
        no_hit_W[0] = (unsigned int)tmp[0];
    }
    else if (mxIsClass(prhs[3], "uint32") || mxIsClass(prhs[3], "int32"))
        no_hit_W = (unsigned int *)mxGetData(prhs[3]);
    else
        mexErrMsgIdAndTxt("QuaDRiGa:ray_mesh_intersect", "Input 'no_hit_W' must be either of type 'double', 'int32' or 'uint32'.");

    // Read verbose variable
    int *verbose = new int[0];
    if (nrhs < 5)
        verbose[0] = 1;
    else if (mxGetNumberOfElements(prhs[4]) != 1)
        mexErrMsgIdAndTxt("QuaDRiGa:ray_mesh_intersect", "Input 'verbose' must be scalar.");
    else if (mxIsDouble(prhs[4]))
    {
        double *tmp = (double *)mxGetData(prhs[4]);
        verbose[0] = (int)tmp[0];
    }
    else if (mxIsClass(prhs[4], "logical"))
    {
        bool *tmp = (bool *)mxGetData(prhs[4]);
        verbose[0] = (tmp[0]) ? 1 : 0;
    }
    else
        mexErrMsgIdAndTxt("QuaDRiGa:ray_mesh_intersect", "Input 'verbose' must be either of type 'double' or 'logical'.");

    // Create ouput arrays
    mwSize no_ray = mxGetM(prhs[0]);                                              // Number of rays
    mwSize no_mesh = mxGetM(prhs[2]);                                             // Number of mesh elements
    plhs[0] = mxCreateNumericMatrix(no_ray, 1, mxUINT32_CLASS, mxREAL);           // Number of interactions per ray
    plhs[1] = mxCreateNumericMatrix(no_ray, 3, mxSINGLE_CLASS, mxREAL);           // FBS
    plhs[2] = mxCreateNumericMatrix(no_ray, 3, mxSINGLE_CLASS, mxREAL);           // LBS
    plhs[3] = mxCreateNumericMatrix(no_ray, 1, mxUINT32_CLASS, mxREAL);           // Index of first hit elment
    plhs[4] = mxCreateNumericMatrix(no_hit_W[0], no_ray, mxSINGLE_CLASS, mxREAL); // Interactions

    // Get pointers to data
    Matrix orig, dest, mesh, fbs, lbs, Wout;

    orig.height = (unsigned int)no_ray;
    orig.width = 3;
    orig.elements = (float *)mxGetData(prhs[0]);

    dest.height = (unsigned int)no_ray;
    dest.width = 3;
    dest.elements = (float *)mxGetData(prhs[1]);

    mesh.height = (unsigned int)no_mesh;
    mesh.width = 9;
    mesh.elements = (float *)mxGetData(prhs[2]);

    fbs.height = (unsigned int)no_ray;
    fbs.width = 3;
    fbs.elements = (float *)mxGetData(plhs[1]);

    lbs.height = (unsigned int)no_ray;
    lbs.width = 3;
    lbs.elements = (float *)mxGetData(plhs[2]);

    Wout.height = no_hit_W[0];
    Wout.width = (unsigned int)no_ray;
    Wout.elements = (float *)mxGetData(plhs[4]);

    unsigned int *hit, *iFBS;
    hit = (unsigned int *)mxGetData(plhs[0]);
    iFBS = (unsigned int *)mxGetData(plhs[3]);

    // Call CUDA-based intersect function
    ray_mesh_intersect_CUDA(orig, dest, mesh, fbs, lbs, hit, iFBS, Wout, verbose[0]);
}
