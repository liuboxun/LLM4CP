#include "mex.h"
#include "../src/qd_mesh_functions.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate 'mesh'
    if (nlhs != 2 || nrhs < 1)
        mexErrMsgIdAndTxt("QuaDRiGa:mesh_reduce_size", "Wrong number of input/output arguments.");
    if (!mxIsSingle(prhs[0]))
        mexErrMsgIdAndTxt("QuaDRiGa:mesh_reduce_size", "Input 'mesh' must be provided in 'single' precision.");
    if (mxGetN(prhs[0]) != 9)
        mexErrMsgIdAndTxt("QuaDRiGa:mesh_reduce_size", "Input 'mesh' must have 9 columns.");

    mwSize no_ray = 0;                // Number of rays
    mwSize no_mesh = mxGetM(prhs[0]); // Number of mesh elements

    Matrix orig, dest, mesh, meshE;

    mesh.height = (unsigned int)no_mesh;
    mesh.width = 9;
    mesh.elements = (float *)mxGetData(prhs[0]);

    // Validate 'orig'
    bool reduce_mesh_size = true;
    if (nrhs < 2 || mxIsEmpty(prhs[1]))
        reduce_mesh_size = false;
    else if (!mxIsSingle(prhs[1]))
        mexErrMsgIdAndTxt("QuaDRiGa:mesh_reduce_size", "Input 'orig' must be provided in 'single' precision.");
    else if (mxGetN(prhs[1]) != 3)
        mexErrMsgIdAndTxt("QuaDRiGa:mesh_reduce_size", "Input 'orig' must have 3 columns.");
    else
    {
        no_ray = mxGetM(prhs[1]);
        orig.height = (unsigned int)no_ray;
        orig.width = 3;
        orig.elements = (float *)mxGetData(prhs[1]);
    }

    // Validate 'dest'
    if (nrhs < 3 || mxIsEmpty(prhs[2]))
        reduce_mesh_size = false;
    else if (!mxIsSingle(prhs[2]))
        mexErrMsgIdAndTxt("QuaDRiGa:mesh_reduce_size", "Input 'dest' must be provided in 'single' precision.");
    else if (mxGetN(prhs[2]) != 3)
        mexErrMsgIdAndTxt("QuaDRiGa:mesh_reduce_size", "Input 'dest' must have 3 columns.");
    else if (mxGetM(prhs[2]) != no_ray)
        mexErrMsgIdAndTxt("QuaDRiGa:mesh_reduce_size", "Inputs 'orig' and 'dest' must have the same size.");
    else
    {
        dest.height = (unsigned int)no_ray;
        dest.width = 3;
        dest.elements = (float *)mxGetData(prhs[2]);
    }

    float *min_max = new float[5];
    min_max[0] = -3.402823e+38; // Min. x
    min_max[1] = 3.402823e+38;  // Max. x
    min_max[2] = -3.402823e+38; // Min. y
    min_max[3] = 3.402823e+38;  // Max. y
    min_max[4] = -3.402823e+38; // Min. z
    min_max[5] = 3.402823e+38;  // Max. z

    unsigned int *no_mesh_d = new unsigned int[0];
    no_mesh_d[0] = (unsigned int)no_mesh;

    // Get the required number of elements in the compressed mesh
    if (reduce_mesh_size)
        get_mesh_size(orig, dest, mesh, min_max, no_mesh_d);

    // Create ouput arrays
    plhs[0] = mxCreateNumericMatrix(no_mesh_d[0], 9, mxSINGLE_CLASS, mxREAL); // Number of interactions per ray
    plhs[1] = mxCreateLogicalMatrix(no_mesh, 1);                              // Index lit of the used mesh elements

    meshE.height = no_mesh_d[0];
    meshE.width = 9;
    meshE.elements = (float *)mxGetData(plhs[0]);

    bool *usedE = (bool *)mxGetData(plhs[1]);

    // Reduce mesh size
    mesh_reduce_size(mesh, min_max, no_mesh_d, meshE, usedE);
}
