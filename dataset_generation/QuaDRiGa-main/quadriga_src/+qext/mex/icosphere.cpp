#include "mex.h"
#include "../src/qd_mesh_functions.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Validate input
    if (nlhs != 3 || nrhs < 1)
        mexErrMsgIdAndTxt("QuaDRiGa:icosphere", "Wrong number of input/output arguments.");

    // Validate "no_div"
    unsigned int *no_div;
    if (mxGetNumberOfElements(prhs[0]) != 1)
        mexErrMsgIdAndTxt("QuaDRiGa:icosphere", "Input 'no_div' must be scalar.");
    else if (mxIsDouble(prhs[0]))
    {
        double *tmp = (double *)mxGetData(prhs[0]);
        no_div = new unsigned int[0];
        no_div[0] = (unsigned int)tmp[0];
    }
    else if (mxIsClass(prhs[0], "uint32") || mxIsClass(prhs[0], "int32"))
        no_div = (unsigned int *)mxGetData(prhs[0]);
    else
        mexErrMsgIdAndTxt("QuaDRiGa:icosphere", "Input 'no_div' must be either of type 'double', 'int32' or 'uint32'.");

    if (no_div[0] < 1)
        mexErrMsgIdAndTxt("QuaDRiGa:icosphere", "Minimum number of divisions per edge 'no_div' must be 1.");

    mwSize no_mesh_d = 20 * no_div[0] * no_div[0]; // Size of the subdivided mesh

    // Create ouput arrays
    plhs[0] = mxCreateNumericMatrix(no_mesh_d, 3, mxSINGLE_CLASS, mxREAL); // Cartesian coordinates center coordinate
    plhs[1] = mxCreateNumericMatrix(no_mesh_d, 1, mxSINGLE_CLASS, mxREAL); // Vector length
    plhs[2] = mxCreateNumericMatrix(no_mesh_d, 9, mxSINGLE_CLASS, mxREAL); // Vectors pointing from the center to the edge
    
    Matrix dest;
    dest.height = (unsigned int)no_mesh_d;
    dest.width = 3;
    dest.elements = (float *)mxGetData(plhs[0]);

    float *length;
    length = (float *)mxGetData(plhs[1]);

    Matrix trivec;
    trivec.height = (unsigned int)no_mesh_d;
    trivec.width = 9;
    trivec.elements = (float *)mxGetData(plhs[2]);

    icosphere(no_div, dest, length, trivec);
}