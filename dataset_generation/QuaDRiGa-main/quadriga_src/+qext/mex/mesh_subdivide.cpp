#include "mex.h"
#include "../src/qd_mesh_functions.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    // Validate "mesh"
    if (nlhs != 1 || nrhs < 2)
        mexErrMsgIdAndTxt("QuaDRiGa:mesh_subdivide", "Wrong number of input/output arguments.");
    if (!mxIsSingle(prhs[0]))
        mexErrMsgIdAndTxt("QuaDRiGa:mesh_subdivide", "Input 'mesh' must be provided in 'single' precision.");
    if (mxGetN(prhs[0]) != 9)
        mexErrMsgIdAndTxt("QuaDRiGa:mesh_subdivide", "Input 'mesh' must have 9 columns.");

    mwSize no_mesh = mxGetM(prhs[0]); // Number of mesh elements

    Matrix mesh;
    mesh.height = (unsigned int)no_mesh;
    mesh.width = 9;
    mesh.elements = (float *)mxGetData(prhs[0]);

    // Validate "no_div"
    unsigned int *no_div;
    if (mxGetNumberOfElements(prhs[1]) != 1)
        mexErrMsgIdAndTxt("QuaDRiGa:mesh_subdivide", "Input 'no_div' must be scalar.");
    else if (mxIsDouble(prhs[1]))
    {
        double *tmp = (double *)mxGetData(prhs[1]);
        no_div = new unsigned int[0];
        no_div[0] = (unsigned int)tmp[0];
    }
    else if (mxIsClass(prhs[1], "uint32") || mxIsClass(prhs[1], "int32"))
        no_div = (unsigned int *)mxGetData(prhs[1]);
    else
        mexErrMsgIdAndTxt("QuaDRiGa:mesh_subdivide", "Input 'no_div' must be either of type 'double', 'int32' or 'uint32'.");

    if (no_div[0] < 2)
        mexErrMsgIdAndTxt("QuaDRiGa:mesh_subdivide", "Minimum number of divisions per edge 'no_div' must be 2.");

    mwSize no_mesh_d = no_mesh * no_div[0] * no_div[0]; // Size of the subdivided mesh

    // Create ouput arrays
    plhs[0] = mxCreateNumericMatrix(no_mesh_d, 9, mxSINGLE_CLASS, mxREAL); // Number of interactions per ray

    Matrix meshS;
    meshS.height = (unsigned int)no_mesh_d;
    meshS.width = 9;
    meshS.elements = (float *)mxGetData(plhs[0]);

    mesh_subdivide(mesh, no_div, meshS);
}
