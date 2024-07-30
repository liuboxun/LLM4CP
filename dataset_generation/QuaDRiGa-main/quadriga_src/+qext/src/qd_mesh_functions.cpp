#include <iostream>
#include <cmath>
#include "qd_mesh_functions.h"
using namespace std;

// FUNCTION: Compute the required mesh size in order to remove not needed parts of the mesh (improves computing time)
void get_mesh_size(Matrix orig, Matrix dest, Matrix mesh, float *min_max, unsigned int *no_mesh_d)
{
    // Input variables:
    //  orig            Ray origin, matrix of size [no_ray x 3]
    //  dest            Ray destinations, matrix of size [no_ray x 3]
    //  mesh            Mesh coordinates, matrix of size [no_mesh x 9]

    // Output variables:
    //  min_max         Minimum and maximum values of the x, y and z component, Vector [minX, maxX, minY, maxY, minZ, maxZ]
    //  no_mesh_d       Size of the compressed mesh, scalar

    // Check inputs
    if (orig.width != 3 || dest.width != 3)
    {
        cout << "Error: 'orig' and 'dest' must have 3 columns!" << endl;
        return;
    }
    if (dest.height != orig.height)
    {
        cout << "Error: 'orig' and 'dest' must have the same number of rows!" << endl;
        return;
    }
    if (mesh.width != 9)
    {
        cout << "Error: 'mesh' must have 9 columns!" << endl;
        return;
    }

    unsigned int no_ray = orig.height;  // Number of rays, must be the same in orig and dest
    unsigned int no_mesh = mesh.height; // Mesh size

    // Find the minimum and maximum values in orig and dest
    min_max[0] = 3.402823e+38;  // Min. x
    min_max[1] = -3.402823e+38; // Max. x
    min_max[2] = 3.402823e+38;  // Min. y
    min_max[3] = -3.402823e+38; // Max. y
    min_max[4] = 3.402823e+38;  // Min. z
    min_max[5] = -3.402823e+38; // Max. z
    for (unsigned int i = 0; i < no_ray; i++)
    {
        min_max[0] = (orig.elements[i] < min_max[0]) ? orig.elements[i] : min_max[0];
        min_max[0] = (dest.elements[i] < min_max[0]) ? dest.elements[i] : min_max[0];
        min_max[1] = (orig.elements[i] > min_max[1]) ? orig.elements[i] : min_max[1];
        min_max[1] = (dest.elements[i] > min_max[1]) ? dest.elements[i] : min_max[1];
        min_max[2] = (orig.elements[i + no_ray] < min_max[2]) ? orig.elements[i + no_ray] : min_max[2];
        min_max[2] = (dest.elements[i + no_ray] < min_max[2]) ? dest.elements[i + no_ray] : min_max[2];
        min_max[3] = (orig.elements[i + no_ray] > min_max[3]) ? orig.elements[i + no_ray] : min_max[3];
        min_max[3] = (dest.elements[i + no_ray] > min_max[3]) ? dest.elements[i + no_ray] : min_max[3];
        min_max[4] = (orig.elements[i + 2 * no_ray] < min_max[4]) ? orig.elements[i + 2 * no_ray] : min_max[4];
        min_max[4] = (dest.elements[i + 2 * no_ray] < min_max[4]) ? dest.elements[i + 2 * no_ray] : min_max[4];
        min_max[5] = (orig.elements[i + 2 * no_ray] > min_max[5]) ? orig.elements[i + 2 * no_ray] : min_max[5];
        min_max[5] = (dest.elements[i + 2 * no_ray] > min_max[5]) ? dest.elements[i + 2 * no_ray] : min_max[5];
    }

    no_mesh_d[0] = 0;
    for (unsigned int i = 0; i < no_mesh; i++)
    {
        if ((mesh.elements[i] < min_max[1] || mesh.elements[i + no_mesh * 3] < min_max[1] || mesh.elements[i + no_mesh * 6] < min_max[1]) &&
            (mesh.elements[i] > min_max[0] || mesh.elements[i + no_mesh * 3] > min_max[0] || mesh.elements[i + no_mesh * 6] > min_max[0]) &&
            (mesh.elements[i + no_mesh] < min_max[3] || mesh.elements[i + no_mesh * 4] < min_max[3] || mesh.elements[i + no_mesh * 7] < min_max[3]) &&
            (mesh.elements[i + no_mesh] > min_max[2] || mesh.elements[i + no_mesh * 4] > min_max[2] || mesh.elements[i + no_mesh * 7] > min_max[2]) &&
            (mesh.elements[i + no_mesh * 2] < min_max[5] || mesh.elements[i + no_mesh * 5] < min_max[5] || mesh.elements[i + no_mesh * 8] < min_max[5]) &&
            (mesh.elements[i + no_mesh * 2] > min_max[4] || mesh.elements[i + no_mesh * 5] > min_max[4] || mesh.elements[i + no_mesh * 8] > min_max[4]))
            no_mesh_d[0]++;
    }
}

// FUNCTION: Reduce the size of the mesh by removing not needed parts
void mesh_reduce_size(Matrix mesh, float *min_max, unsigned int *no_mesh_d, Matrix meshE, bool *usedE)
{
    // Input variables:
    //  mesh            Mesh coordinates, matrix of size [no_mesh x 9]
    //  min_max         Minimum and maximum values of the x, y and z component
    //  no_mesh_d       Size of the reduced mesh

    // Output variables:
    //  meshE           Reduced mesh, matrix of size [no_mesh_d x 9]
    //  usedE           Indicator if mesh element was used in output [no_mesh x 1]

    unsigned int no_mesh = mesh.height; // Mesh size
    unsigned int j = 0;
    unsigned int nj = no_mesh_d[0];
    for (unsigned int i = 0; i < no_mesh; i++)
    {
        if ((mesh.elements[i] < min_max[1] || mesh.elements[i + no_mesh * 3] < min_max[1] || mesh.elements[i + no_mesh * 6] < min_max[1]) &&
            (mesh.elements[i] > min_max[0] || mesh.elements[i + no_mesh * 3] > min_max[0] || mesh.elements[i + no_mesh * 6] > min_max[0]) &&
            (mesh.elements[i + no_mesh] < min_max[3] || mesh.elements[i + no_mesh * 4] < min_max[3] || mesh.elements[i + no_mesh * 7] < min_max[3]) &&
            (mesh.elements[i + no_mesh] > min_max[2] || mesh.elements[i + no_mesh * 4] > min_max[2] || mesh.elements[i + no_mesh * 7] > min_max[2]) &&
            (mesh.elements[i + no_mesh * 2] < min_max[5] || mesh.elements[i + no_mesh * 5] < min_max[5] || mesh.elements[i + no_mesh * 8] < min_max[5]) &&
            (mesh.elements[i + no_mesh * 2] > min_max[4] || mesh.elements[i + no_mesh * 5] > min_max[4] || mesh.elements[i + no_mesh * 8] > min_max[4]))
        {
            meshE.elements[j] = mesh.elements[i];                        // v1x
            meshE.elements[j + nj] = mesh.elements[i + no_mesh];         // v1y
            meshE.elements[j + 2 * nj] = mesh.elements[i + 2 * no_mesh]; // v1z
            meshE.elements[j + 3 * nj] = mesh.elements[i + 3 * no_mesh]; // v2x
            meshE.elements[j + 4 * nj] = mesh.elements[i + 4 * no_mesh]; // v2y
            meshE.elements[j + 5 * nj] = mesh.elements[i + 5 * no_mesh]; // v2z
            meshE.elements[j + 6 * nj] = mesh.elements[i + 6 * no_mesh]; // v3x
            meshE.elements[j + 7 * nj] = mesh.elements[i + 7 * no_mesh]; // v3y
            meshE.elements[j + 8 * nj] = mesh.elements[i + 8 * no_mesh]; // v3z
            usedE[i] = true;
            j++;
        }
    }
}

// FUNCTION: Subdivide mesh triangels into smaller triangles
void mesh_subdivide(Matrix mesh, unsigned int *no_div, Matrix meshS)
{
    // Input variables:
    //  mesh            Mesh coordinates, matrix of size [no_mesh x 9]
    //  no_div          Number of sub-segments per edge

    // Output variables:
    //  meshS           Sub-divided mesh [no_mesh*no_div^2 x 9]

    if (mesh.width != 9 || meshS.width != 9)
    {
        cout << "Error: 'mesh' must have 9 columns!" << endl;
        return;
    }
    if (meshS.height != mesh.height * no_div[0] * no_div[0])
    {
        cout << "Error: 'meshS' does not have the required number of rows!" << endl;
        return;
    }

    // Process each mesh element
    unsigned int no_mesh = mesh.height;                       // Mesh size
    unsigned int no_mesh_d = no_mesh * no_div[0] * no_div[0]; // Output mesh size
    unsigned int cnt = 0;                                     // Counter
    float v1x, v1y, v1z, e12x, e12y, e12z, e13x, e13y, e13z;  // Current mesh variables
    float ful, fuu, fvl, fvu;                                 // Offset within current mesh element
    float stp = 1 / (float)no_div[0];                         // Step sive
    for (unsigned int i = 0; i < no_mesh; i++)
    {
        // Read current mesh element
        v1x = mesh.elements[i];
        v1y = mesh.elements[i + no_mesh];
        v1z = mesh.elements[i + 2 * no_mesh];
        e12x = mesh.elements[i + 3 * no_mesh] - v1x;
        e12y = mesh.elements[i + 4 * no_mesh] - v1y;
        e12z = mesh.elements[i + 5 * no_mesh] - v1z;
        e13x = mesh.elements[i + 6 * no_mesh] - v1x;
        e13y = mesh.elements[i + 7 * no_mesh] - v1y;
        e13z = mesh.elements[i + 8 * no_mesh] - v1z;

        for (unsigned int u = 0; u < no_div[0]; u++)
        {
            ful = (float)u * stp;
            fuu = ful + stp;
            for (unsigned int v = 0; v < no_div[0] - u; v++)
            {
                fvl = (float)v * stp;
                fvu = fvl + stp;

                // Lower triangle first vertex
                meshS.elements[cnt] = v1x + fvl * e12x + ful * e13x;                 // w1x
                meshS.elements[cnt + no_mesh_d] = v1y + fvl * e12y + ful * e13y;     // w1y
                meshS.elements[cnt + 2 * no_mesh_d] = v1z + fvl * e12z + ful * e13z; // w1z

                // Lower triangle second vertex
                meshS.elements[cnt + 3 * no_mesh_d] = v1x + fvu * e12x + ful * e13x; // w2x
                meshS.elements[cnt + 4 * no_mesh_d] = v1y + fvu * e12y + ful * e13y; // w2y
                meshS.elements[cnt + 5 * no_mesh_d] = v1z + fvu * e12z + ful * e13z; // w2z

                // Lower triangle third vertex
                meshS.elements[cnt + 6 * no_mesh_d] = v1x + fvl * e12x + fuu * e13x; // w2x
                meshS.elements[cnt + 7 * no_mesh_d] = v1y + fvl * e12y + fuu * e13y; // w2y
                meshS.elements[cnt + 8 * no_mesh_d] = v1z + fvl * e12z + fuu * e13z; // w2z
                cnt++;

                if (v < no_div[0] - u - 1)
                {
                    // Upper triangle first vertex
                    meshS.elements[cnt] = v1x + fvl * e12x + fuu * e13x;                 // w1x
                    meshS.elements[cnt + no_mesh_d] = v1y + fvl * e12y + fuu * e13y;     // w1y
                    meshS.elements[cnt + 2 * no_mesh_d] = v1z + fvl * e12z + fuu * e13z; // w1z

                    // Upper triangle second vertex
                    meshS.elements[cnt + 3 * no_mesh_d] = v1x + fvu * e12x + fuu * e13x; // w2x
                    meshS.elements[cnt + 4 * no_mesh_d] = v1y + fvu * e12y + fuu * e13y; // w2y
                    meshS.elements[cnt + 5 * no_mesh_d] = v1z + fvu * e12z + fuu * e13z; // w2z

                    // Upper triangle third vertex
                    meshS.elements[cnt + 6 * no_mesh_d] = v1x + fvu * e12x + ful * e13x; // w2x
                    meshS.elements[cnt + 7 * no_mesh_d] = v1y + fvu * e12y + ful * e13y; // w2y
                    meshS.elements[cnt + 8 * no_mesh_d] = v1z + fvu * e12z + ful * e13z; // w2z
                    cnt++;
                }
            }
        }
    }
}

// FUNCTION: Constructs an geodesic polyhedron, a convex polyhedron made from triangles
void icosphere(unsigned int *no_div, Matrix dest, float *length, Matrix trivec)
{
    // Input variables:
    //  no_div          Number of sub-segments per edge, resuts in no_ray = 20 * no_div^2 elements

    // Output variables:
    //  dest            Vector pointing from the origin to the center of the triangle, matrix of size [no_ray x 3]
    //  length          Length of the vector "dest" (slightly smaller than 1), vector of lenght[no_ray x 1]
    //  trivec          Vectors pointing from "dest" to the vertices of the triangle, matrix of size [no_ray x 9], [x1 y1 z1 x2 y2 z3 x3 y3 z3]

    unsigned int no_ray = no_div[0] * no_div[0] * 20;

    if (dest.width != 3)
    {
        cout << "Error: 'dest' must have 3 columns!" << endl;
        return;
    }
    if (trivec.width != 9)
    {
        cout << "Error: 'trivec' must have 9 columns!" << endl;
        return;
    }
    if (dest.height != no_ray)
    {
        cout << "Error: 'dest' does not have the required number of rows!" << endl;
        return;
    }
    if (trivec.height != no_ray)
    {
        cout << "Error: 'trivec' does not have the required number of rows!" << endl;
        return;
    }

    // Vertex coordinates of a regular isohedron
    float p = 1.6180340;
    float val[180] = {0, 0, 0, 0, 0, 0, 0, 0, -p, -p, p, p, -1, -1, 1, 0, 0, 0, 0, 0,
                      1, 1, 1, 1, 1, -1, -1, -1, 0, 0, 0, 0, -p, -p, -p, 1, 1, 1, 1, 1,
                      p, p, p, p, p, p, p, p, 1, 1, 1, 1, 0, 0, 0, -p, -p, -p, -p, -p,
                      0, 0, 1, -1, 1, p, -1, -p, -1, -p, 1, 1, -p, 1, 0, 1, -1, -p, 0, 1,
                      -1, -1, p, p, p, 0, -p, 0, p, 0, -p, p, 0, -p, -1, p, p, 0, -1, p,
                      p, p, 0, 0, 0, 1, 0, 1, 0, -1, 0, 0, -1, 0, -p, 0, 0, -1, -p, 0,
                      p, -p, p, -p, -1, 1, 1, -1, -p, -1, p, p, 0, 0, p, -1, -p, 0, p, p,
                      0, 0, 0, 0, p, -p, -p, -p, 0, -p, 0, 0, -1, -1, 0, p, 0, -1, 0, 0,
                      1, 1, 1, 1, 0, 0, 0, 0, -1, 0, -1, -1, -p, -p, -1, 0, -1, -p, -1, -1};

    // Rotate x and y-coordinates slightly to avoid artefacts in regular grids
    float si = 0.0078329f;
    float co = 0.9999693f;
    float tmp;
    for (unsigned int i = 0; i < 20; i++)
    {
        tmp = val[i];
        val[i] = co * tmp - si * val[i + 20];
        val[i + 20] = si * tmp + co * val[i + 20];
        tmp = val[i + 60];
        val[i + 60] = co * tmp - si * val[i + 80];
        val[i + 80] = si * tmp + co * val[i + 80];
        tmp = val[i + 120];
        val[i + 120] = co * tmp - si * val[i + 140];
        val[i + 140] = si * tmp + co * val[i + 140];
    }

    Matrix mesh;
    mesh.height = 20;
    mesh.width = 9;
    mesh.elements = val;

    // Subdivide the faces of the isohedron
    mesh_subdivide(mesh, no_div, trivec);

    // Process all faces
    p = 1.0F / 3.0F;
    for (unsigned int i = 0; i < no_ray; i++)
    {
        // Project triangles onto the unit sphere
        // First vertex
        tmp = 1.0F / sqrt(trivec.elements[i] * trivec.elements[i] +
                          trivec.elements[i + no_ray] * trivec.elements[i + no_ray] +
                          trivec.elements[i + 2 * no_ray] * trivec.elements[i + 2 * no_ray]);
        trivec.elements[i] = trivec.elements[i] * tmp;
        trivec.elements[i + no_ray] = trivec.elements[i + no_ray] * tmp;
        trivec.elements[i + 2 * no_ray] = trivec.elements[i + 2 * no_ray] * tmp;

        // Second vertex
        tmp = 1.0F / sqrt(trivec.elements[i + 3 * no_ray] * trivec.elements[i + 3 * no_ray] +
                          trivec.elements[i + 4 * no_ray] * trivec.elements[i + 4 * no_ray] +
                          trivec.elements[i + 5 * no_ray] * trivec.elements[i + 5 * no_ray]);
        trivec.elements[i + 3 * no_ray] = trivec.elements[i + 3 * no_ray] * tmp;
        trivec.elements[i + 4 * no_ray] = trivec.elements[i + 4 * no_ray] * tmp;
        trivec.elements[i + 5 * no_ray] = trivec.elements[i + 5 * no_ray] * tmp;

        // Third vertex
        tmp = 1.0F / sqrt(trivec.elements[i + 6 * no_ray] * trivec.elements[i + 6 * no_ray] +
                          trivec.elements[i + 7 * no_ray] * trivec.elements[i + 7 * no_ray] +
                          trivec.elements[i + 8 * no_ray] * trivec.elements[i + 8 * no_ray]);
        trivec.elements[i + 6 * no_ray] = trivec.elements[i + 6 * no_ray] * tmp;
        trivec.elements[i + 7 * no_ray] = trivec.elements[i + 7 * no_ray] * tmp;
        trivec.elements[i + 8 * no_ray] = trivec.elements[i + 8 * no_ray] * tmp;

        // Calculate center cordinate
        dest.elements[i] = (trivec.elements[i] + trivec.elements[i + 3 * no_ray] + trivec.elements[i + 6 * no_ray]) * p;                           // x
        dest.elements[i + no_ray] = (trivec.elements[i + no_ray] + trivec.elements[i + 4 * no_ray] + trivec.elements[i + 7 * no_ray]) * p;         // y
        dest.elements[i + 2 * no_ray] = (trivec.elements[i + 2 * no_ray] + trivec.elements[i + 5 * no_ray] + trivec.elements[i + 8 * no_ray]) * p; // z

        // Calculate length
        length[i] = sqrt(dest.elements[i] * dest.elements[i] +
                         dest.elements[i + no_ray] * dest.elements[i + no_ray] +
                         dest.elements[i + 2 * no_ray] * dest.elements[i + 2 * no_ray]);

        // Calculate vectors pointing from the face center to the triangle vertices
        trivec.elements[i] = trivec.elements[i] - dest.elements[i];
        trivec.elements[i + no_ray] = trivec.elements[i + no_ray] - dest.elements[i + no_ray];
        trivec.elements[i + 2 * no_ray] = trivec.elements[i + 2 * no_ray] - dest.elements[i + 2 * no_ray];
        trivec.elements[i + 3 * no_ray] = trivec.elements[i + 3 * no_ray] - dest.elements[i];
        trivec.elements[i + 4 * no_ray] = trivec.elements[i + 4 * no_ray] - dest.elements[i + no_ray];
        trivec.elements[i + 5 * no_ray] = trivec.elements[i + 5 * no_ray] - dest.elements[i + 2 * no_ray];
        trivec.elements[i + 6 * no_ray] = trivec.elements[i + 6 * no_ray] - dest.elements[i];
        trivec.elements[i + 7 * no_ray] = trivec.elements[i + 7 * no_ray] - dest.elements[i + no_ray];
        trivec.elements[i + 8 * no_ray] = trivec.elements[i + 8 * no_ray] - dest.elements[i + 2 * no_ray];
    }
}

