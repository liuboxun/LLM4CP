#include <iostream>
#include <fstream>
#include <algorithm>
#include "qd_mesh_functions.h"
#include "ray_mesh_intersect.h"
using namespace std;

// Simple code for parsing the command line argumants
char *getCmdOption(char **begin, char **end, const string &option)
{
    char **itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
        return *itr;
    return 0;
}
bool cmdOptionExists(char **begin, char **end, const std::string &option)
{
    return std::find(begin, end, option) != end;
}

int main(int argc, char **argv)
{
    // Parse command line arguments
    string fn_in = (string)getCmdOption(argv, argv + argc, "-i");
    string fn_out = (string)getCmdOption(argv, argv + argc, "-o");

    // Determine if we want status reports (read from command line)
    int verbose = 0;
    if (cmdOptionExists(argv, argv + argc, "-v"))
        verbose = 2;

    // Determine the number of interection points that should be returned (read from command line)
    unsigned int no_hit_W = 1;
    string no_hit_W_str = "0";
    if (cmdOptionExists(argv, argv + argc, "-w"))
    {
        no_hit_W_str = (string)getCmdOption(argv, argv + argc, "-w");
        no_hit_W = (unsigned int)std::stof(no_hit_W_str);
        no_hit_W = (no_hit_W < 1) ? 1 : no_hit_W;
    }

    // Variables for file-input/putput
    unsigned int height, width;
    Matrix orig;
    Matrix dest;
    Matrix mesh;

    unsigned int no_ray, no_mesh;

    if (verbose == 2)
        cout << "Read file:      " << fn_in << endl;

    // Read input data from file
    ifstream file(fn_in, ios::in | ios::binary);
    if (file.is_open())
    {
        file.read((char *)&no_ray, sizeof(unsigned int));
        file.read((char *)&width, sizeof(unsigned int));
        orig.height = no_ray;
        orig.width = width;
        orig.elements = new float[no_ray * width];
        file.read((char *)orig.elements, sizeof(float) * no_ray * width);

        file.read((char *)&height, sizeof(unsigned int));
        file.read((char *)&width, sizeof(unsigned int));
        if (height != no_ray)
        {
            cout << "Size of dest does not match size of orig!" << endl;
            file.close();
            return -1;
        }
        dest.height = no_ray;
        dest.width = width;
        dest.elements = new float[no_ray * width];
        file.read((char *)dest.elements, sizeof(float) * no_ray * width);

        file.read((char *)&no_mesh, sizeof(unsigned int));
        file.read((char *)&width, sizeof(unsigned int));
        mesh.height = no_mesh;
        mesh.width = width;
        mesh.elements = new float[no_mesh * width];
        file.read((char *)mesh.elements, sizeof(float) * no_mesh * width);
        file.close();
    }
    else
    {
        cout << "Input file not found!" << endl;
        return -1;
    }

    // Calculate reduced mesh size
    float *min_max = new float[5];
    unsigned int *no_mesh_d = new unsigned int[0];
    get_mesh_size(orig, dest, mesh, min_max, no_mesh_d);

    // Initialize variables for the output
    Matrix fbs, lbs, Wout, meshE;
    fbs.height = no_ray;
    fbs.width = 3;
    fbs.elements = new float[no_ray * 3];

    lbs.height = no_ray;
    lbs.width = 3;
    lbs.elements = new float[no_ray * 3];

    Wout.height = no_hit_W;
    Wout.width = no_ray;
    Wout.elements = new float[no_hit_W * no_ray];

    meshE.height = no_mesh_d[0];
    meshE.width = 9;
    meshE.elements = new float[no_mesh_d[0] * 9];

    bool *usedE = new bool[no_mesh];

    // Get the mesh edges
    mesh_reduce_size(mesh, min_max, no_mesh_d, meshE, usedE);

    unsigned int *hit, *iFBS;
    hit = new unsigned int[no_ray];
    iFBS = new unsigned int[no_ray];

    ray_mesh_intersect_CUDA(orig, dest, meshE, fbs, lbs, hit, iFBS, Wout, verbose);

    // Write resuts to file
    if (verbose == 2)
        cout << "Write file:     " << fn_out << endl;

    unsigned int one = 1;
    
    ofstream file_w(fn_out, ios::out | ios::binary);
    if (file_w.is_open())
    {
        file_w.write((char *)&no_ray, sizeof(unsigned int));
        file_w.write((char *)&one, sizeof(unsigned int));
        file_w.write((char *)&hit[0], no_ray * sizeof(unsigned int));

        file_w.write((char *)&fbs.height, sizeof(unsigned int));
        file_w.write((char *)&fbs.width, sizeof(unsigned int));
        file_w.write((char *)&fbs.elements[0], 3 * no_ray * sizeof(float));

        file_w.write((char *)&lbs.height, sizeof(unsigned int));
        file_w.write((char *)&lbs.width, sizeof(unsigned int));
        file_w.write((char *)&lbs.elements[0], 3 * no_ray * sizeof(float));

        file_w.write((char *)&no_ray, sizeof(unsigned int));
        file_w.write((char *)&one, sizeof(unsigned int));
        file_w.write((char *)&iFBS[0], no_ray * sizeof(unsigned int));

        file_w.write((char *)&Wout.height, sizeof(unsigned int));
        file_w.write((char *)&Wout.width, sizeof(unsigned int));
        file_w.write((char *)&Wout.elements[0], Wout.height * Wout.width * sizeof(float));
        file_w.close();
    }
    else
    {
        cout << "Output file not found!" << endl;
        return -1;
    }
}
