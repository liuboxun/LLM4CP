#ifndef qd_mesh_functions_H
#define qd_mesh_functions_H

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct
{
    unsigned int height;
    unsigned int width;
    float *elements;
} Matrix;

void get_mesh_size(Matrix orig, Matrix dest, Matrix mesh, float *min_max, unsigned int *no_mesh_d);

void mesh_reduce_size(Matrix mesh, float *min_max, unsigned int *no_mesh_d, Matrix meshE, bool *usedE);

void mesh_subdivide(Matrix mesh, unsigned int *no_div, Matrix meshS);

void icosphere(unsigned int *no_div, Matrix dest, float *length, Matrix trivec);

#endif
