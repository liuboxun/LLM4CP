#ifndef ray_mesh_intersect_H
#define ray_mesh_intersect_H

void ray_mesh_intersect_CUDA(Matrix orig, Matrix dest, Matrix mesh, Matrix fbs, Matrix lbs,
                             unsigned int *hit, unsigned int *iFBS, Matrix Wout, int verbose);

void ray_rx_intersect_CUDA(Matrix rx_pos, Matrix orig, Matrix dest, float *path_length, Matrix trivec,
                           int verbose, unsigned int *hit_cnt, unsigned int *iRAY, Matrix ray_length);

void ray_reflect_CUDA(Matrix orig, Matrix dest, Matrix fbs, float *path_length, Matrix trivec,
                 unsigned int *iFBS, Matrix mesh, Matrix mtl_prop,
                 Matrix origN, Matrix destN, float *path_lengthN, Matrix trivecN, Matrix refl_coeff);

#endif
