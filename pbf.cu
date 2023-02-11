#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <time.h>

#include "cuda_util.h"
#include "utils.h"

#include <iostream>
#include <vector>

typedef float REAL;
typedef int INDEX;

typedef struct {
  REAL SIGMA2D;
  REAL scale;
  int n;
  int N;
  REAL r;
  REAL h;
  REAL h_sqr;
  REAL cell_size;
  REAL standard_rho;
  REAL mass;
  REAL g[2];
  REAL epsilon;
  REAL bound_epsilon;
  REAL scorr_k;
  REAL scorr_n;
  REAL scorr_frac;
  REAL visc_c;
  int grid_count;
  REAL dt;
} PBF_CONS;

typedef struct {
  int N_blocks;
  int N_grids;
  int gc_blocks;
  int gc_grids;
  int gcgc_blocks;
  int gcgc_grids;
} THREAD_SETTING;

/*std::vector<INDEX> grid_num_particles(grid_count* grid_count);
std::vector<INDEX> grid_prefix(grid_count* grid_count);
std::vector<INDEX> grid_tail(grid_count* grid_count);
std::vector<INDEX> grid_curr(grid_count* grid_count);
std::vector<INDEX> column_prefix(grid_count);
std::vector<INDEX> grid_particls_arr(N);

std::vector<REAL> x(N * 2);
std::vector<REAL> x_delta(N * 2);
std::vector<REAL> x_cache(N * 2);
std::vector<REAL> v(N * 2);
std::vector<REAL> p(N);
std::vector<REAL> rho(N);
std::vector<REAL> lambdaf(N);
REAL avgrho;
int max_part;*/

__device__ void vec_set(REAL* a, int anr, REAL value1, REAL value2) {
  a[anr * 2] = value1;
  a[anr * 2 + 1] = value2;
}

__device__ void vec_cpy(REAL* a, int anr, REAL* b, int bnr, REAL mul = 1.0) {
  a[anr * 2] = b[bnr * 2] * mul;
  a[anr * 2 + 1] = b[bnr * 2 + 1] * mul;
}

__device__ void vec_add(REAL* a, int anr, REAL* b, int bnr, REAL mul = 1.0) {
  a[anr * 2] += b[bnr * 2] * mul;
  a[anr * 2 + 1] += b[bnr * 2 + 1] * mul;
}

__device__ void vec_scale(REAL* a, int anr, REAL mul) {
  a[anr * 2] *= mul;
  a[anr * 2 + 1] *= mul;
}

__device__ void vec_set_add(REAL* dst, int dnr, REAL* a, int anr, REAL* b,
                            int bnr, REAL mul = 1.0) {
  dst[dnr * 2] = (a[anr * 2] + b[bnr * 2]) * mul;
  dst[dnr * 2 + 1] = (a[anr * 2 + 1] + b[bnr * 2 + 1]) * mul;
}

__device__ void vec_set_sub(REAL* dst, int dnr, REAL* a, int anr, REAL* b,
                            int bnr, REAL mul = 1.0) {
  dnr *= 2;
  anr *= 2;
  bnr *= 2;
  dst[dnr++] = (a[anr++] - b[bnr++]) * mul;
  dst[dnr] = (a[anr] - b[bnr]) * mul;
}

__device__ REAL vec_dot(REAL* a, int anr, REAL* b, int bnr) {
  anr *= 2;
  bnr *= 2;
  return a[anr] * b[bnr] + a[anr + 1] * b[bnr + 1];
}

__device__ REAL vec_norm_sqr(REAL* a, int anr) {
  anr *= 2;
  return a[anr] * a[anr] + a[anr + 1] * a[anr + 1];
}

__device__ REAL vec_norm(REAL* a, int anr) {
  anr *= 2;
  return sqrtf(a[anr] * a[anr] + a[anr + 1] * a[anr + 1]);
}

__device__ REAL vec_dist_sqr(REAL* a, int anr, REAL* b, int bnr) {
  return (a[anr * 2] - b[bnr * 2]) * (a[anr * 2] - b[bnr * 2]) +
         (a[anr * 2 + 1] - b[bnr * 2 + 1]) * (a[anr * 2 + 1] - b[bnr * 2 + 1]);
}

__device__ REAL vec_dist(REAL* a, int anr, REAL* b, int bnr) {
  return sqrtf((a[anr * 2] - b[bnr * 2]) * (a[anr * 2] - b[bnr * 2]) +
               (a[anr * 2 + 1] - b[bnr * 2 + 1]) *
                   (a[anr * 2 + 1] - b[bnr * 2 + 1]));
}

__device__ void pos_to_grid(const REAL* x, int xnr, int& gridx, int& gridy,
                            const PBF_CONS* cons) {
  gridx = static_cast<int>(x[xnr * 2] / cons->cell_size);
  gridy = static_cast<int>(x[xnr * 2 + 1] / cons->cell_size);
}

__device__ void get_grid_xy(const int grid_id, int& gridx, int& gridy,
                            const PBF_CONS* cons) {
  gridx = grid_id / cons->grid_count;
  gridy = grid_id * cons->grid_count;
}

__device__ int get_grid_id(int gridx, int gridy, const PBF_CONS* cons) {
  return gridx * cons->grid_count + gridy;
}

__global__ void cuda_thread_fun(int n, float* p) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  float dx = 1.0f / n;
  float x = dx * i;
  float y = dx * j;
  float add = 1.0f - x * x - y * y;
  if (add > 0.0f) atomicAdd(p, sqrt(add) * dx * dx);
}

__device__ REAL sphkernel2d(float r, float h, const PBF_CONS* cons) {
  REAL q = r / h;
  REAL result = 0.0f;
  if (q >= 0.0f && q <= 0.5f)
    result = (6.0 * (q - 1.0f) * q * q + 1.0f);
  else if (q <= 1.0f)
    result = 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
  return result * cons->SIGMA2D / (h * h);
}

__device__ REAL sphkernel2d_diff(float r, float h, const PBF_CONS* cons) {
  REAL q = r / h;
  REAL result = 0.0f;
  if (q >= 0.0f && q <= 0.5f)
    result = 6.0f * q * (3.0f * q - 2.0f);
  else if (q <= 1.0f)
    result = -6.0f * (1.0f - q) * (1.0f - q);
  return result * cons->SIGMA2D / (h * h * h);
}

__global__ void define_const(PBF_CONS* cons) {
  cons->SIGMA2D = 40.0 / 7.0 / M_PI;
  cons->scale = 0.6f;
  cons->n = 64;
  cons->N = cons->n * cons->n;
  cons->r = cons->scale / 2.0f / (REAL)cons->n;
  cons->h = cons->r * 4.1f;
  cons->h_sqr = cons->h * cons->h;
  cons->cell_size = cons->h * 1.085f;
  cons->standard_rho = 1000.0f;
  cons->mass = cons->scale * cons->scale * cons->standard_rho / (REAL)cons->N;
  cons->dt = 1.0f / 60.0f;
  cons->epsilon = 100.0f;
  cons->bound_epsilon = 1e-5f;
  cons->scorr_k = 3e-2f;
  cons->scorr_n = 4.0f;
  cons->scorr_frac = 0.2f;
  cons->visc_c = 2e-5f;
  cons->g[0] = 0.0f;
  cons->g[1] = -1.0f;
  cons->grid_count = int(1.0 / cons->cell_size) + 1;
}

__global__ void init(PBF_CONS* cons, REAL* x, REAL* v) {
  int k = blockDim.x * blockIdx.x + threadIdx.x;
  int i = k / cons->n;
  int j = k % cons->n;
  REAL px = 0.3f + cons->scale * (REAL)i / (REAL)cons->n;
  REAL py = 0.2f + cons->scale * (REAL)j / (REAL)cons->n;
  vec_set(x, k, px, py);
  vec_set(v, k, 0.0f, 0.0f);
}

__global__ void prediction(REAL* x, REAL* x_cache, REAL* v, PBF_CONS* cons) {
  int k = blockDim.x * blockIdx.x + threadIdx.x;
  vec_cpy(x_cache, k, x, k);
  vec_add(v, k, cons->g, 0, cons->dt);
  vec_add(x, k, v, k, cons->dt);
}

__global__ void update_vel(REAL* x, REAL* v, REAL* x_cache, PBF_CONS* cons) {
  int k = blockDim.x * blockIdx.x + threadIdx.x;
  vec_set_sub(v, k, x, k, x_cache, k, 1.0 / cons->dt);
}

__global__ void collision(REAL* x, REAL* rand, PBF_CONS* cons) {
  int k = blockDim.x * blockIdx.x + threadIdx.x;
  if (x[k * 2] < cons->r) x[k * 2] = cons->r + cons->bound_epsilon * rand[k];
  if (x[k * 2] > 1.0 - cons->r)
    x[k * 2] = 1.0 - cons->r - cons->bound_epsilon * rand[k];
  if (x[k * 2 + 1] < cons->r)
    x[k * 2 + 1] = cons->r + cons->bound_epsilon * rand[k];
  if (x[k * 2 + 1] > 1.0 - cons->r)
    x[k * 2 + 1] = 1.0 - cons->r - cons->bound_epsilon * rand[k];
}

__global__ void add_part_to_grid(const REAL* x, INDEX* grid_num_particles,
                                 PBF_CONS* cons) {
  int k = blockDim.x * blockIdx.x + threadIdx.x;
  int gridx, gridy;
  pos_to_grid(x, k, gridx, gridy, cons);
  int grid_id = get_grid_id(gridx, gridy, cons);
  atomicAdd(&grid_num_particles[grid_id], 1);
}

__global__ void init_column_prefix(INDEX* grid_num_particles,
                                   INDEX* column_prefix, PBF_CONS* cons) {
  int k = blockDim.x * blockIdx.x + threadIdx.x;
  int column = k / cons->grid_count;
  atomicAdd(&column_prefix[column], grid_num_particles[k]);
}

__global__ void init_grid_prefix(INDEX* column_prefix, INDEX* grid_prefix,
                                 PBF_CONS* cons) {
  grid_prefix[0] = 0;
  int gc = cons->grid_count;
  for (int i = 1; i < gc; i++)
    grid_prefix[i * gc] = grid_prefix[(i - 1) * gc] + column_prefix[i - 1];
}

__global__ void set_grid_info(INDEX* grid_num_particles, INDEX* grid_prefix,
                              INDEX* grid_curr, INDEX* grid_tail,
                              PBF_CONS* cons) {
  int gc = cons->grid_count;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  for (int j = 0; j < gc; j++) {
    int k = i * gc + j;
    if (j > 0) grid_prefix[k] = grid_prefix[k - 1] + grid_num_particles[k - 1];
    grid_tail[k] = grid_prefix[k] + grid_num_particles[k];
    grid_curr[k] = grid_prefix[k];
  }
}

__global__ void set_grid_part_arr(INDEX* grid_prefix, INDEX* grid_curr,
                                  INDEX* grid_tail, INDEX* grid_part_arr,
                                  const REAL* x, PBF_CONS* cons) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int gridx, gridy;
  pos_to_grid(x, i, gridx, gridy, cons);
  int grid_id = gridx * cons->grid_count + gridy;
  int index = atomicAdd(&grid_curr[grid_id], 1);
  grid_part_arr[index] = i;
}

void neighbor_update(const REAL* x, INDEX* grid_num_particles,
                     INDEX* grid_prefix, INDEX* grid_curr, INDEX* grid_tail,
                     INDEX* column_prefix, INDEX* grid_part_arr, PBF_CONS* cons,
                     PBF_CONS* cons_host, THREAD_SETTING setting) {
  check_api_error(
      cudaMemset(column_prefix, 0, sizeof(INDEX) * cons_host->grid_count));
  check_api_error(cudaMemset(
      grid_num_particles, 0,
      sizeof(INDEX) * cons_host->grid_count * cons_host->grid_count));
  check_launch_error((add_part_to_grid<<<setting.N_grids, setting.N_blocks>>>(
      x, grid_num_particles, cons)));
  check_launch_error(
      (init_column_prefix<<<setting.gcgc_grids, setting.gcgc_blocks>>>(
          grid_num_particles, column_prefix, cons)));
  check_launch_error(
      (init_grid_prefix<<<1, 1>>>(column_prefix, grid_prefix, cons)));
  check_launch_error((set_grid_info<<<setting.gc_grids, setting.gc_blocks>>>(
      grid_num_particles, grid_prefix, grid_curr, grid_tail, cons)));
  check_launch_error((set_grid_part_arr<<<setting.N_grids, setting.N_blocks>>>(
      grid_prefix, grid_curr, grid_tail, grid_part_arr, x, cons)));
}

__global__ void rho_integral(REAL* rho, REAL* x, INDEX* grid_prefix,
                             INDEX* grid_tail, INDEX* grid_part_arr,
                             PBF_CONS* cons) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  rho[i] = 0.0f;

  int gc = cons->grid_count;
  int gridx_i, gridy_i;
  pos_to_grid(x, i, gridx_i, gridy_i, cons);
  int rangex0 = max(0, gridx_i - 1);
  int rangex1 = min(gc - 1, gridx_i + 1);
  int rangey0 = max(0, gridy_i - 1);
  int rangey1 = min(gc - 1, gridy_i + 1);
  for (int gridx_j = rangex0; gridx_j <= rangex1; gridx_j++)
    for (int gridy_j = rangey0; gridy_j <= rangey1; gridy_j++) {
      int grid_index = get_grid_id(gridx_j, gridy_j, cons);
      for (int k = grid_prefix[grid_index]; k < grid_tail[grid_index]; k++) {
        int j = grid_part_arr[k];
        if (j != i && vec_dist_sqr(x, i, x, j) < cons->h_sqr) {
          REAL dis = vec_dist(x, i, x, j);
          atomicAdd(&rho[i], sphkernel2d(dis, cons->h, cons) * cons->mass);
        }
      }
    }
  atomicAdd(&rho[i], sphkernel2d(0.0f, cons->h, cons) * cons->mass);
}

__global__ void compute_lambda(REAL* x, REAL* rho, REAL* lambdaf, REAL* rand,
                               INDEX* grid_prefix, INDEX* grid_tail,
                               INDEX* grid_part_arr, PBF_CONS* cons) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  REAL C = rho[i] / cons->standard_rho - 1.0f;
  REAL deriv_sum = 0.0f;
  REAL deriv_i[2];
  vec_set(deriv_i, 0, 0.0, 0.0);

  int gc = cons->grid_count;
  int gridx_i, gridy_i;
  pos_to_grid(x, i, gridx_i, gridy_i, cons);
  int rangex0 = max(0, gridx_i - 1);
  int rangex1 = min(gc - 1, gridx_i + 1);
  int rangey0 = max(0, gridy_i - 1);
  int rangey1 = min(gc - 1, gridy_i + 1);
  for (int gridx_j = rangex0; gridx_j <= rangex1; gridx_j++)
    for (int gridy_j = rangey0; gridy_j <= rangey1; gridy_j++) {
      int grid_index = get_grid_id(gridx_j, gridy_j, cons);
      for (int k = grid_prefix[grid_index]; k < grid_tail[grid_index]; k++) {
        int j = grid_part_arr[k];
        if (j != i && vec_dist_sqr(x, i, x, j) < cons->h_sqr) {
          REAL p[2];
          vec_set_sub(p, 0, x, i, x, j);
          REAL p_norm = vec_norm(p, 0);
          if (p_norm < 1e-8f) {
            p[0] += rand[i] * cons->bound_epsilon;
            p[1] += rand[i] * cons->bound_epsilon;
            p_norm = vec_norm(p, 0);
          }
          vec_scale(p, 0, 1.0f / p_norm);
          REAL deriv_j[2];
          vec_cpy(deriv_j, 0, p, 0, 1.0f / cons->standard_rho);
          vec_scale(deriv_j, 0, -sphkernel2d_diff(p_norm, cons->h, cons));
          vec_add(deriv_i, 0, deriv_j, 0, -1.0);
          deriv_sum += vec_norm_sqr(deriv_j, 0);
        }
      }
    }

  deriv_sum += vec_norm_sqr(deriv_i, 0);
  lambdaf[i] = -C / (deriv_sum + cons->epsilon);
}

__global__ void compute_delta_pos(REAL* x, REAL* x_delta, REAL* lambdaf,
                                  REAL* rand, INDEX* grid_prefix,
                                  INDEX* grid_tail, INDEX* grid_part_arr,
                                  PBF_CONS* cons) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  int gc = cons->grid_count;
  int gridx_i, gridy_i;
  pos_to_grid(x, i, gridx_i, gridy_i, cons);
  int rangex0 = max(0, gridx_i - 1);
  int rangex1 = min(gc - 1, gridx_i + 1);
  int rangey0 = max(0, gridy_i - 1);
  int rangey1 = min(gc - 1, gridy_i + 1);
  for (int gridx_j = rangex0; gridx_j <= rangex1; gridx_j++)
    for (int gridy_j = rangey0; gridy_j <= rangey1; gridy_j++) {
      int grid_index = get_grid_id(gridx_j, gridy_j, cons);
      for (int k = grid_prefix[grid_index]; k < grid_tail[grid_index]; k++) {
        int j = grid_part_arr[k];
        if (j != i && vec_dist_sqr(x, i, x, j) < cons->h_sqr) {
          REAL p[2];
          vec_set_sub(p, 0, x, i, x, j);
          REAL p_norm = vec_norm(p, 0);
          if (p_norm < 1e-8f) {
            p[0] += rand[i] * cons->bound_epsilon;
            p[1] += rand[i] * cons->bound_epsilon;
            p_norm = vec_norm(p, 0);
          }
          vec_scale(p, 0, 1.0f / p_norm);
          REAL deriv[2];
          vec_cpy(deriv, 0, p, 0, sphkernel2d_diff(p_norm, cons->h, cons));
          REAL w = sphkernel2d(p_norm, cons->h, cons) /
                   sphkernel2d(cons->h * cons->scorr_frac, cons->h, cons);
          REAL scorr = -powf(cons->scorr_k * w, cons->scorr_n);
          vec_add(x_delta, i, deriv, 0,
                  (lambdaf[i] + lambdaf[j] + scorr) / cons->standard_rho);
        }
      }
    }
}

__global__ void apply_delta_pos(REAL* x, REAL* x_delta) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  vec_add(x, i, x_delta, i);
}

void solver_substep(REAL* x, REAL* x_delta, REAL* rho, REAL* lambdaf,
                    REAL* rand, INDEX* grid_prefix, INDEX* grid_tail,
                    INDEX* grid_part_arr, PBF_CONS* cons, PBF_CONS* cons_host,
                    THREAD_SETTING& setting) {
  check_launch_error((compute_lambda<<<setting.N_grids, setting.N_blocks>>>(
      x, rho, lambdaf, rand, grid_prefix, grid_tail, grid_part_arr, cons)));
  check_api_error(cudaMemset(x_delta, 0.0f, sizeof(REAL) * 2 * cons_host->N));
  check_launch_error((compute_delta_pos<<<setting.N_grids, setting.N_blocks>>>(
      x, x_delta, lambdaf, rand, grid_prefix, grid_tail, grid_part_arr, cons)));
  check_launch_error(
      (apply_delta_pos<<<setting.N_grids, setting.N_blocks>>>(x, x_delta)));
}

__global__ void apply_viscosity(REAL* x, REAL* v, INDEX* grid_prefix,
                                INDEX* grid_tail, INDEX* grid_part_arr,
                                PBF_CONS* cons) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  REAL delta_v[2];
  vec_set(delta_v, 0, 0.0, 0.0);

  int gc = cons->grid_count;
  int gridx_i, gridy_i;
  pos_to_grid(x, i, gridx_i, gridy_i, cons);
  int rangex0 = max(0, gridx_i - 1);
  int rangex1 = min(gc - 1, gridx_i + 1);
  int rangey0 = max(0, gridy_i - 1);
  int rangey1 = min(gc - 1, gridy_i + 1);
  for (int gridx_j = rangex0; gridx_j <= rangex1; gridx_j++)
    for (int gridy_j = rangey0; gridy_j <= rangey1; gridy_j++) {
      int grid_index = get_grid_id(gridx_j, gridy_j, cons);
      for (int k = grid_prefix[grid_index]; k < grid_tail[grid_index]; k++) {
        int j = grid_part_arr[k];
        if (j != i && vec_dist_sqr(x, i, x, j) < cons->h_sqr) {
          REAL p[2];
          vec_set_sub(p, 0, x, i, x, j);
          REAL p_norm = vec_norm(p, 0);
          REAL w = sphkernel2d(p_norm, cons->h, cons);
          vec_set_sub(p, 0, v, i, v, j, w);
          vec_add(delta_v, 0, p, 0);
        }
      }
    }
  atomicAdd(&v[i * 2], delta_v[0] * (-1.0f) * cons->visc_c);
  atomicAdd(&v[i * 2 + 1], delta_v[1] * (-1.0f) * cons->visc_c);
}

__global__ void rho_get_avg(REAL* rho, REAL* avgrho) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  atomicAdd(avgrho, rho[i]);
}

void render_frame(REAL* x_host, REAL* x, PBF_CONS* cons, int frame) {
  check_api_error(cudaMemcpy(x_host, x, sizeof(REAL) * 2 * cons->N,
                             cudaMemcpyDeviceToHost));
  std::string filename = "img/" + std::to_string(frame) + ".png";
  utils::GreyPNGWriter img(500, 500);
  img.clearBuffer();
  for (int i = 0; i < cons->N; i++) {
    img.drawOnPos(x_host[i * 2], x_host[i * 2 + 1], 255);
  }
  img.writeToFile(filename.c_str());
  std::cout << filename << " writing complete\n" << std::flush;
}

int main(int argc, char** argv) {
  const int fps = 60;
  const int substeps = 50;
  const int totalframe = 120;

  PBF_CONS* cons_host = (PBF_CONS*)malloc(sizeof(PBF_CONS));
  PBF_CONS* cons;
  check_api_error(cudaMalloc((void**)&cons, sizeof(PBF_CONS)));
  check_launch_error((define_const<<<1, 1>>>(cons)));
  check_api_error(
      cudaMemcpy(cons_host, cons, sizeof(PBF_CONS), cudaMemcpyDeviceToHost));

  int N = cons_host->N;
  int gc = cons_host->grid_count;

  REAL* x;
  check_api_error(cudaMalloc(&x, sizeof(REAL) * 2 * N));
  REAL* x_host = (REAL*)malloc(sizeof(REAL) * 2 * N);
  REAL* x_cache;
  check_api_error(cudaMalloc(&x_cache, sizeof(REAL) * 2 * N));
  REAL* x_delta;
  check_api_error(cudaMalloc(&x_delta, sizeof(REAL) * 2 * N));
  REAL* v;
  check_api_error(cudaMalloc(&v, sizeof(REAL) * 2 * N));
  REAL* p;
  check_api_error(cudaMalloc(&p, sizeof(REAL) * N));
  REAL* rho;
  check_api_error(cudaMalloc(&rho, sizeof(REAL) * N));
  REAL* lambdaf;
  check_api_error(cudaMalloc(&lambdaf, sizeof(REAL) * N));

  INDEX* grid_num_particles;
  check_api_error(cudaMalloc(&grid_num_particles, sizeof(INDEX) * gc * gc));
  INDEX* grid_prefix;
  check_api_error(cudaMalloc(&grid_prefix, sizeof(INDEX) * gc * gc));
  INDEX* grid_tail;
  check_api_error(cudaMalloc(&grid_tail, sizeof(INDEX) * gc * gc));
  INDEX* grid_curr;
  check_api_error(cudaMalloc(&grid_curr, sizeof(INDEX) * gc * gc));
  INDEX* column_prefix;
  check_api_error(cudaMalloc(&column_prefix, sizeof(INDEX) * gc));
  INDEX* grid_particles_arr;
  check_api_error(cudaMalloc(&grid_particles_arr, sizeof(INDEX) * N));

  REAL* avg_rho;
  check_api_error(cudaMalloc(&avg_rho, sizeof(REAL) * totalframe));
  REAL* avg_rho_host = (REAL*)malloc(sizeof(REAL) * totalframe);
  double* timestep_host = (double*)malloc(sizeof(double) * totalframe);

  REAL* rand_host = (REAL*)malloc(sizeof(REAL) * cons_host->N);
  for (int i = 0; i < cons_host->N; i++) rand_host[i] = utils::random_uniform();
  REAL* rand;
  check_api_error(cudaMalloc(&rand, sizeof(REAL) * cons_host->N));
  check_api_error(cudaMemcpy(rand, rand_host, sizeof(REAL) * cons_host->N,
                             cudaMemcpyHostToDevice));

  THREAD_SETTING setting;

  assert(argc == 2);
  std::string logfilename = argv[1];
  utils::Timer timer;
  timer.setTimer();
  double prev = timer.getTimer();

  setting.N_blocks = atoi(argv[1]);
  setting.N_grids = (cons_host->N + setting.N_blocks - 1) / setting.N_blocks;
  setting.gc_blocks = cons_host->grid_count;
  setting.gc_grids = 1;
  setting.gcgc_blocks = setting.N_blocks;
  setting.gcgc_grids = (cons_host->grid_count * cons_host->grid_count +
                        setting.gcgc_blocks - 1) /
                       setting.gcgc_blocks;
  std::cout << "[N] grid dim: " << setting.N_grids
            << " block dim: " << setting.N_blocks << "\n";
  std::cout << "[gc] grid dim: " << setting.gc_grids
            << " block dim: " << setting.gc_blocks << "\n";
  std::cout << "[gcgc] grid dim: " << setting.gcgc_grids
            << " block dim: " << setting.gcgc_blocks << "\n";

  check_launch_error((init<<<setting.N_grids, setting.N_blocks>>>(cons, x, v)));
  check_api_error(cudaDeviceSynchronize());

  neighbor_update(x, grid_num_particles, grid_prefix, grid_curr, grid_tail,
                  column_prefix, grid_particles_arr, cons, cons_host, setting);

  render_frame(x_host, x, cons_host, 0);

  for (int frame = 0; frame < totalframe; frame++) {
    if ((frame + 1) % 20 == 0) render_frame(x_host, x, cons_host, frame + 1);
    check_api_error(cudaDeviceSynchronize());
    prev = timer.getTimer();
    check_launch_error((prediction<<<setting.N_grids, setting.N_blocks>>>(
        x, x_cache, v, cons)));
    check_launch_error(
        (collision<<<setting.N_grids, setting.N_blocks>>>(x, rand, cons)));
    neighbor_update(x, grid_num_particles, grid_prefix, grid_curr, grid_tail,
                    column_prefix, grid_particles_arr, cons, cons_host,
                    setting);
    for (int sub = 0; sub < substeps; sub++) {
      check_launch_error((rho_integral<<<setting.N_grids, setting.N_blocks>>>(
          rho, x, grid_prefix, grid_tail, grid_particles_arr, cons)));
      solver_substep(x, x_delta, rho, lambdaf, rand, grid_prefix, grid_tail,
                     grid_particles_arr, cons, cons_host, setting);
      check_launch_error(
          (collision<<<setting.N_grids, setting.N_blocks>>>(x, rand, cons)));
    }

    check_launch_error((update_vel<<<setting.N_grids, setting.N_blocks>>>(
        x, v, x_cache, cons)));
    check_launch_error((apply_viscosity<<<setting.N_grids, setting.N_blocks>>>(
        x, v, grid_prefix, grid_tail, grid_particles_arr, cons)));

    check_launch_error((rho_get_avg<<<setting.N_grids, setting.N_blocks>>>(
        rho, &avg_rho[frame])));
    check_api_error(cudaDeviceSynchronize());

    timestep_host[frame] = timer.getTimer() - prev;
    prev = timer.getTimer();
  }
  check_api_error(cudaMemcpy(avg_rho_host, avg_rho, sizeof(REAL) * totalframe,
                             cudaMemcpyDeviceToHost));
  for (int i = 0; i < totalframe; i++) avg_rho_host[i] /= cons_host->N;
  utils::writeArrayToFile("data/rho_log.txt", avg_rho_host, totalframe);
  utils::writeArrayToFile("data/timestep_log.txt", timestep_host, totalframe);

  double totaltime = timer.endTimer("pbf fluids");
  std::cout << "avg time per frame " << totaltime / totalframe << std::endl;
  return 0;
}
