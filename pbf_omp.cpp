#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>

#include <iostream>
#include <random>
#include <functional>
#include <string>
#include <chrono>
#include <sstream>
#include <memory>
#include <iomanip>
#include <vector>
#include <immintrin.h>

#include "utils.h"

typedef float REAL;
typedef uint16_t INDEX;

const REAL SIGMA2D = 40.0 / 7.0 / M_PI;
const REAL scale = 0.6f;
const int n = 64;
const int N = n * n;
const REAL r = scale / 2.0f / static_cast<REAL>(n);
const REAL h = r * 4.1f;
const REAL h_sqr = h * h;
const REAL cell_size = h * 1.085f;

const REAL standard_rho = 1000.0f;
const REAL mass = scale * scale * standard_rho / static_cast<REAL>(N);
std::vector<REAL> g = {0.0, -1.0};
std::vector<REAL> g_vec = {0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0,
                           0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0};

const int fps = 60;
const REAL dt = 1.0f / static_cast<REAL>(fps);
const int substeps = 50;
const REAL epsilon = 100.0f;
const REAL bound_epsilon = 1e-5f;

const REAL scorr_k = 3e-2f;
const REAL scorr_n = 4.0f;
const REAL scorr_frac = 0.2f;
const REAL visc_c = 2e-5f;

const int grid_count = static_cast<int>(1.0 / cell_size) + 1;

std::vector<INDEX> grid_num_particles(grid_count* grid_count);
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

inline void vec_set(REAL* a, int anr, REAL value1, REAL value2) {
  a[anr * 2] = value1;
  a[anr * 2 + 1] = value2;
}

inline void vec_cpy(REAL* a, int anr, REAL* b, int bnr, REAL mul = 1.0) {
  a[anr * 2] = b[bnr * 2] * mul;
  a[anr * 2 + 1] = b[bnr * 2 + 1] * mul;
}

inline void vec_cpy_8(REAL* a, int anr, REAL* b, int bnr, REAL mul = 1.0) {
  __m512 b_vec = _mm512_loadu_ps(b + bnr * 2);
  __m512 mul_vec = _mm512_set1_ps(mul);
  _mm512_storeu_ps(a + anr * 2, _mm512_mul_ps(b_vec, mul_vec));
}

inline void vec_add(REAL* a, int anr, REAL* b, int bnr, REAL mul = 1.0) {
  a[anr * 2] += b[bnr * 2] * mul;
  a[anr * 2 + 1] += b[bnr * 2 + 1] * mul;
}

inline void vec_add_8(REAL* a, int anr, REAL* b, int bnr, REAL mul = 1.0) {
  __m512 a_vec = _mm512_loadu_ps(a + anr * 2);
  __m512 b_vec = _mm512_loadu_ps(b + bnr * 2);
  __m512 mul_vec = _mm512_set1_ps(mul);
  _mm512_storeu_ps(a + anr * 2,
                   _mm512_add_ps(a_vec, _mm512_mul_ps(b_vec, mul_vec)));
}

inline void vec_scale(REAL* a, int anr, REAL mul) {
  a[anr * 2] *= mul;
  a[anr * 2 + 1] *= mul;
}

inline void vec_set_add(REAL* dst, int dnr, REAL* a, int anr, REAL* b, int bnr,
                        REAL mul = 1.0) {
  dst[dnr * 2] = (a[anr * 2] + b[bnr * 2]) * mul;
  dst[dnr * 2 + 1] = (a[anr * 2 + 1] + b[bnr * 2 + 1]) * mul;
}

inline void vec_set_add_8(REAL* dst, int dnr, REAL* a, int anr, REAL* b,
                          int bnr, REAL mul = 1.0) {
  __m512 a_vec = _mm512_loadu_ps(a + anr * 2);
  __m512 b_vec = _mm512_loadu_ps(b + bnr * 2);
  __m512 mul_vec = _mm512_set1_ps(mul);
  _mm512_storeu_ps(dst + dnr * 2,
                   _mm512_mul_ps(_mm512_add_ps(a_vec, b_vec), mul_vec));
}

inline void vec_set_sub(REAL* dst, int dnr, REAL* a, int anr, REAL* b, int bnr,
                        REAL mul = 1.0) {
  dnr *= 2;
  anr *= 2;
  bnr *= 2;
  dst[dnr++] = (a[anr++] - b[bnr++]) * mul;
  dst[dnr] = (a[anr] - b[bnr]) * mul;
}

inline void vec_set_sub_8(REAL* dst, int dnr, REAL* a, int anr, REAL* b,
                          int bnr, REAL mul = 1.0) {
  __m512 a_vec = _mm512_loadu_ps(a + anr * 2);
  __m512 b_vec = _mm512_loadu_ps(b + bnr * 2);
  __m512 mul_vec = _mm512_set1_ps(mul);
  _mm512_storeu_ps(dst + dnr * 2,
                   _mm512_mul_ps(_mm512_sub_ps(a_vec, b_vec), mul_vec));
}

inline REAL vec_dot(REAL* a, int anr, REAL* b, int bnr) {
  anr *= 2;
  bnr *= 2;
  return a[anr] * b[bnr] + a[anr + 1] * b[bnr + 1];
}

inline REAL vec_norm_sqr(REAL* a, int anr) {
  anr *= 2;
  return a[anr] * a[anr] + a[anr + 1] * a[anr + 1];
}

inline REAL vec_norm(REAL* a, int anr) {
  anr *= 2;
  return sqrtf(a[anr] * a[anr] + a[anr + 1] * a[anr + 1]);
}

inline REAL vec_dist_sqr(REAL* a, int anr, REAL* b, int bnr) {
  return (a[anr * 2] - b[bnr * 2]) * (a[anr * 2] - b[bnr * 2]) +
         (a[anr * 2 + 1] - b[bnr * 2 + 1]) * (a[anr * 2 + 1] - b[bnr * 2 + 1]);
}

inline REAL vec_dist(REAL* a, int anr, REAL* b, int bnr) {
  return sqrtf((a[anr * 2] - b[bnr * 2]) * (a[anr * 2] - b[bnr * 2]) +
               (a[anr * 2 + 1] - b[bnr * 2 + 1]) *
                   (a[anr * 2 + 1] - b[bnr * 2 + 1]));
}

inline void print_vec(REAL* a, int anr) {
  std::cout << a[anr * 2] << " " << a[anr * 2 + 1];
}

inline REAL random_uniform() {
  static std::uniform_real_distribution<REAL> distribution(0.0, 1.0);
  static std::mt19937 generator;
  return distribution(generator);
}

inline void pos_to_grid(REAL* x, int xnr, int& gridx, int& gridy) {
  gridx = static_cast<int>(x[xnr * 2] / cell_size);
  gridy = static_cast<int>(x[xnr * 2 + 1] / cell_size);
}

inline void get_grid_xy(const int grid_id, int& gridx, int& gridy) {
  gridx = grid_id / grid_count;
  gridy = grid_id * grid_count;
}

inline int get_grid_id(int gridx, int gridy) {
  return gridx * grid_count + gridy;
}

inline REAL sphkernel2d(float r, float h) {
  REAL q = r / h;
  REAL result = 0.0f;
  if (q >= 0.0f && q <= 0.5f)
    result = (6.0 * (q - 1.0f) * q * q + 1.0f);
  else if (q <= 1.0f)
    result = 2.0f * (1.0f - q) * (1.0f - q) * (1.0f - q);
  return result * SIGMA2D / (h * h);
}

inline REAL sphkernel2d_diff(float r, float h) {
  REAL q = r / h;
  REAL result = 0.0f;
  if (q >= 0.0f && q <= 0.5f)
    result = 6.0f * q * (3.0f * q - 2.0f);
  else if (q <= 1.0f)
    result = -6.0f * (1.0f - q) * (1.0f - q);
  return result * SIGMA2D / (h * h * h);
}

void init() {
#pragma omp parallel for
  for (int k = 0; k < N; k++) {
    int i = k / n;
    int j = k % n;
    REAL px = 0.3f + scale * static_cast<float>(i) / static_cast<float>(n);
    REAL py = 0.2f + scale * static_cast<float>(j) / static_cast<float>(n);
    vec_set(x.data(), k, px, py);
    vec_set(v.data(), k, 0.0f, 0.0f);
  }
}

void prediction() {
#pragma omp parallel for
  for (int k = 0; k < N; k++) {
    vec_cpy(x_cache.data(), k, x.data(), k);
    vec_add(v.data(), k, g.data(), 0, dt);
    vec_add(x.data(), k, v.data(), k, dt);
  }

  /*for (int k = 0; k < N; k += 8) {
    vec_cpy_8(x_cache.data(), k, x.data(), k);
    vec_add_8(v.data(), k, g_vec.data(), 0, dt);
    vec_add_8(x.data(), k, v.data(), k, dt);
  }*/
}

void update_vel() {
#pragma omp parallel for
  for (int k = 0; k < N; k += 8) {
    vec_set_sub_8(v.data(), k, x.data(), k, x_cache.data(), k, 1.0 / dt);
  }
}

void collision() {
#pragma omp parallel for
  for (int k = 0; k < N; k++) {
    if (x[k * 2] < r) x[k * 2] = r + bound_epsilon * random_uniform();
    if (x[k * 2] > 1.0 - r)
      x[k * 2] = 1.0 - r - bound_epsilon * random_uniform();
    if (x[k * 2 + 1] < r) x[k * 2 + 1] = r + bound_epsilon * random_uniform();
    if (x[k * 2 + 1] > 1.0 - r)
      x[k * 2 + 1] = 1.0 - r - bound_epsilon * random_uniform();
  }
}

void neighbor_update() {
  memset(grid_num_particles.data(), 0, sizeof(INDEX) * grid_count * grid_count);
  memset(column_prefix.data(), 0, sizeof(INDEX) * grid_count);

  INDEX* gnp = grid_num_particles.data();

#pragma omp parallel for reduction(+ : gnp[:grid_count * grid_count])
  for (int i = 0; i < N; i++) {
    int gridx, gridy;
    pos_to_grid(x.data(), i, gridx, gridy);
    assert(0 <= gridx && gridx < grid_count);
    assert(0 <= gridy && gridy < grid_count);
    gnp[get_grid_id(gridx, gridy)] += 1;
  }

  INDEX* cp = column_prefix.data();
#pragma omp parallel for reduction(+ : cp[:grid_count])
  for (int i = 0; i < grid_count * grid_count; i++) {
    cp[i / grid_count] += grid_num_particles[i];
  }

  grid_prefix[0] = 0;
  for (int i = 1; i < grid_count; i++)
    grid_prefix[i * grid_count] =
        grid_prefix[(i - 1) * grid_count] + column_prefix[i - 1];

#pragma omp parallel for
  for (int i = 0; i < grid_count; i++) {
    for (int j = 0; j < grid_count; j++) {
      int index = i * grid_count + j;
      if (j > 0)
        grid_prefix[index] =
            grid_prefix[index - 1] + grid_num_particles[index - 1];
      grid_tail[index] = grid_prefix[index] + grid_num_particles[index];
      grid_curr[index] = grid_prefix[index];
    }
  }

  for (int i = 0; i < N; i++) {
    int gridx, gridy;
    pos_to_grid(x.data(), i, gridx, gridy);
    int index = grid_curr[gridx * grid_count + gridy];
    grid_curr[gridx * grid_count + gridy] += 1;
    grid_particls_arr[index] = i;
  }
}

template <typename Callable>
void iterate_neighbor(int part_id, Callable func) {
  int gridx_i, gridy_i;
  pos_to_grid(x.data(), part_id, gridx_i, gridy_i);
  int rangex0 = utils::max(0, gridx_i - 1);
  int rangex1 = utils::min(grid_count - 1, gridx_i + 1);
  int rangey0 = utils::max(0, gridy_i - 1);
  int rangey1 = utils::min(grid_count - 1, gridy_i + 1);
  for (int gridx_j = rangex0; gridx_j <= rangex1; gridx_j++)
    for (int gridy_j = rangey0; gridy_j <= rangey1; gridy_j++) {
      int grid_index = get_grid_id(gridx_j, gridy_j);
      for (int k = grid_prefix[grid_index]; k < grid_tail[grid_index]; k++) {
        int j = grid_particls_arr[k];
        if (j != part_id &&
            vec_dist_sqr(x.data(), part_id, x.data(), j) < h_sqr)
          func(j);
      }
    }
}

void rho_integral() {
  memset(rho.data(), 0.0f, sizeof(REAL) * N);
  REAL* rhop = rho.data();

#pragma omp parallel for reduction(+ : rhop[:N])
  for (int i = 0; i < N; i++) {
    iterate_neighbor(i, [&](int j) {
      REAL dis = vec_dist(x.data(), i, x.data(), j);
      rhop[i] += sphkernel2d(dis, h) * mass;
    });
    rhop[i] += sphkernel2d(0.0, h) * mass;
  }
}

void compute_lambda() {
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    REAL C = rho[i] / standard_rho - 1.0f;
    REAL deriv_sum = 0.0f;
    REAL deriv_i[2];
    vec_set(deriv_i, 0, 0.0, 0.0);
    iterate_neighbor(i, [&](int j) {
      REAL p[2];
      vec_set_sub(p, 0, x.data(), i, x.data(), j);
      REAL p_norm = vec_norm(p, 0);
      if (p_norm < 1e-8f) {
        p[0] += random_uniform() * bound_epsilon;
        p[1] += random_uniform() * bound_epsilon;
        p_norm = vec_norm(p, 0);
      }
      vec_scale(p, 0, 1.0f / p_norm);
      REAL deriv_j[2];
      vec_cpy(deriv_j, 0, p, 0, 1.0f / standard_rho);
      vec_scale(deriv_j, 0, -sphkernel2d_diff(p_norm, h));
      vec_add(deriv_i, 0, deriv_j, 0, -1.0);
      deriv_sum += vec_norm_sqr(deriv_j, 0);
    });
    deriv_sum += vec_norm_sqr(deriv_i, 0);
    lambdaf[i] = -C / (deriv_sum + epsilon);
  }
}

void compute_delta_pos() {
  memset(x_delta.data(), 0.0f, N * 2 * sizeof(REAL));
  REAL* xdp = x_delta.data();

#pragma omp parallel for reduction(+ : xdp[:2 * N])
  for (int i = 0; i < N; i++) {
    iterate_neighbor(i, [&](int j) {
      REAL p[2];
      vec_set_sub(p, 0, x.data(), i, x.data(), j);
      REAL p_norm = vec_norm(p, 0);
      if (p_norm < 1e-8f) {
        p[0] += random_uniform() * bound_epsilon;
        p[1] += random_uniform() * bound_epsilon;
        p_norm = vec_norm(p, 0);
      }
      vec_scale(p, 0, 1.0f / p_norm);
      REAL deriv[2];
      vec_cpy(deriv, 0, p, 0, sphkernel2d_diff(p_norm, h));
      REAL w = sphkernel2d(p_norm, h) / sphkernel2d(h * scorr_frac, h);
      REAL scorr = -powf(scorr_k * w, scorr_n);
      vec_add(xdp, i, deriv, 0,
              (lambdaf[i] + lambdaf[j] + scorr) / standard_rho);
    });
  }
  for (int i = 0; i < N; i++) vec_add(x.data(), i, x_delta.data(), i);
}

void apply_viscosity() {
  REAL* velp = v.data();

#pragma omp parallel for reduction(+ : velp[:2 * N])
  for (int i = 0; i < N; i++) {
    REAL delta_v[2];
    vec_set(delta_v, 0, 0.0, 0.0);
    iterate_neighbor(i, [&](int j) {
      REAL p[2];
      vec_set_sub(p, 0, x.data(), i, x.data(), j);
      REAL p_norm = vec_norm(p, 0);
      REAL w = sphkernel2d(p_norm, h);
      vec_set_sub(p, 0, velp, i, velp, j, w);
      vec_add(delta_v, 0, p, 0);
    });
    vec_add(velp, i, delta_v, 0, -1.0f * visc_c);
  }
}

void get_average_rho() {
  avgrho = 0.0f;
  __m512 avgrho_vec = _mm512_setzero_ps();
  for (int i = 0; i < N; i += 16) {
    __m512 rho_vec = _mm512_loadu_ps(rho.data() + i);
    avgrho_vec = _mm512_add_ps(avgrho_vec, rho_vec);
  }

  for (int i = 0; i < 16; i++) avgrho += avgrho_vec[i];
  avgrho /= N;
}

void write_frame_to_png(const char* filename) {
  utils::GreyPNGWriter img(500, 500);
  img.clearBuffer();
  for (int i = 0; i < N; i++) {
    img.drawOnPos(x[i * 2], x[i * 2 + 1], 255);
  }
  img.writeToFile(filename);
}

template <typename T>
void print_array(T* arr, int len) {
  for (int i = 0; i < len; i++) std::cout << arr[i] << ' ';
  std::cout << std::endl;
}

void write_frame(int frame) {
  std::string filename = "img/" + std::to_string(frame) + ".png";
  write_frame_to_png(filename.c_str());
  std::cout << filename << " writing complete\n" << std::flush;
}

void check_isnan(std::string info) {
  int nan_count = 0;
  for (int i = 0; i < N; i++) {
    if (isnan(x[i * 2]) || isnan(x[i * 2 + 1])) nan_count++;
  }
  if (nan_count > 0)
    std::cout << "nan check failed after " << info << "\n" << std::flush;
  assert(nan_count == 0);
}

void solver_summary() {
  std::cout << "==================== Solver Summary ==================\n";
  std::cout << "Number of particles: " << N << std::endl;
  std::cout << "Mass per particle: " << mass << std::endl;
  std::cout << "Radius: " << r << std::endl;
  std::cout << "Smooth length: " << h << std::endl;
  std::cout << "Cell size: " << cell_size << std::endl;
  std::cout << "Grid size: " << grid_count << 'x' << grid_count << std::endl;
  std::cout << "Expected average density: " << standard_rho << std::endl;
  std::cout << "Initial average density: " << avgrho << std::endl;
}

template <typename T>
void print_avg(std::ofstream& filestream, std::string array_name, T* array,
               int start, int end) {
  T avg = utils::getArrayAverage(array, start, end);
  filestream << array_name << " average: " << avg << std::endl;
}

int main(int argc, char** argv) {
  assert(argc == 2);
  std::string logfilename = argv[1];

  utils::Timer timer;
  timer.setTimer();
  double prev = timer.getTimer();
  utils::Timer inframe_timer;

  init();
  neighbor_update();
  rho_integral();
  get_average_rho();
  solver_summary();

  write_frame(0);

  int totalframe = 120;
  REAL rho_log[totalframe + 1];
  double timestep[totalframe + 1];
  rho_log[0] = avgrho;
  timestep[0] = timer.getTimer() - prev;
  prev = timer.getTimer();

  double prediction_timer[totalframe];
  double collision_timer[totalframe];
  double neighbor_update_timer[totalframe];
  double rho_integral_timer[totalframe];
  double lambda_compute_timer[totalframe];
  double pos_update_timer[totalframe];
  double vel_update_timer[totalframe];
  double viscosity_timer[totalframe];
  int max_part_in_cell[totalframe];

  auto set_time = [&inframe_timer](double* array, int frame, double& prev,
                                   double& curr) {
    curr = inframe_timer.getTimer();
    array[frame] += curr - prev;
    prev = curr;
  };

  for (int frame = 0; frame < totalframe; frame++) {
    if ((frame + 1) % 20 == 0) write_frame(frame + 1);

    prediction_timer[frame] = 0.0;
    collision_timer[frame] = 0.0;
    neighbor_update_timer[frame] = 0.0;
    rho_integral_timer[frame] = 0.0;
    lambda_compute_timer[frame] = 0.0;
    pos_update_timer[frame] = 0.0;
    vel_update_timer[frame] = 0.0;
    viscosity_timer[frame] = 0.0;
    inframe_timer.setTimer();
    double inframe_prev = inframe_timer.getTimer();
    double curr_time = 0.0;

    prediction();
    set_time(prediction_timer, frame, inframe_prev, curr_time);

    collision();
    set_time(collision_timer, frame, inframe_prev, curr_time);

    neighbor_update();
    set_time(neighbor_update_timer, frame, inframe_prev, curr_time);

    for (int sub = 0; sub < substeps; sub++) {
      rho_integral();
      set_time(rho_integral_timer, frame, inframe_prev, curr_time);
      compute_lambda();
      set_time(lambda_compute_timer, frame, inframe_prev, curr_time);
      compute_delta_pos();
      set_time(pos_update_timer, frame, inframe_prev, curr_time);
      collision();
      set_time(collision_timer, frame, inframe_prev, curr_time);
    }

    update_vel();
    set_time(vel_update_timer, frame, inframe_prev, curr_time);
    apply_viscosity();
    set_time(viscosity_timer, frame, inframe_prev, curr_time);
    collision_timer[frame] /= substeps + 1;
    rho_integral_timer[frame] /= substeps;
    lambda_compute_timer[frame] /= substeps;
    pos_update_timer[frame] /= substeps;

    max_part_in_cell[frame] = utils::getMaxValue(grid_num_particles.data(), 0,
                                                 grid_count * grid_count);

    get_average_rho();
    rho_log[frame + 1] = avgrho;
    timestep[frame + 1] =
        timer.getTimer("frame " + std::to_string(frame), prev);
    prev = timer.getTimer();

    inframe_timer.endTimer();
  }

  std::ofstream file;
  file.open("data/" + logfilename + "_datalog.txt");
  print_avg(file, "prediction time cost", prediction_timer, 0, totalframe);
  print_avg(file, "collision time cost", collision_timer, 0, totalframe);
  print_avg(file, "neighbor finding time cost", neighbor_update_timer, 0,
            totalframe);
  print_avg(file, "rho integral time cost", rho_integral_timer, 0, totalframe);
  print_avg(file, "lambda compute time cost", lambda_compute_timer, 0,
            totalframe);
  print_avg(file, "position update time cost", pos_update_timer, 0, totalframe);
  print_avg(file, "velocity update time cost", vel_update_timer, 0, totalframe);
  print_avg(file, "viscosity update time cost", viscosity_timer, 0, totalframe);
  file.close();

  utils::writeArrayToFile("data/rho_log.txt", rho_log, totalframe + 1);
  utils::writeArrayToFile("data/timestep_log.txt", timestep, totalframe + 1);
  utils::writeArrayToFile("data/prediction_log.txt", prediction_timer,
                          totalframe);
  utils::writeArrayToFile("data/collision_log.txt", collision_timer,
                          totalframe);
  utils::writeArrayToFile("data/neighbor_log.txt", neighbor_update_timer,
                          totalframe);
  utils::writeArrayToFile("data/rho_integral_log.txt", rho_integral_timer,
                          totalframe);
  utils::writeArrayToFile("data/lambda_compute_log.txt", lambda_compute_timer,
                          totalframe);
  utils::writeArrayToFile("data/pos_update_log.txt", pos_update_timer,
                          totalframe);
  utils::writeArrayToFile("data/vel_update_log.txt", vel_update_timer,
                          totalframe);
  utils::writeArrayToFile("data/viscosity_log.txt", viscosity_timer,
                          totalframe);
  utils::writeArrayToFile("data/max_part_in_cell_log.txt", max_part_in_cell,
                          totalframe);

  double totaltime = timer.endTimer("pbf fluids");
  std::cout << "avg time per frame " << totaltime / totalframe << std::endl;

  return 0;
}
