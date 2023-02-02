#define _USE_MATH_DEFINES
#include <math.h>

#include <iostream>
#include <random>
#include <functional>
#include <string>
#include <chrono>
#include <sstream>
#include <memory>
#include <iomanip>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

typedef float REAL;
typedef uint16_t INDEX;

const REAL SIGMA2D = 40.0 / 7.0 / M_PI;
const REAL scale = 0.6f;
const int n = 64;
const int N = n * n;
const REAL r = scale / 2.0f / static_cast<REAL>(n);
const REAL h = r * 4.1f;
const REAL h_sqr = h * h;
const REAL cell_size = h * 1.01f;

const REAL standard_rho = 1000.0f;
const REAL mass = scale * scale * standard_rho / static_cast<REAL>(N);
REAL* g = new REAL[2];

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

INDEX* grid_num_particles = new INDEX[grid_count * grid_count];
INDEX* grid_prefix = new INDEX[grid_count * grid_count];
INDEX* grid_tail = new INDEX[grid_count * grid_count];
INDEX* grid_curr = new INDEX[grid_count * grid_count];
INDEX* column_prefix = new INDEX[grid_count];
INDEX* grid_particls_arr = new INDEX[N];

REAL* x = new REAL[N * 2];
REAL* x_delta = new REAL[N * 2];
REAL* x_cache = new REAL[N * 2];
REAL* v = new REAL[N * 2];
REAL* p = new REAL[N];
REAL* rho = new REAL[N];
REAL* lambdaf = new REAL[N];
REAL avgrho;

template <typename T>
inline T min(T a, T b) {
  return a < b ? a : b;
}

template <typename T>
inline T max(T a, T b) {
  return a > b ? a : b;
}

inline void vec_set(REAL* a, int anr, REAL value1, REAL value2) {
  a[anr * 2] = value1;
  a[anr * 2 + 1] = value2;
}

inline void vec_cpy(REAL* a, int anr, REAL* b, int bnr, REAL mul = 1.0) {
  a[anr * 2] = b[bnr * 2] * mul;
  a[anr * 2 + 1] = b[bnr * 2 + 1] * mul;
}

inline void vec_add(REAL* a, int anr, REAL* b, int bnr, REAL mul = 1.0) {
  a[anr * 2] += b[bnr * 2] * mul;
  a[anr * 2 + 1] += b[bnr * 2 + 1] * mul;
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

inline void vec_set_sub(REAL* dst, int dnr, REAL* a, int anr, REAL* b, int bnr,
                        REAL mul = 1.0) {
  dnr *= 2;
  anr *= 2;
  bnr *= 2;
  dst[dnr++] = (a[anr++] - b[bnr++]) * mul;
  dst[dnr] = (a[anr] - b[bnr]) * mul;
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
  g[0] = 0.0f;
  g[1] = -1.0f;
  for (int k = 0; k < N; k++) {
    int i = k / n;
    int j = k % n;
    REAL px = 0.3f + scale * static_cast<float>(i) / static_cast<float>(n);
    REAL py = 0.2f + scale * static_cast<float>(j) / static_cast<float>(n);
    vec_set(x, k, px, py);
    vec_set(v, k, 0.0f, 0.0f);
  }
}

void prediction() {
  for (int k = 0; k < N; k++) {
    vec_cpy(x_cache, k, x, k);
    vec_add(v, k, g, 0, dt);
    vec_add(x, k, v, k, dt);
  }
}

void update_vel() {
  for (int k = 0; k < N; k++) {
    vec_set_sub(v, k, x, k, x_cache, k, 1.0 / dt);
  }
}

void collision() {
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
  memset(grid_num_particles, 0, sizeof(INDEX) * grid_count * grid_count);
  memset(column_prefix, 0, sizeof(INDEX) * grid_count);

  for (int i = 0; i < N; i++) {
    int gridx, gridy;
    pos_to_grid(x, i, gridx, gridy);
    grid_num_particles[get_grid_id(gridx, gridy)] += 1;
  }

  for (int i = 0; i < grid_count * grid_count; i++) {
    column_prefix[i / grid_count] += grid_num_particles[i];
  }

  grid_prefix[0] = 0;
  for (int i = 1; i < grid_count; i++)
    grid_prefix[i * grid_count] =
        grid_prefix[(i - 1) * grid_count] + column_prefix[i - 1];

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
    pos_to_grid(x, i, gridx, gridy);
    int index = grid_curr[gridx * grid_count + gridy];
    grid_curr[gridx * grid_count + gridy] += 1;
    grid_particls_arr[index] = i;
  }
}

template <typename Callable>
void iterate_neighbor(int part_id, Callable func) {
  int gridx_i, gridy_i;
  pos_to_grid(x, part_id, gridx_i, gridy_i);
  int rangex0 = max(0, gridx_i - 1);
  int rangex1 = min(grid_count - 1, gridx_i + 1);
  int rangey0 = max(0, gridy_i - 1);
  int rangey1 = min(grid_count - 1, gridy_i + 1);
  for (int gridx_j = rangex0; gridx_j <= rangex1; gridx_j++)
    for (int gridy_j = rangey0; gridy_j <= rangey1; gridy_j++) {
      int grid_index = get_grid_id(gridx_j, gridy_j);
      for (int k = grid_prefix[grid_index]; k < grid_tail[grid_index]; k++) {
        int j = grid_particls_arr[k];
        if (j != part_id && vec_dist_sqr(x, part_id, x, j) < h_sqr) func(j);
      }
    }
}

void rho_integral() {
  memset(rho, 0.0f, sizeof(REAL) * N);
  for (int i = 0; i < N; i++) {
    iterate_neighbor(i, [&](int j) {
      REAL dis = vec_dist(x, i, x, j);
      rho[i] += sphkernel2d(dis, h) * mass;
    });
    rho[i] += sphkernel2d(0.0, h) * mass;
  }
}

void compute_lambda() {
  for (int i = 0; i < N; i++) {
    REAL C = rho[i] / standard_rho - 1.0f;
    REAL deriv_sum = 0.0f;
    REAL* deriv_i = new REAL[2];
    iterate_neighbor(i, [&](int j) {
      REAL* p = new REAL[2];
      vec_set_sub(p, 0, x, i, x, j);
      REAL p_norm = vec_norm(p, 0);
      if (p_norm < 1e-8f) {
        p[0] += random_uniform() * bound_epsilon;
        p[1] += random_uniform() * bound_epsilon;
        p_norm = vec_norm(p, 0);
      }
      vec_scale(p, 0, 1.0f / p_norm);
      REAL* deriv_j = new REAL[2];
      vec_cpy(deriv_j, 0, p, 0, 1.0f / standard_rho);
      vec_scale(deriv_j, 0, -sphkernel2d_diff(p_norm, h));
      vec_add(deriv_i, 0, deriv_j, 0, -1.0);
      deriv_sum += vec_norm_sqr(deriv_j, 0);
      delete[] deriv_j;
    });
    deriv_sum += vec_norm_sqr(deriv_i, 0);
    lambdaf[i] = -C / (deriv_sum + epsilon);
    delete[] deriv_i;
  }
}

void compute_delta_pos() {
  memset(x_delta, 0.0f, N * 2 * sizeof(REAL));
  for (int i = 0; i < N; i++) {
    iterate_neighbor(i, [&](int j) {
      REAL* p = new REAL[2];
      vec_set_sub(p, 0, x, i, x, j);
      REAL p_norm = vec_norm(p, 0);
      if (p_norm < 1e-8f) {
        p[0] += random_uniform() * bound_epsilon;
        p[1] += random_uniform() * bound_epsilon;
        p_norm = vec_norm(p, 0);
      }
      vec_scale(p, 0, 1.0f / p_norm);
      REAL* deriv = new REAL[2];
      vec_cpy(deriv, 0, p, 0, sphkernel2d_diff(p_norm, h));
      REAL w = sphkernel2d(p_norm, h) / sphkernel2d(h * scorr_frac, h);
      REAL scorr = -powf(scorr_k * w, scorr_n);
      vec_add(x_delta, i, deriv, 0,
              (lambdaf[i] + lambdaf[j] + scorr) / standard_rho);
      delete[] p;
      delete[] deriv;
    });
  }
  for (int i = 0; i < N; i++) vec_add(x, i, x_delta, i);
}

void apply_viscosity() {
  for (int i = 0; i < N; i++) {
    REAL* delta_v = new REAL[2];
    vec_set(delta_v, 0, 0.0, 0.0);
    iterate_neighbor(i, [&](int j) {
      REAL* p = new REAL[2];
      vec_set_sub(p, 0, x, i, x, j);
      REAL p_norm = vec_norm(p, 0);
      REAL w = sphkernel2d(p_norm, h);
      vec_set_sub(p, 0, v, i, v, j, w);
      vec_add(delta_v, 0, p, 0);
      delete[] p;
    });
    vec_add(v, i, delta_v, 0, -1.0f * visc_c);
    delete[] delta_v;
  }
}

void get_average_rho() {
  avgrho = 0.0f;
  for (int i = 0; i < N; i++) avgrho += rho[i];
  avgrho /= N;
}

const int img_size = 500;
unsigned char* img = new unsigned char[img_size * img_size];

void write_frame_to_png(const char* filename) {
  const REAL fsize = static_cast<REAL>(img_size);
  memset(img, 0, sizeof(unsigned char) * img_size * img_size);
  for (int i = 0; i < img_size * img_size; i++) img[i] = 0;
  for (int i = 0; i < N; i++) {
    int xpos = static_cast<int>(x[i * 2] * fsize);
    int ypos = static_cast<int>((1.0f - x[i * 2 + 1]) * fsize);
    img[ypos * img_size + xpos] = 255;
  }
  stbi_write_png(filename, img_size, img_size, 1, img,
                 sizeof(unsigned char) * img_size);
}

template <typename T>
void print_array(T* arr, int len) {
  for (int i = 0; i < len; i++) std::cout << arr[i] << ' ';
  std::cout << std::endl;
}

void write_frame(int frame) {
  std::string filename = std::to_string(frame) + ".png";
  write_frame_to_png(filename.c_str());
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

class Timer {
 public:
  void setTimer() {
    if (m_base_timer != nullptr) {
      std::cerr << "Warning! Overwriting existing timer\n";
      m_base_timer = nullptr;
    }
    m_base_timer = std::make_unique<BaseTimer>(
        BaseTimer{std::chrono::system_clock::now()});
  }

  double getTimer(std::string message = "") {
    if (m_base_timer == nullptr) {
      std::cerr << "Error! No timer initialized\n";
      return -1.0;
    } else {
      const auto time_in_microsec = get_elapsed_time();
      const auto time_in_millisec = time_in_microsec / 1000.0;
      const auto time_in_second = time_in_millisec / 1000.0;
      std::ostringstream sstream;
      sstream << "[ " << std::setw(max<int>(30, message.length())) << message
              << " :\t";
      if (time_in_second > 1.0)
        sstream << std::setw(8) << time_in_second << "s \t]" << std::endl;
      else
        sstream << std::setw(8) << time_in_millisec << "ms\t]" << std::endl;
      std::cout << sstream.str();
      return time_in_second;
    }
  }

  double endTimer(std::string message = "") {
    double t = getTimer(message);
    m_base_timer = nullptr;
    return t;
  }

  int64_t get_elapsed_time() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::system_clock::now() - m_base_timer->m_start_time)
        .count();
  }

 private:
  typedef struct {
    std::chrono::system_clock::time_point m_start_time;
  } BaseTimer;

  std::unique_ptr<BaseTimer> m_base_timer = nullptr;
};

int main() {
  Timer timer;

  init();
  neighbor_update();
  rho_integral();
  get_average_rho();
  solver_summary();

  timer.setTimer();

  write_frame(0);

  int totalframe = 20;
  for (int frame = 0; frame < totalframe; frame++) {
    if ((frame + 1) % 20 == 0) write_frame(frame + 1);

    prediction();
    collision();
    neighbor_update();

    for (int sub = 0; sub < substeps; sub++) {
      rho_integral();
      compute_lambda();
      compute_delta_pos();
      collision();
    }

    update_vel();
    apply_viscosity();
    timer.getTimer("frame " + std::to_string(frame));
  }

  double totaltime = timer.endTimer("pbf fluids");
  std::cout << "avg time per frame " << totaltime / totalframe << std::endl;

  return 0;
}
