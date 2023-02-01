import taichi as ti
import math
import matplotlib.pyplot as plt

ti.init(arch=ti.cuda)

SIGMA2D = 40.0 / 7.0 / math.pi


@ti.func
def sph_kernel2d(r: float, h: float):
  q = r / h
  result = 0.0
  if 0.0 <= q and q <= 0.5:
    result = (6.0 * (q - 1.0) * ti.pow(q, 2.0) + 1.0)
  elif q <= 1.0:
    result = 2.0 * ti.pow((1.0 - q), 3.0)
  else:
    result = 0.0
  return result * SIGMA2D / ti.pow(h, 2)


@ti.func
def sph_kernel2d_diff(r: float, h: float):
  q = r / h
  result = 0.0
  if 0.0 <= q and q <= 0.5:
    result = 6.0 * q * (3 * q - 2)
  elif q <= 1.0:
    result = -6.0 * ti.pow(1 - q, 2)
  else:
    result = 0.0
  return result * SIGMA2D / ti.pow(h, 2) / h


n = 50
N = n * n
scale = 0.6
r = scale / n / 2.0
h = r * 4.1
h_sqr = h * h
cell_size = h * 1.01

standard_rho = 1000.0
mass = scale * scale * standard_rho / N
g = ti.Vector([0.0, -1.0])
fps = 60
dt = 1.0 / fps
substeps = 50
epsilon = 100.0

scorr_k = 3e-2
scorr_n = 4.0
scorr_frac = 0.2
visc_c = 2e-5

grid_count = int(1.0 / cell_size) + 1

max_neighbors = 64
max_particle_in_cell = 64

grid_num_particles = ti.field(dtype=ti.i32, shape=(grid_count, grid_count))
grid_prefix = ti.field(dtype=ti.i32, shape=(grid_count, grid_count))
grid_tail = ti.field(dtype=ti.i32, shape=(grid_count, grid_count))
grid_curr = ti.field(dtype=ti.i32, shape=(grid_count, grid_count))
column_prefix = ti.field(dtype=ti.i32, shape=grid_count)
grid_particles_arr = ti.field(dtype=ti.i32, shape=N)

particle_num_neighbours = ti.field(dtype=ti.i32, shape=N)
particle_neighbours = ti.field(dtype=ti.i32, shape=(N, max_neighbors))

x = ti.Vector.field(2, dtype=ti.f32, shape=N)
x_delta = ti.Vector.field(2, dtype=ti.f32, shape=N)
x_cache = ti.Vector.field(2, dtype=ti.f32, shape=N)
v = ti.Vector.field(2, dtype=ti.f32, shape=N)
p = ti.field(dtype=ti.f32, shape=N)
rho = ti.field(dtype=ti.f32, shape=N)
lambdaf = ti.field(dtype=ti.f32, shape=N)
avgrho = ti.field(dtype=ti.f32, shape=())
varrho = ti.field(dtype=ti.f32, shape=())
per_part_color = ti.Vector.field(3, dtype=ti.f32, shape=N)
per_part_color.fill(ti.Vector([1.0, 1.0, 1.0]))


@ti.kernel
def init():
  for k in range(N):
    i = int(k / n) / float(n)
    j = int(k % n) / float(n)
    px = 0.3 + scale * i
    py = 0.2 + scale * j
    x[k] = ti.Vector([px, py])


@ti.kernel
def prediciton():
  for k in range(N):
    x_cache[k] = x[k]
    x[k] = x[k] + v[k] * dt + g * dt * dt


@ti.kernel
def update_vel():
  for k in range(N):
    v[k] = (x[k] - x_cache[k]) / dt
    #x_cache[k] = x[k]


@ti.kernel
def apply_viscosity():
  for i in x:
    p_i = x[i]
    delta_v = ti.Vector([0.0, 0.0])
    for k in range(particle_num_neighbours[i]):
      j = particle_neighbours[i, k]
      p_j = x[j]
      delta_v += (v[i] - v[j]) * sph_kernel2d((p_i - p_j).norm(), h)
    v[i] = v[i] - delta_v * visc_c


@ti.kernel
def neighbor_summary():
  max_neighbor_count = 0
  avg_neighbor_count = 0.0
  max_grid_count = 0
  avg_grid_count = 0.0

  for i in x:
    ti.atomic_max(max_neighbor_count, particle_num_neighbours[i])
    avg_neighbor_count += ti.cast(particle_num_neighbours[i], ti.f32)

  for i, j in grid_num_particles:
    ti.atomic_max(max_grid_count, grid_num_particles[i, j])
    avg_grid_count += ti.cast(grid_num_particles[i, j], ti.f32)

  print("============ neighbor status summary ============")
  print("Max Neighbors:", max_neighbor_count)
  print("Average Neighbors:", avg_neighbor_count / ti.cast(N, ti.f32))
  print("Max Particle in Cells: ", max_grid_count)
  print("Average Particle in Cells: ", avg_grid_count / ti.cast(N, ti.f32))


@ti.kernel
def preupdate():
  pass


@ti.func
def get_grid_pos(input_pos):
  return ti.cast(input_pos / cell_size, int)


@ti.func
def is_in_grid(index):
  return (index[0] >= 0 and index[0] < grid_count and index[1] >= 0 and
          index[1] < grid_count)


bound_epsilon = 1e-5


@ti.kernel
def collision():
  for i in x:
    if x[i][0] < r:
      x[i][0] = r + bound_epsilon * ti.random()
    if x[i][0] > 1.0 - r:
      x[i][0] = (1.0 - r) - bound_epsilon * ti.random()
    if x[i][1] < r:
      x[i][1] = r + bound_epsilon * ti.random()
    if x[i][1] > 1.0 - r:
      x[i][1] = (1.0 - r) - bound_epsilon * ti.random()


@ti.kernel
def dem_update():
  grid_num_particles.fill(0)
  particle_num_neighbours.fill(0)
  column_prefix.fill(0)
  for i in x:
    cell_index = get_grid_pos(x[i])
    grid_num_particles[cell_index] += 1

  for i, j in grid_num_particles:
    column_prefix[i] += grid_num_particles[i, j]

  grid_prefix[0, 0] = 0
  ti.loop_config(serialize=True)
  for i in range(1, grid_count):
    grid_prefix[i, 0] = grid_prefix[i - 1, 0] + column_prefix[i - 1]

  for i in range(grid_count):
    for j in range(grid_count):
      if j > 0:
        grid_prefix[i, j] = grid_prefix[i, j - 1] + grid_num_particles[i, j - 1]
      grid_tail[i, j] = grid_prefix[i, j] + grid_num_particles[i, j]
      grid_curr[i, j] = grid_prefix[i, j]

  for i in x:
    cell_index = get_grid_pos(x[i])
    index = ti.atomic_add(grid_curr[cell_index], 1)
    grid_particles_arr[index] = i

  for i in x:
    p_i = x[i]
    cell_index = get_grid_pos(p_i)
    n_count = 0
    for offsets in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
      cell_n = cell_index + offsets
      if is_in_grid(cell_n):
        for k in range(grid_prefix[cell_n], grid_tail[cell_n]):
          j = grid_particles_arr[k]
          if i != j and n_count < max_neighbors:
            p_j = x[j]
            if (p_i - p_j).norm_sqr() < h_sqr:
              particle_neighbours[i, n_count] = j
              n_count += 1
    particle_num_neighbours[i] = n_count


@ti.kernel
def get_avg_rho():
  avgrho[None] = 0.0
  Nf = ti.cast(N, ti.f32)
  for i in rho:
    avgrho[None] += rho[i] / Nf

  varrho[None] = 0.0
  for i in rho:
    diff = avgrho[None] - rho[i]
    varrho[None] += diff * diff / Nf


@ti.kernel
def rho_integral():
  rho.fill(0.0)
  for i in x:
    p_i = x[i]
    for k in range(particle_num_neighbours[i]):
      j = particle_neighbours[i, k]
      p_j = x[j]
      p = p_i - p_j
      rho[i] += sph_kernel2d(p.norm(), h) * mass
    rho[i] += sph_kernel2d(0.0, h) * mass


@ti.kernel
def compute_lambda():
  for i in x:
    C_i = rho[i] / standard_rho - 1.0
    dev_sum = 0.0
    dev_i = ti.Vector([0.0, 0.0])
    p_i = x[i]
    for k in range(particle_num_neighbours[i]):
      j = particle_neighbours[i, k]
      p_j = x[j]
      p = p_i - p_j
      p_norm = p.norm()
      if p_norm < 1e-8:
        p = p + bound_epsilon * ti.Vector([ti.random(), ti.random()])
        p_norm = p.norm()
      p_dir = p.normalized()
      if ti.math.isnan(p_dir[0]) or ti.math.isnan(p_dir[1]):
        print('get nan! ', i, p_i, j, p_j)
      dev_j = -sph_kernel2d_diff(p_norm, h) * p_dir / standard_rho
      dev_i += -dev_j
      dev_sum += dev_j.norm_sqr()
    dev_sum += dev_i.norm_sqr()
    lambdaf[i] = -C_i / (dev_sum + epsilon)


@ti.kernel
def compute_delta_pos():
  for i in x:
    x_delta[i] = ti.Vector([0.0, 0.0])
    p_i = x[i]
    for k in range(particle_num_neighbours[i]):
      j = particle_neighbours[i, k]
      p_j = x[j]
      p = p_i - p_j
      if p.norm() < 1e-8:
        p = p + bound_epsilon * ti.Vector([ti.random(), ti.random()])
      dev = sph_kernel2d_diff(p.norm(), h) * p.normalized()
      w = sph_kernel2d(p.norm(), h) / sph_kernel2d(h * scorr_frac, h)
      scorr = -ti.pow(scorr_k * w, scorr_n)
      x_delta[i] += (lambdaf[i] + lambdaf[j] + scorr) * dev / standard_rho

  for i in x:
    x[i] = x[i] + x_delta[i]


nan_count = ti.field(dtype=ti.f32, shape=())


@ti.kernel
def check_nan():
  nan_count[None] = 0
  for i in x:
    if ti.math.isnan(x[i][0]) or ti.math.isnan(x[i][1]):
      nan_count[None] += 1


def nan_debug(prase: str):
  check_nan()
  if nan_count[None] > 0:
    print(f"nan particles {nan_count[None]} in {prase}")
    quit()


def plot_density_distribution():
  rhonp = rho.to_numpy()
  plt.hist(rhonp, bins=50)
  plt.show()


init()
dem_update()
rho_integral()
get_avg_rho()

print("============ solver summary ============")
print("Number of Particles: ", N)
print("Mass: ", mass * N)
print("Mass per particle: ", mass)
print("Radius: ", r)
print("Smooth Length: ", h)
print("Cell Size: ", cell_size)
print("Grid Size: ", grid_count)
print("Expect particle per cell: ", cell_size / r)
print("Expect max neighbors: ", (2 * h / r)**2)
print("Expected density: ", mass / (4.0 * r * r))
print("Initial average density: ", avgrho[None])
print("Initial density variance: ", varrho[None])

neighbor_summary()

window = ti.ui.Window('pbf fluids', res=(800, 800), vsync=True)
canvas = window.get_canvas()

avg_rho_frames = []
var_rho_frames = []

while window.running:

  preupdate()
  prediciton()
  collision()

  dem_update()

  for _ in range(substeps):
    rho_integral()
    compute_lambda()
    compute_delta_pos()
    collision()

  get_avg_rho()
  avg_rho_frames.append(avgrho[None])
  var_rho_frames.append(varrho[None])

  update_vel()
  apply_viscosity()

  canvas.set_background_color(color=(0.0, 0.0, 0.0))
  canvas.circles(x, color=(0.7, 0.3, 0.2), radius=r)
  #canvas.circles(x, per_vertex_color=per_part_color, radius=r)
  window.show()
window.destroy()

plt.plot(avg_rho_frames)
plt.show()
plt.plot(var_rho_frames)
plt.show()