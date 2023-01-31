import taichi as ti
import numpy as np
import math

ti.init(arch=ti.cpu)

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


n = 100
N = n * n

r = 0.1 / n
smooth = ti.field(dtype=ti.f32, shape=())
smooth[None] = r * 4.0
mass = 0.16 * 1000.0 / N
rho = ti.field(dtype=ti.f32, shape=())
rho[None] = mass

x = ti.Vector.field(2, dtype=ti.f32, shape=N)
center = ti.Vector.field(2, dtype=ti.f32, shape=1)
center[0] = ti.Vector([0.5, 0.5])

dx = 1.0 / n


@ti.kernel
def init():
  for k in range(N):
    i = int(k / n) * dx
    j = int(k % n) * dx
    px = 0.3 + 0.4 * i
    py = 0.4 + 0.4 * j
    x[k] = ti.Vector([px, py])


@ti.kernel
def update_radius(x: ti.f32, y: ti.f32):
  p = ti.Vector([x, y])
  smooth[None] = (p - center[0]).norm()


@ti.kernel
def integrate_density():
  rho[None] = 0.0
  for i in x:
    dis = (x[i] - center[0]).norm()
    if dis < smooth[None]:
      rho[None] += mass * sph_kernel2d(dis, smooth[None])


init()
integrate_density()

window = ti.ui.Window('kernel test', res=(800, 800), vsync=True)
canvas = window.get_canvas()

while window.running:

  if window.get_event(ti.ui.PRESS):
    if window.event.key == 'LMB':
      pos = window.get_cursor_pos()
      update_radius(pos[0], pos[1])
      integrate_density()

  gui = window.get_gui()
  with gui.sub_window('density', 0.0, 0.0, 0.3, 0.3):
    gui.text(str(rho[None]))

  canvas.set_background_color(color=(0.0, 0.0, 0.0))
  canvas.circles(center, color=(0.3, 0.3, 0.3), radius=smooth[None])
  canvas.circles(x, color=(0.7, 0.3, 0.2), radius=r)

  window.show()
