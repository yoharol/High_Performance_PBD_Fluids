import taichi as ti

ti.init(arch=ti.cuda)

n = 100
N = n * n
r = 0.2 / n

x = ti.Vector.field(2, dtype=ti.f32, shape=N)
x_cache = ti.Vector.field(2, dtype=ti.f32, shape=N)
v = ti.Vector.field(2, dtype=ti.f32, shape=N)
p = ti.field(dtype=ti.f32, shape=N)
rho = ti.field(dtype=ti.f32, shape=N)


@ti.kernel
def init():
  for k in range(N):
    i = int(k / n)
    j = int(k % n)
    px = 0.3 + 0.4 * float(i)
    py = 0.4 + 0.4 * float(j)
    x[k] = ti.Vector([px, py])
    if k < 100:
      print(i, j, px, py, x[k])


init()

window = ti.ui.Window('pbf fluids', res=(800, 800), vsync=True)
canvas = window.get_canvas()

while window.running:
  canvas.set_background_color(color=(0.0, 0.0, 0.0))
  canvas.circles(x, color=(0.7, 0.3, 0.2), radius=r * 10.0)
  window.show()
