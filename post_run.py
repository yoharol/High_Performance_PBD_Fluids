import matplotlib.pyplot as plt
from glob import glob
import sys
import os

assert len(sys.argv)==2
filename = sys.argv[1]

for foldername in [f"img/{filename}"]:
  if not os.path.exists(foldername):
    os.makedirs(foldername)


if filename == "cuda":
  with open("data/rho_log.txt", 'r') as f:
    lines = f.readlines()
    rho = []
    sum_rho = 0
    for line in lines:
      rho.append(float(line))
      sum_rho += float(line)
    avg_rho = sum_rho / len(rho)
    plt.plot(rho)
    plt.text(0.0, rho[0], f'rho0: {rho[0]:.2f}')
    plt.text(0.0, avg_rho, f'avgrho: {avg_rho:.2f}')
    plt.savefig(f"img/{filename}/{filename}_rho.png")
  plt.clf()
  with open("data/timestep_log.txt", 'r') as f:
    lines = f.readlines()
    timesteps = []
    sum_time = 0.0
    for line in lines:
      timesteps.append(float(line))
      sum_time += float(line)
    avg_timestep = sum_time / len(timesteps)
    plt.plot(timesteps)
    plt.text(0.0, avg_timestep, f'avg time: {avg_timestep:.4f}', color='r')
    plt.savefig(f"img/{filename}/{filename}_timesteps.png")
  quit()
      

with open("data/rho_log.txt", 'r') as f:
  lines = f.readlines()
  rho = []
  sum_rho = 0
  for line in lines:
    rho.append(float(line))
    sum_rho += float(line)
  avg_rho = (sum_rho-rho[0]) / (len(rho)-1)
  plt.plot(rho)
  plt.text(0.0, rho[0], f'rho0: {rho[0]:.2f}')
  plt.text(0.0, avg_rho, f'avgrho: {avg_rho:.2f}')
  plt.savefig(f"img/{filename}/{filename}_rho.png")

plt.clf()

with open("data/timestep_log.txt", 'r') as f:
  lines = f.readlines()
  timesteps = []
  sum_time = 0.0
  for line in lines:
    timesteps.append(float(line))
    sum_time += float(line)
  avg_timestep = (sum_time - timesteps[0]) / (len(timesteps)-1)
  plt.plot(timesteps)
  plt.text(0.0, timesteps[0], f'init: {timesteps[0]:.2f}', color='r')
  plt.text(0.0, avg_timestep, f'avg time: {avg_timestep:.2f}', color='r')
  plt.savefig(f"img/{filename}/{filename}_timesteps.png")

plt.clf()


with open("data/max_part_in_cell_log.txt", 'r') as f:
  lines = f.readlines()
  max_pic = []
  for line in lines:
    max_pic.append(int(line))
  plt.plot(max_pic)

  plt.savefig(f"img/{filename}/{filename}_maxpic.png")

plt.clf()

for logname in ['prediction', 'collision', 'neighbor', 'rho_integral','lambda_compute', 'pos_update', 'vel_update', 'viscosity']:
  with open(f"data/{logname}_log.txt", 'r') as f:
    lines = f.readlines()
    data_log = []
    data_sum = 0.0
    for line in lines:
      data_log.append(float(line))
      data_sum += float(line)
    plt.plot(data_log, label=logname)
plt.legend()
plt.savefig(f"img/{filename}/{filename}_datalog.png")

log_file_list = glob('data/*_log.txt')
for file in log_file_list:
    os.remove(file)