import sys
from PIL import Image

filelist = ['20.png', '40.png', '60.png', '80.png', '100.png', '120.png']

images = [Image.open(x) for x in filelist]
widths, heights = zip(*(i.size for i in images))

total_width = int(sum(widths) / 2)
total_height = int(max(heights) * 2)

new_im = Image.new('RGB', (total_width, total_height))

x_offset = 0
y_offset = 0
for i in range(len(images)):
  im = images[i]
  new_im.paste(im, (x_offset, y_offset))
  x_offset += im.size[0]
  if i == len(images) // 2 - 1:
    y_offset += im.size[1]
    x_offset = 0

new_im.save('combined.jpg')