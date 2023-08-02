from PIL import Image
import glob
import re

imgs = glob.glob("data/output/random/*.png")
imgs.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

# Create the frames
frames = []
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

# Save into a GIF file that loops forever
frames[0].save('map.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=1000, loop=0)