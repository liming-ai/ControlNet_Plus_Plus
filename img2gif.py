import os
import imageio

task = 'seg'

# Define the filenames of your images
root_dir = f'images/{task}'
filenames = os.listdir(root_dir)
filenames = [os.path.join(root_dir, filename) for filename in filenames]

# Read each image file and append it to a list
images = []
for filename in filenames:
    images.append(imageio.imread(filename))

# Write the images to a gif file
imageio.mimsave(f'{task}.gif', images, fps=0.3)