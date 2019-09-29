from numba import cuda
from sys import argv
import numpy as np
from pylab import imshow, show
from timeit import default_timer as timer
from matplotlib import animation
import matplotlib.pyplot as plt

zoom = complex(1, 0)
start = complex(0,0)


steps = 1000

def mandel(x, y, max_iters, zoom):
	
	c = complex(x,y)/zoom + complex(-0.743643887037158704752191506114774, 0.131825904205311970493132056385139)
	z = 0.0j
	for i in range(max_iters):
		z = z*z + c
		if (z.real*z.real + z.imag*z.imag) >= 4:
			return i
	return max_iters

mandel_gpu = cuda.jit(device=True)(mandel)


@cuda.jit
def mandel_kernel(min_x, max_x, min_y, max_y, image, iters, steps):
	height = image.shape[0]
	width = image.shape[1]
	
	pixel_size_x = (max_x - min_x) / width
	pixel_size_y = (max_y - min_y) / height
	
	startX , startY = cuda.grid(2)
	gridX = cuda.gridDim.x * cuda.blockDim.x
	gridY = cuda.gridDim.y * cuda.blockDim.y
	
	#start = complex(0,0)
	#zoom = 1
	for i in xrange(steps):
		for x in range(startX, width, gridX):
			real = min_x + x*pixel_size_x
			for y in range(startY, height, gridY):
				imag = min_y + y * pixel_size_y
				image[y, x, i] = iters - mandel_gpu(real, imag, iters, 1.2**i)
			
gimage = np.ones((1024, 1536, steps), dtype=np.uint8)
blockdim = (32, 8)
griddim = (32, 16)

start = timer()
d_image = cuda.to_device(gimage)
mandel_kernel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, d_image, 20000, steps)
d_image.to_host()
dt = timer() - start

print "Mandelbrot created on GPU in %f s" % dt

fig = plt.figure()
im = imshow(gimage[:,:,0])
def animate(i):
	im.set_array(gimage[:,:,i])
	return [im]
anim = animation.FuncAnimation(fig, animate, frames=steps, interval=100)
plt.show()

if len(argv) == 2:
	np.save("Datos_mandelbrot/" + argv[1], gimage)

