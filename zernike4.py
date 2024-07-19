import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import scipy.special as sp
from scipy.fft import fft2, fftshift
from scipy import ndimage

# USE TO FIND N AND M VALUES
# https://www.gatinel.com/wp-content/uploads/2015/09/radial-order-and-azimutal-frequency.png
# USE TO FIND OUTPUT OF POLY ON PSF
# https://image.slideserve.com/140346/double-index-zernike-polynomial-psfs-l.jpg

class ZernikePolynomial:
	def __init__(self, n, m):
		"""
		Initialize a Zernike polynomial of order n and frequency m.

		Args:
			n (int): The order of the Zernike polynomial (non-negative integer).
			m (int): The frequency of the Zernike polynomial (integer between -n and n).

		Raises:
			ValueError: If n or m are invalid.
		"""
		if n < 0 or m < -n or m > n:
			raise ValueError("Invalid values for n and m.")
		self.n = n
		self.m = m

	def evaluate(self, x, y):
		"""
		Evaluate the Zernike polynomial at given Cartesian coordinates (x, y).

		Args:
			x (float): x-coordinate.
			y (float): y-coordinate.

		Returns:
			float: The value of the Zernike polynomial at (x, y).
		"""
		rho = np.sqrt(x**2 + y**2)
		phi = np.arctan2(y, x)
		R_nm = self.radial_polynomial(rho)
		return R_nm * np.cos(self.m * phi)

	def radial_polynomial(self, rho):
		"""
		Calculate the radial polynomial component of the Zernike polynomial.

		Args:
			rho (float): Radial distance from the origin.

		Returns:
			float: The value of the radial polynomial at rho.
		"""
		result = 0.0
		for s in range((self.n - self.m) // 2 + 1):
			c = (-1)**s * sp.comb(self.n - s, s)
			result += c * rho**(self.n - 2 * s)
		return result

	def generate_phasemap(self, grid_size=256):
		"""
		Plot the Zernike polynomial in Cartesian coordinates.

		Args:
			grid_size (int): Number of points to use for the grid.
		"""
		x = np.linspace(-1, 1, grid_size)
		y = np.linspace(-1, 1, grid_size)
		X, Y = np.meshgrid(x, y)
		Z = np.zeros_like(X)

		for i in range(grid_size):
			for j in range(grid_size):
				Z[i, j] = self.evaluate(X[i, j], Y[i, j])

		return Z

	def generate_psf(self, phasemap, wavelength, aperture_radius, psf_size, pixel_size):
		# Create a grid of coordinates
		size = phasemap.shape[0]
		x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
		rho = np.sqrt(x**2 + y**2)

		# Compute the pupil function based on the aperture and phase map
		pupil_function = (rho <= aperture_radius) * np.exp(1j * 2 * np.pi * phasemap / wavelength)

		# Perform a 2D FFT to obtain the PSF
		psf = np.abs(fftshift(fft2(pupil_function)))**2

		# Scale the PSF by the pixel size
		psf /= (pixel_size ** 2)

		# Resize the PSF to the desired size for better visualization
		psf = np.pad(psf, ((psf_size-size)//2, (psf_size-size)//2), mode='constant')

		return psf

	def plot_phasemap(self, phasemap):
		plt.figure()
		plt.imshow(phasemap, extent=(-1, 1, -1, 1), origin='lower', cmap='RdGy')
		plt.colorbar()
		plt.title(f'Zernike Polynomial n={self.n} m={self.m}')
		plt.xlabel('x')
		plt.ylabel('y')
		plt.show()

	def plot_psf(self, psf):
		plt.figure()
		plt.imshow(psf, cmap='viridis')
		plt.title('Point Spread Function (PSF)')
		plt.colorbar(label='Intensity')
		plt.xlabel('Pixel')
		plt.ylabel('Pixel')
		plt.show()


	def apply_psf_to_image(self, image, psf):
		# Separate the image into color channels
		r_channel, g_channel, b_channel = image[:,:,0], image[:,:,1], image[:,:,2]

		# Convolve each color channel with the PSF
		r_convolved = convolve2d(r_channel, psf, mode='same', boundary='wrap')
		g_convolved = convolve2d(g_channel, psf, mode='same', boundary='wrap')
		b_convolved = convolve2d(b_channel, psf, mode='same', boundary='wrap')

		# Stack the convolved color channels back into an RGB image
		convolved_image = np.stack((r_convolved, g_convolved, b_convolved), axis=-1)

		# Normalize the values to the range [0, 255]
		convolved_image = (convolved_image - np.min(convolved_image)) / (np.max(convolved_image) - np.min(convolved_image)) * 255

		# Ensure the data type is uint8 (for displaying as an image)
		convolved_image = convolved_image.astype(np.uint8)

		return convolved_image

	def rotate_psf(self, psf, theta):

		rotated_psf = ndimage.rotate(psf, theta)
		return rotated_psf
