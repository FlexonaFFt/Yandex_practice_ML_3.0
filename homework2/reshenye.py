import numpy as np
def compute_sobel_gradients_two_loops(image):
    # Get image dimensions
    height, width = image.shape

    # Initialize output gradients
    gradient_x = np.zeros_like(image, dtype=np.float64)
    gradient_y = np.zeros_like(image, dtype=np.float64)

    # Pad the image with zeros to handle borders
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant', constant_values=0)
# __________end of block__________

    # Define the Sobel kernels for X and Y gradients
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # Apply Sobel filter for X and Y gradients using convolution
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            window = padded_image[i-1:i+2, j-1:j+2]
            gradient_x[i-1, j-1] = np.sum(window * sobel_x)
            gradient_y[i-1, j-1] = np.sum(window * sobel_y)
    return gradient_x, gradient_y

def compute_gradient_magnitude(sobel_x, sobel_y):
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return magnitude


def compute_gradient_direction(sobel_x, sobel_y):
    direction = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
    return direction

cell_size = 7
def compute_hog(image, pixels_per_cell=(7, 7), bins=9):
    # 1. Convert to grayscale if needed
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)
    
    # 2. Compute gradients
    gx, gy = compute_sobel_gradients_two_loops(image)
    
    # 3. Compute magnitude and orientation (in degrees)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx) * 180 / np.pi  # range [-180, 180]
    
    # 4. Convert to unsigned gradients [0, 180]
    ang = (ang + 180) % 180
    
    # 5. Calculate histogram
    cell_h, cell_w = pixels_per_cell
    height, width = image.shape
    n_cells_y = height // cell_h
    n_cells_x = width // cell_w
    
    hist = np.zeros((n_cells_y, n_cells_x, bins))
    bin_size = 180 / bins
    
    for i in range(n_cells_y):
        for j in range(n_cells_x):
            # Extract cell
            cell_mag = mag[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            cell_ang = ang[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            
            # Compute histogram
            for y in range(cell_h):
                for x in range(cell_w):
                    angle = cell_ang[y, x]
                    magnitude = cell_mag[y, x]
                    
                    # Find appropriate bin
                    bin_idx = int(angle / bin_size)
                    if bin_idx == bins:  # Handle edge case
                        bin_idx = 0
                    hist[i, j, bin_idx] += magnitude
            
            # Normalize the histogram
            eps = 1e-5
            hist[i, j] /= np.sqrt(np.sum(hist[i, j]**2) + eps)
    
    return hist