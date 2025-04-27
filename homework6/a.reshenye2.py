import numpy as np
from scipy import signal
from scipy.optimize import minimize
from itertools import permutations
import cv2
import os

def load_data(input_dir='best_pictures'):
    image_files = []
    target_files = {}
    
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.png'):
            img_path = os.path.join(input_dir, filename)
            txt_path = os.path.join(input_dir, filename.replace('.png', '.txt'))
            
            if os.path.exists(txt_path):
                image_files.append(img_path)
                target_files[img_path] = txt_path
    
    return image_files, target_files

def apply_filters(img, filters):
    result = img.copy().astype(np.float32)
    
    for f in filters:
        result = signal.convolve2d(result, f, mode='same', boundary='symm')
        result_min = result.min()
        result_max = result.max()
        if result_max - result_min > 1e-6:  
            result = (result - result_min) / (result_max - result_min) * 255
    return result

def calculate_mse(order, image_files, target_files, sample_size=None):
    total_mse = 0.0
    count = 0
    
    if sample_size is not None:
        image_files = image_files[:sample_size]
    
    for img_path in image_files:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        target = np.loadtxt(target_files[img_path]).astype(np.float32)
        
        result = apply_filters(img, order)
        mse = np.mean((result - target) ** 2)
        total_mse += mse
        count += 1
    
    return total_mse / count if count > 0 else float('inf')

def optimize_filter(filter1, filter2, image_files, target_files):
    initial_filter = np.array([-1, -2, -1, 0, 0, 0, 1, 2, 1], dtype=np.float32)
    
    def loss_function(filter_params):
        filter3 = filter_params.reshape(3, 3)
        return calculate_mse([filter1, filter2, filter3], image_files, target_files, sample_size=5)
    
    bounds = [(-2, 2)] * 9
    result = minimize(loss_function, initial_filter, method='L-BFGS-B', bounds=bounds)
    
    return result.x.reshape(3, 3)

def find_best_order(filter1, filter2, filter3, image_files, target_files):
    best_order = None
    best_mse = float('inf')
    
    for order in permutations([filter1, filter2, filter3]):
        current_mse = calculate_mse(order, image_files, target_files, sample_size=5)
        
        if current_mse < best_mse:
            best_mse = current_mse
            best_order = order
            print(f"Новый лучший порядок с MSE: {best_mse:.4f}")
    
    return best_order

def main():
    with open('best_pictures/algos.csv', 'r') as f:
        lines = f.readlines()
        filter1 = np.array([float(x) for x in lines[0].strip().split(',')]).reshape(3, 3)
        filter2 = np.array([float(x) for x in lines[1].strip().split(',')]).reshape(3, 3)
    
    image_files, target_files = load_data()
    filter3 = optimize_filter(filter1, filter2, image_files, target_files)
    print("Оптимизированный третий фильтр:")
    print(filter3)
    
    best_order = find_best_order(filter1, filter2, filter3, image_files, target_files)
    
    final_mse = calculate_mse(best_order, image_files, target_files)
    print(f"\nФинальное MSE: {final_mse:.4f}")
    
    with open('reconstructed_algos.csv', 'w') as f:
        for filt in best_order:
            f.write(','.join([f"{x:.6f}" for x in filt.flatten()]) + '\n')
    
    print("\nРезультат сохранен в reconstructed_algos.csv")

if __name__ == "__main__":
    main()