import numpy as np
import cv2
from itertools import permutations
import os 

def load_data():
    filters_known = np.loadtxt("best_pictures/algos.csv", delimiter=",").reshape(2, 3, 3)
    F1, F2 = filters_known[0], filters_known[1]
    input_images = []
    output_matrices = []
    
    png_files = [f for f in os.listdir("best_pictures") if f.endswith('.png')]
    txt_files = [f.replace('.png', '.txt') for f in png_files]
    
    for png_file, txt_file in zip(png_files, txt_files):
        img = cv2.imread(os.path.join("best_pictures", png_file), cv2.IMREAD_GRAYSCALE)
        matrix = np.loadtxt(os.path.join("best_pictures", txt_file))
        input_images.append(img)
        output_matrices.append(matrix)

    return F1, F2, input_images, output_matrices

def apply_filter(image, filter):
    return cv2.filter2D(image, -1, filter, borderType=cv2.BORDER_CONSTANT)

def compute_mse(pred, target):
    return np.mean((pred - target) ** 2)

def find_best_F3_and_order(F1, F2, input_images, output_matrices):
    best_mse = float('inf')
    best_order = None
    best_F3 = None

    for order in permutations([0, 1, 2]):
        A = []
        b = []
        
        for img, out in zip(input_images, output_matrices):
            temp = img.copy()
            for f_idx in order[:2]:
                if f_idx == 0:
                    temp = apply_filter(temp, F1)
                elif f_idx == 1:
                    temp = apply_filter(temp, F2)
            
            for i in range(1, temp.shape[0] - 1):
                for j in range(1, temp.shape[1] - 1):
                    window = temp[i-1:i+2, j-1:j+2]
                    target = out[i, j]
                    A.append(window.flatten())
                    b.append(target)
        
        A = np.array(A).reshape(-1, 9)
        b = np.array(b)
        
        if len(A) == 0:
            continue
        
        try:
            F3_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            F3_candidate = F3_flat.reshape(3, 3)
        except np.linalg.LinAlgError:
            continue

        total_mse = 0
        for img, out in zip(input_images, output_matrices):
            temp = img.copy()
            for f_idx in order:
                if f_idx == 0:
                    temp = apply_filter(temp, F1)
                elif f_idx == 1:
                    temp = apply_filter(temp, F2)
                else:
                    temp = apply_filter(temp, F3_candidate)
            mse = compute_mse(temp, out)
            total_mse += mse

        if total_mse < best_mse:
            best_mse = total_mse
            best_order = order
            best_F3 = F3_candidate

    return best_order, best_F3

def save_result(order, F1, F2, F3):
    filters = [F1, F2, F3]
    ordered_filters = [filters[i] for i in order]
    # Преобразуем список 2D массивов в один 2D массив (3x9)
    result = np.vstack([f.reshape(1, 9) for f in ordered_filters])
    np.savetxt("reconstructed_algos.csv", result, delimiter=",", fmt="%.6f")

if __name__ == "__main__":
    F1, F2, input_images, output_matrices = load_data()
    best_order, best_F3 = find_best_F3_and_order(F1, F2, input_images, output_matrices)
    save_result(best_order, F1, F2, best_F3)
    print("Готово! Результат сохранён в reconstructed_algos.csv")