import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from skimage import color, feature
import cv2
import warnings
from tqdm import tqdm
import csv

class Config:
    image_size = 512
    crop_size = 448
    batch_size = 32
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_neighbors = 6

class ArtDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, self.image_files[idx]

class ArtFeatureExtractor:
    def __init__(self):
        self.config = Config()
        self.model = self._init_model()
        self.transform = self._init_transform()
        
    def _init_model(self):
        weights = EfficientNet_B3_Weights.DEFAULT
        model = efficientnet_b3(weights=weights)
        model.classifier = nn.Identity()
        model.eval()
        model.to(self.config.device)
        return model
    
    def _init_transform(self):
        return transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.CenterCrop(self.config.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
        ])
    
    def extract_deep_features(self, dataloader):
        features = []
        filenames = []
        
        with torch.no_grad():
            for batch, names in tqdm(dataloader, desc="Extracting deep features"):
                batch = batch.to(self.config.device)
                feats = self.model(batch).cpu().numpy()
                features.append(feats)
                filenames.extend(names)
                
        return np.vstack(features), filenames
    
    def extract_handcrafted_features(self, image_paths):
        features = []
        
        for path in tqdm(image_paths, desc="Extracting handcrafted features"):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            hsv = color.rgb2hsv(img)
            hist_h = np.histogram(hsv[:,:,0], bins=20, range=(0,1))[0]
            hist_s = np.histogram(hsv[:,:,1], bins=20, range=(0,1))[0]
            
            gray = color.rgb2gray(img)
            lbp = feature.local_binary_pattern(gray, P=16, R=2, method='uniform')
            hist_lbp = np.histogram(lbp, bins=20, range=(0,20))[0]
            
            edges = cv2.Canny(img, 100, 200)
            edge_density = np.mean(edges)
            
            feat = np.concatenate([
                hist_h.astype(np.float32),
                hist_s.astype(np.float32),
                hist_lbp.astype(np.float32),
                [edge_density]
            ])
            features.append(feat)
            
        return np.vstack(features)

class ArtSimilarityPipeline:
    def __init__(self, image_dir):
        self.config = Config()
        self.image_dir = image_dir
        self.extractor = ArtFeatureExtractor()
        self.dataset = ArtDataset(image_dir, self.extractor.transform)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False
        )
        
    def _get_image_paths(self):
        return [os.path.join(self.image_dir, f) for f in self.dataset.image_files]
    
    def _combine_features(self, deep_feats, handcrafted_feats):
        deep_feats = deep_feats / (np.linalg.norm(deep_feats, axis=1, keepdims=True) + 1e-10)
        handcrafted_feats = handcrafted_feats / (np.linalg.norm(handcrafted_feats, axis=1, keepdims=True) + 1e-10)
        
        return np.concatenate([
            deep_feats * 0.8,  
            handcrafted_feats * 0.2
        ], axis=1)
    
    def _find_similar_images(self, features):
        nbrs = NearestNeighbors(
            n_neighbors=self.config.n_neighbors + 1,
            algorithm='auto',
            metric='cosine'
        ).fit(features)
        
        distances, indices = nbrs.kneighbors(features)
        return indices[:, 1:]  
    
    def _save_results(self, indices, filenames, output_file="submission.csv"):
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename', 'ranking'])
            
            for i, neighbors in enumerate(indices):
                query_name = filenames[i]
                neighbor_names = ' '.join([filenames[j] for j in neighbors])
                writer.writerow([query_name, neighbor_names])
    
    def run(self):
        deep_features, filenames = self.extractor.extract_deep_features(self.dataloader)
        image_paths = self._get_image_paths()
        handcrafted_features = self.extractor.extract_handcrafted_features(image_paths)
        combined_features = self._combine_features(deep_features, handcrafted_features)
        indices = self._find_similar_images(combined_features)
        self._save_results(indices, filenames)
        print("Processing completed. Results saved to submission.csv")

if __name__ == "__main__":
    image_dir = "dataset"  
    pipeline = ArtSimilarityPipeline(image_dir)
    pipeline.run()