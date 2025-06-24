import cv2
import numpy as np
import os
import random
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SegmentationAugmenter:
    def __init__(self, target_counts=None):
        """
        target_counts: 각 클래스별 목표 개수 딕셔너리
        예: {0: 2000, 1: 3000, 2: 1500, 3: 2000}
        """
        self.target_counts = target_counts or {0: 2000, 1: 3000, 2: 1500, 3: 2000}
        self.class_names = ['ac', 'lctc', 'pc', 'ph']
        
        # LCTC(클래스 1)에 집중적인 증강을 위한 강력한 변환
        self.heavy_transform = A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=45, p=0.7),
            ], p=0.8),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            ], p=0.7),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.4),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1], per_channel=True, p=0.3),
            ], p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=0.3),
                A.MedianBlur(blur_limit=5, p=0.3),
                A.GaussianBlur(blur_limit=5, p=0.3),
            ], p=0.4),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.4),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                A.Perspective(scale=(0.05, 0.1), p=0.3),
            ], p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.2),
        ])
        
        # 일반적인 증강 (다른 클래스용)
        self.normal_transform = A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=30, p=0.6),
            ], p=0.7),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
            ], p=0.6),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
                A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=0.3),
            ], p=0.4),
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.2),
            ], p=0.3),
            A.RandomGamma(gamma_limit=(90, 110), p=0.2),
        ])
        
        # 가벼운 증강 (포트홀용 - 이미 충분한 데이터가 있으므로)
        self.light_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
            A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
            A.Rotate(limit=15, p=0.3),
        ])
    
    def parse_yolo_label(self, label_path):
        """YOLO 라벨 파싱"""
        annotations = []
        if not os.path.exists(label_path):
            return annotations
            
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 7:  # class_id + at least 3 points (6 coordinates)
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                
                # 좌표를 (x, y) 쌍으로 변환
                points = []
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        points.append([coords[i], coords[i + 1]])
                        
                annotations.append({
                    'class_id': class_id,
                    'points': points
                })
                
        return annotations
    
    def save_yolo_label(self, annotations, label_path, img_width, img_height):
        """YOLO 라벨 저장"""
        with open(label_path, 'w') as f:
            for ann in annotations:
                class_id = ann['class_id']
                points = ann['points']
                
                # 좌표 정규화 및 변환
                normalized_coords = []
                for point in points:
                    x_norm = max(0, min(1, point[0]))
                    y_norm = max(0, min(1, point[1]))
                    normalized_coords.extend([x_norm, y_norm])
                
                if len(normalized_coords) >= 6:  # 최소 3개 점
                    coords_str = ' '.join(map(str, normalized_coords))
                    f.write(f"{class_id} {coords_str}\n")
    
    def apply_augmentation(self, image, annotations, transform):
        """증강 적용"""
        img_height, img_width = image.shape[:2]
        
        # 폴리곤을 keypoints로 변환
        keypoints = []
        keypoint_classes = []
        
        for ann in annotations:
            for point in ann['points']:
                x = point[0] * img_width
                y = point[1] * img_height
                keypoints.append([x, y])
                keypoint_classes.append(ann['class_id'])
        
        if not keypoints:
            # 키포인트가 없으면 이미지만 변환
            transformed = transform(image=image)
            return transformed['image'], annotations
        
        # Albumentations 변환 적용
        try:
            transformed = transform(
                image=image,
                keypoints=keypoints,
                keypoint_classes=keypoint_classes
            )
            
            aug_image = transformed['image']
            aug_keypoints = transformed.get('keypoints', [])
            aug_classes = transformed.get('keypoint_classes', [])
            
            # 변환된 키포인트를 다시 annotation 형태로 변환
            aug_annotations = []
            current_ann = {'class_id': None, 'points': []}
            
            for i, (keypoint, class_id) in enumerate(zip(aug_keypoints, aug_classes)):
                x_norm = keypoint[0] / aug_image.shape[1]
                y_norm = keypoint[1] / aug_image.shape[0]
                
                if current_ann['class_id'] is None:
                    current_ann['class_id'] = class_id
                
                if class_id == current_ann['class_id']:
                    current_ann['points'].append([x_norm, y_norm])
                else:
                    if current_ann['points']:
                        aug_annotations.append(current_ann)
                    current_ann = {'class_id': class_id, 'points': [[x_norm, y_norm]]}
            
            if current_ann['points']:
                aug_annotations.append(current_ann)
                
            return aug_image, aug_annotations
            
        except Exception as e:
            print(f"증강 적용 중 오류 발생: {e}")
            return image, annotations
    
    def augment_dataset(self, input_img_dir, input_label_dir, output_img_dir, output_label_dir):
        """데이터셋 증강"""
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)
        
        # 현재 클래스별 개수 계산
        class_counts = defaultdict(int)
        image_by_class = defaultdict(list)
        
        # 기존 이미지들을 복사하고 클래스별로 분류
        for img_file in os.listdir(input_img_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(input_img_dir, img_file)
            label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
            label_path = os.path.join(input_label_dir, label_file)
            
            # 기존 파일 복사
            shutil.copy2(img_path, os.path.join(output_img_dir, img_file))
            if os.path.exists(label_path):
                shutil.copy2(label_path, os.path.join(output_label_dir, label_file))
                
                # 클래스 개수 계산
                annotations = self.parse_yolo_label(label_path)
                for ann in annotations:
                    class_id = ann['class_id']
                    class_counts[class_id] += 1
                    image_by_class[class_id].append(img_file)
        
        print("현재 클래스별 개수:")
        for class_id, count in class_counts.items():
            print(f"  {self.class_names[class_id]}: {count}")
        
        # 각 클래스별로 증강
        for class_id, target_count in self.target_counts.items():
            current_count = class_counts[class_id]
            need_count = max(0, target_count - current_count)
            
            if need_count == 0:
                continue
                
            print(f"\n{self.class_names[class_id]} 클래스 증강: {need_count}개 필요")
            
            # 해당 클래스가 포함된 이미지들
            class_images = image_by_class[class_id]
            if not class_images:
                continue
            
            # 증강 변환 선택
            if class_id == 1:  # LCTC - 강력한 증강
                transform = self.heavy_transform
                aug_per_image = max(1, need_count // len(class_images))
            elif class_id == 3:  # 포트홀 - 가벼운 증강
                transform = self.light_transform
                aug_per_image = max(1, need_count // len(class_images))
            else:  # AC, PC - 일반 증강
                transform = self.normal_transform
                aug_per_image = max(1, need_count // len(class_images))
            
            generated_count = 0
            for img_file in class_images:
                if generated_count >= need_count:
                    break
                    
                img_path = os.path.join(input_img_dir, img_file)
                label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
                label_path = os.path.join(input_label_dir, label_file)
                
                # 이미지 로드
                image = cv2.imread(img_path)
                if image is None:
                    continue
                    
                # 라벨 로드
                annotations = self.parse_yolo_label(label_path)
                
                # 해당 클래스가 포함된 annotation만 필터링
                class_annotations = [ann for ann in annotations if ann['class_id'] == class_id]
                if not class_annotations:
                    continue
                
                # 여러 번 증강
                for aug_idx in range(aug_per_image):
                    if generated_count >= need_count:
                        break
                        
                    # 증강 적용
                    aug_image, aug_annotations = self.apply_augmentation(
                        image, annotations, transform
                    )
                    
                    # 파일명 생성
                    base_name = os.path.splitext(img_file)[0]
                    aug_img_file = f"{base_name}_aug_{class_id}_{aug_idx}.jpg"
                    aug_label_file = f"{base_name}_aug_{class_id}_{aug_idx}.txt"
                    
                    # 증강된 이미지 저장
                    cv2.imwrite(
                        os.path.join(output_img_dir, aug_img_file),
                        aug_image
                    )
                    
                    # 증강된 라벨 저장
                    self.save_yolo_label(
                        aug_annotations,
                        os.path.join(output_label_dir, aug_label_file),
                        aug_image.shape[1],
                        aug_image.shape[0]
                    )
                    
                    generated_count += 1
                    
                    if generated_count % 100 == 0:
                        print(f"  {generated_count}/{need_count} 생성 완료")
            
            print(f"  {self.class_names[class_id]} 증강 완료: {generated_count}개 생성")

# 사용 예시
if __name__ == "__main__":
    # 목표 개수 설정 (85% 성능을 위해 충분한 데이터 확보)
    target_counts = {
        0: 2500,  # AC
        1: 4000,  # LCTC (가장 많이 증강)
        2: 2000,  # PC
        3: 2500   # PH
    }
    
    augmenter = SegmentationAugmenter(target_counts)
    
    base_dir = "C:/Users/dadab/Desktop/processed_dataset"
    augmented_dir = "C:/Users/dadab/Desktop/augmented_dataset"
    
    print("학습 데이터 증강 시작...")
    augmenter.augment_dataset(
        f"{base_dir}/train/images",
        f"{base_dir}/train/labels",
        f"{augmented_dir}/train/images",
        f"{augmented_dir}/train/labels"
    )
    
    print("\n검증 데이터 복사...")
    # 검증 데이터는 증강하지 않고 그대로 복사
    os.makedirs(f"{augmented_dir}/val/images", exist_ok=True)
    os.makedirs(f"{augmented_dir}/val/labels", exist_ok=True)
    
    for img_file in os.listdir(f"{base_dir}/val/images"):
        shutil.copy2(
            os.path.join(f"{base_dir}/val/images", img_file),
            os.path.join(f"{augmented_dir}/val/images", img_file)
        )
    
    for label_file in os.listdir(f"{base_dir}/val/labels"):
        shutil.copy2(
            os.path.join(f"{base_dir}/val/labels", label_file),
            os.path.join(f"{augmented_dir}/val/labels", label_file)
        )
    
    print("데이터 증강 완료!")
