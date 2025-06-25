import cv2
import numpy as np
import os
import random
import shutil
from collections import defaultdict
import albumentations as A

class SegmentationAugmenter:
    def __init__(self, target_counts=None):
        self.target_counts = target_counts or {0: 2000, 1: 3000, 2: 1500, 3: 2000}
        self.class_names = ['ac', 'lctc', 'pc', 'ph']

        # 고품질 증강 전략 구성
        self.heavy_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=45, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.MotionBlur(blur_limit=3, p=0.3),
            ], p=0.4),
            A.Perspective(scale=(0.05, 0.1), p=0.3)
        ], keypoint_params=A.KeypointParams(format='xy'))

        self.normal_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        ], keypoint_params=A.KeypointParams(format='xy'))

        self.light_transform = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        ], keypoint_params=A.KeypointParams(format='xy'))

    def parse_yolo_label(self, label_path):
        annotations = []
        if not os.path.exists(label_path):
            return annotations
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:
                    class_id = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    points = [[coords[i], coords[i + 1]] for i in range(0, len(coords) - 1, 2)]
                    annotations.append({'class_id': class_id, 'points': points})
        return annotations

    def save_yolo_label(self, annotations, label_path):
        with open(label_path, 'w') as f:
            for ann in annotations:
                class_id = ann['class_id']
                coords = [str(round(c, 6)) for point in ann['points'] for c in point]
                if len(coords) >= 6:
                    f.write(f"{class_id} {' '.join(coords)}\n")

    def apply_augmentation(self, image, annotations, transform):
        h, w = image.shape[:2]
        keypoints = []
        keypoint_classes = []

        for ann in annotations:
            for point in ann['points']:
                if len(point) != 2:
                    continue
                x = np.clip(point[0] * w, 0, w - 1e-3)
                y = np.clip(point[1] * h, 0, h - 1e-3)
                keypoints.append([x, y])
                keypoint_classes.append(ann['class_id'])

        if not keypoints:
            transformed = transform(image=image)
            return transformed['image'], annotations

        try:
            transformed = transform(image=image, keypoints=keypoints)
            aug_image = transformed['image']
            aug_keypoints = transformed['keypoints']

            ann_map = defaultdict(list)
            for (x, y), cls in zip(aug_keypoints, keypoint_classes):
                x_norm = x / w
                y_norm = y / h
                ann_map[cls].append([x_norm, y_norm])

            new_annotations = []
            for cls, points in ann_map.items():
                if len(points) >= 3:
                    new_annotations.append({'class_id': cls, 'points': points})

            return aug_image, new_annotations
        except Exception as e:
            print(f"증강 적용 중 오류 발생: {e}")
            return image, annotations

    def augment_dataset(self, input_img_dir, input_label_dir, output_img_dir, output_label_dir):
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)

        class_counts = defaultdict(int)
        image_by_class = defaultdict(list)

        for img_file in os.listdir(input_img_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            base = os.path.splitext(img_file)[0]
            img_path = os.path.join(input_img_dir, img_file)
            label_path = os.path.join(input_label_dir, f"{base}.txt")
            shutil.copy2(img_path, os.path.join(output_img_dir, img_file))
            if os.path.exists(label_path):
                shutil.copy2(label_path, os.path.join(output_label_dir, f"{base}.txt"))
                annotations = self.parse_yolo_label(label_path)
                for ann in annotations:
                    class_counts[ann['class_id']] += 1
                    image_by_class[ann['class_id']].append(img_file)

        print("현재 클래스별 개수:")
        for cid, cnt in class_counts.items():
            print(f"  {self.class_names[cid]}: {cnt}")

        for class_id, target_count in self.target_counts.items():
            current = class_counts[class_id]
            needed = target_count - current
            if needed <= 0:
                print(f"{self.class_names[class_id]} 증강 불필요 (보유 {current})")
                continue
            print(f"{self.class_names[class_id]} 증강: {needed}개 필요")

            selected_images = image_by_class[class_id]
            if not selected_images:
                print("  ⚠️ 해당 클래스 이미지 없음")
                continue

            transform = self.heavy_transform if class_id == 1 else self.light_transform if class_id == 3 else self.normal_transform
            per_image = max(1, needed // len(selected_images))
            count = 0

            for img_file in selected_images:
                if count >= needed:
                    break
                base = os.path.splitext(img_file)[0]
                img_path = os.path.join(input_img_dir, img_file)
                label_path = os.path.join(input_label_dir, f"{base}.txt")
                image = cv2.imread(img_path)
                if image is None:
                    continue
                annotations = self.parse_yolo_label(label_path)
                for i in range(per_image):
                    if count >= needed:
                        break
                    aug_img, aug_anns = self.apply_augmentation(image, annotations, transform)
                    if not aug_anns:
                        continue
                    out_name = f"{base}_aug_{class_id}_{i}"
                    cv2.imwrite(os.path.join(output_img_dir, f"{out_name}.jpg"), aug_img)
                    self.save_yolo_label(aug_anns, os.path.join(output_label_dir, f"{out_name}.txt"))
                    count += 1
            print(f"  생성 완료: {count}개")

if __name__ == "__main__":
    augmenter = SegmentationAugmenter({0: 2500, 1: 4000, 2: 2000, 3: 2500})
    base = "C:/Users/dadab/Desktop/processed_dataset"
    out = "C:/Users/dadab/Desktop/augmented_dataset"

    print("📈 학습 데이터 증강 시작...")
    augmenter.augment_dataset(f"{base}/train/images", f"{base}/train/labels", f"{out}/train/images", f"{out}/train/labels")

    print("\n📦 검증 데이터 복사...")
    os.makedirs(f"{out}/val/images", exist_ok=True)
    os.makedirs(f"{out}/val/labels", exist_ok=True)
    for f in os.listdir(f"{base}/val/images"):
        shutil.copy2(os.path.join(f"{base}/val/images", f), os.path.join(f"{out}/val/images", f))
    for f in os.listdir(f"{base}/val/labels"):
        shutil.copy2(os.path.join(f"{base}/val/labels", f), os.path.join(f"{out}/val/labels", f))
    print("✅ 완료!")
