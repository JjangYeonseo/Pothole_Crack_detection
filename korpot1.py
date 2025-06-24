import os
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict

class DataPreprocessor:
    def __init__(self):
        # 클래스 매핑 정의
        self.class_mapping = {
            'ac': 0,  # 알리게이터 크랙
            'lc': 1,  # 길이 크랙 -> lctc로 통합
            'tc': 1,  # 횡단 크랙 -> lctc로 통합  
            'pc': 2,  # 패치
            'ph': 3   # 포트홀
        }
        
        self.class_names = ['ac', 'lctc', 'pc', 'ph']
        
    def load_json_label(self, json_path):
        """JSON 라벨 파일 로드"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def convert_to_yolo_format(self, json_data, img_width, img_height):
        """JSON 데이터를 YOLO 세그멘테이션 형식으로 변환"""
        yolo_lines = []
        
        if 'annotations' in json_data:
            annotations = json_data['annotations']
        elif 'shapes' in json_data:  # labelme 형식
            annotations = json_data['shapes']
        else:
            return yolo_lines
            
        for ann in annotations:
            # 클래스 라벨 확인
            if 'label' in ann:
                label = ann['label'].lower()
            elif 'category' in ann:
                label = ann['category'].lower()
            else:
                continue
                
            # 클래스 매핑
            if label not in self.class_mapping:
                continue
                
            class_id = self.class_mapping[label]
            
            # 폴리곤 좌표 추출
            if 'segmentation' in ann:
                points = ann['segmentation']
            elif 'points' in ann:
                points = ann['points']
            else:
                continue
                
            # 좌표 정규화
            normalized_points = []
            for point in points:
                if isinstance(point, list) and len(point) == 2:
                    x_norm = point[0] / img_width
                    y_norm = point[1] / img_height
                    normalized_points.extend([x_norm, y_norm])
                    
            if len(normalized_points) >= 6:  # 최소 3개 점 필요
                yolo_line = f"{class_id} " + " ".join(map(str, normalized_points))
                yolo_lines.append(yolo_line)
                
        return yolo_lines
    
    def process_original_data(self, img_dir, label_dir, output_img_dir, output_label_dir):
        """원본 데이터 처리"""
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)
        
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]
        
        for img_file in img_files:
            img_path = os.path.join(img_dir, img_file)
            label_file = img_file.replace('.jpg', '.json')
            label_path = os.path.join(label_dir, label_file)
            
            if not os.path.exists(label_path):
                continue
                
            # 이미지 로드
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img_height, img_width = img.shape[:2]
            
            # 라벨 로드 및 변환
            json_data = self.load_json_label(label_path)
            yolo_lines = self.convert_to_yolo_format(json_data, img_width, img_height)
            
            if yolo_lines:  # 유효한 라벨이 있는 경우만 저장
                # 이미지 복사
                shutil.copy2(img_path, os.path.join(output_img_dir, img_file))
                
                # YOLO 라벨 저장
                txt_file = img_file.replace('.jpg', '.txt')
                with open(os.path.join(output_label_dir, txt_file), 'w') as f:
                    f.write('\n'.join(yolo_lines))
                    
    def process_pothole_data(self, img_dir, label_dir, output_img_dir, output_label_dir):
        """포트홀 데이터 처리"""
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith('.png')]
        
        for img_file in img_files:
            img_path = os.path.join(img_dir, img_file)
            
            # 라벨 파일 찾기 (여러 하위 폴더에 있을 수 있음)
            label_file = img_file.replace('.png', '.json')
            label_path = None
            
            for root, dirs, files in os.walk(label_dir):
                if label_file in files:
                    label_path = os.path.join(root, label_file)
                    break
                    
            if not label_path or not os.path.exists(label_path):
                continue
                
            # 이미지 로드
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img_height, img_width = img.shape[:2]
            
            # 라벨 로드 및 변환
            json_data = self.load_json_label(label_path)
            yolo_lines = self.convert_to_yolo_format(json_data, img_width, img_height)
            
            # 포트홀 클래스만 필터링
            ph_lines = [line for line in yolo_lines if line.startswith('3')]
            
            if ph_lines:  # 포트홀이 있는 경우만 저장
                # 이미지를 jpg로 변환하여 저장
                jpg_file = img_file.replace('.png', '.jpg')
                cv2.imwrite(os.path.join(output_img_dir, jpg_file), img)
                
                # YOLO 라벨 저장
                txt_file = img_file.replace('.png', '.txt')
                with open(os.path.join(output_label_dir, txt_file), 'w') as f:
                    f.write('\n'.join(ph_lines))
    
    def analyze_dataset(self, label_dir):
        """데이터셋 분석"""
        class_counts = defaultdict(int)
        total_images = 0
        
        for txt_file in os.listdir(label_dir):
            if not txt_file.endswith('.txt'):
                continue
                
            total_images += 1
            txt_path = os.path.join(label_dir, txt_file)
            
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
                    
        print(f"총 이미지 수: {total_images}")
        print("클래스별 객체 수:")
        for class_id, count in class_counts.items():
            print(f"  {self.class_names[class_id]}: {count}")
            
        return class_counts

# 사용 예시
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    # 출력 디렉토리 설정
    base_output_dir = "C:/Users/dadab/Desktop/processed_dataset"
    
    # 학습 데이터 처리
    print("원본 학습 데이터 처리 중...")
    preprocessor.process_original_data(
        "C:/Users/dadab/Desktop/Final data and augmented datasets/train/images",
        "C:/Users/dadab/Desktop/Final data and augmented datasets/train/labels",
        f"{base_output_dir}/train/images",
        f"{base_output_dir}/train/labels"
    )
    
    # 검증 데이터 처리
    print("원본 검증 데이터 처리 중...")
    preprocessor.process_original_data(
        "C:/Users/dadab/Desktop/Final data and augmented datasets/val/images",
        "C:/Users/dadab/Desktop/Final data and augmented datasets/val/labels",
        f"{base_output_dir}/val/images",
        f"{base_output_dir}/val/labels"
    )
    
    # 포트홀 데이터 처리
    print("포트홀 데이터 처리 중...")
    preprocessor.process_pothole_data(
        "C:/Users/dadab/Desktop/183.이륜자동차 안전 위험 시설물 데이터/01.데이터/1.Training/원천데이터_230222_add/TS_Bounding Box_26.포트홀",
        "C:/Users/dadab/Desktop/183.이륜자동차 안전 위험 시설물 데이터/01.데이터/1.Training/라벨링테이터_230222_add",
        f"{base_output_dir}/train/images",
        f"{base_output_dir}/train/labels"
    )
    
    # 데이터셋 분석
    print("\n학습 데이터 분석:")
    preprocessor.analyze_dataset(f"{base_output_dir}/train/labels")
    print("\n검증 데이터 분석:")
    preprocessor.analyze_dataset(f"{base_output_dir}/val/labels")
