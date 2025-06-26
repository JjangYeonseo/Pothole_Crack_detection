import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import os
from pathlib import Path
from collections import defaultdict
import torch
import matplotlib.font_manager as fm 

# 폰트 경로 유효성 검사 및 설정
if os.path.exists(font_path):
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
    print(f"Matplotlib 한글 폰트 설정 완료: {font_name}")
else:
    print("Warning: 한글 폰트 경로를 찾을 수 없습니다. 그래프에 한글이 깨질 수 있습니다. 'DejaVu Sans' 사용.")
    plt.rcParams['font.family'] = 'DejaVu Sans' # 기본 영문 폰트
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지 (이전 줄과 중복 가능성 있지만 안전을 위해 유지)

class ModelEvaluator:
    def __init__(self, model_path, data_dir):
        # YOLO 모델 로드
        self.model = YOLO(model_path)
        self.data_dir = Path(data_dir)
        # 클래스 이름 정의
        self.class_names = ['ac', 'lctc', 'pc', 'ph']
        # 시각화를 위한 클래스별 색상 정의
        self.class_colors = {
            0: (255, 0, 0),     # AC - 빨강
            1: (0, 255, 0),     # LCTC - 초록
            2: (0, 0, 255),     # PC - 파랑
            3: (255, 255, 0)    # PH - 노랑
        }
    
    def visualize_predictions(self, num_samples=10, output_dir="evaluation_results"):
        """
        모델의 예측 결과를 시각화하여 이미지 파일로 저장합니다.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        val_img_dir = self.data_dir / 'val' / 'images'
        img_files = sorted([f for f in os.listdir(val_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) 
        
        if len(img_files) == 0:
            print(f"Warning: {val_img_dir}에서 이미지를 찾을 수 없습니다. 시각화를 건너킵니다.")
            return
            
        sample_files = np.random.choice(img_files, min(num_samples, len(img_files)), replace=False)
        
        print(f"\n예측 결과 이미지 {len(sample_files)}개 시각화 중...")
        for i, img_file in enumerate(sample_files):
            img_path = val_img_dir / img_file
            
            # 모델 예측 실행
            # conf와 iou 임계값을 설정하여 예측 결과를 필터링할 수 있습니다.
            # 너무 높은 임계값은 예측 결과가 거의 없을 수 있습니다.
            # 모델이 아무것도 탐지하지 못했다면 낮은 conf로 시도해보세요.
            results_pred = self.model(str(img_path), conf=0.25, iou=0.5, verbose=False) 
            
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Error: 이미지를 읽을 수 없습니다: {img_path}. 시각화를 건너뜁니다.")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 예측 결과 (마스크)를 이미지에 그리기
            if results_pred[0].masks is not None and len(results_pred[0].masks.data) > 0:
                for j, mask_tensor in enumerate(results_pred[0].masks.data):
                    class_id = int(results_pred[0].boxes.cls[j])
                    confidence = float(results_pred[0].boxes.conf[j])
                    
                    mask_np = mask_tensor.cpu().numpy()
                    
                    if mask_np.shape[0] != image.shape[0] or mask_np.shape[1] != image.shape[1]:
                         mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) 
                    else:
                        mask_resized = mask_np
                    
                    color = self.class_colors.get(class_id, (255, 255, 255)) 
                    colored_mask = np.zeros_like(image, dtype=np.uint8)
                    colored_mask[mask_resized > 0.5] = color 
                    
                    image = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
                    
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
                    label = f"{class_name}: {confidence:.2f}"
                    
                    y_coords, x_coords = np.where(mask_resized > 0.5)
                    if len(y_coords) > 0 and len(x_coords) > 0:
                        text_y, text_x = int(np.mean(y_coords)), int(np.mean(x_coords))
                        text_x = max(0, text_x - len(label) * 5) 
                        text_y = max(20, text_y) 
                        
                        cv2.putText(image, label, (text_x, text_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(image)
            plt.title(f"Prediction Result: {img_file}") # 그래프 제목은 영어로 유지
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"prediction_{i+1}.png"), dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"예측 결과 이미지가 '{output_dir}'에 저장되었습니다.")

if __name__ == "__main__":
    # 모델 경로와 데이터 디렉토리 설정 (실제 경로로 변경 필요)
    model_path = r"C:\Users\dadab\Desktop\runs\segment\train_augmented_dataset\weights\best.pt"
    data_dir = "C:/Users/dadab/Desktop/augmented_dataset"
    
    # ModelEvaluator 초기화
    evaluator = ModelEvaluator(model_path, data_dir)
    
    # 예측 결과 시각화만 실행
    # num_samples=10으로 설정하여 10장의 이미지를 저장합니다.
    evaluator.visualize_predictions(num_samples=10) 
    
    print("\n작업 완료!")
