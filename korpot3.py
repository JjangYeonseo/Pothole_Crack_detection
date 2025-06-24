import os
import yaml
from ultralytics import YOLO
import torch
import numpy as np
from pathlib import Path

class YOLOv8SegmentationTrainer:
    def __init__(self, data_dir, model_name="yolov8n-seg.pt"):
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.class_names = ['ac', 'lctc', 'pc', 'ph']
        
    def create_data_yaml(self, output_path="dataset.yaml"):
        """데이터셋 YAML 파일 생성"""
        data_config = {
            'path': str(self.data_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': 4,  # 클래스 수
            'names': self.class_names
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        print(f"데이터셋 YAML 파일 생성: {output_path}")
        return output_path
    
    def train_model(self, 
                   epochs=300,
                   imgsz=640,
                   batch_size=16,
                   lr0=0.01,
                   lrf=0.01,
                   momentum=0.937,
                   weight_decay=0.0005,
                   warmup_epochs=3.0,
                   warmup_momentum=0.8,
                   warmup_bias_lr=0.1,
                   box_loss_gain=7.5,
                   cls_loss_gain=0.5,
                   dfl_loss_gain=1.5,
                   device='0'):
        """YOLOv8 모델 학습"""
        
        # 데이터셋 YAML 생성
        data_yaml = self.create_data_yaml()
        
        # 모델 로드
        model = YOLO(self.model_name)
        
        # GPU 사용 가능 여부 확인
        if torch.cuda.is_available():
            print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
            device = 'cuda'
        else:
            print("GPU 사용 불가, CPU 사용")
            device = 'cpu'
        
        # 학습 하이퍼파라미터 설정 (85% 성능을 위한 최적화)
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'device': device,
            'workers': 8,
            'patience': 50,  # 조기 종료 방지
            'save': True,
            'save_period': 50,  # 50 에포크마다 저장
            'cache': True,  # 이미지 캐싱으로 속도 향상
            'amp': True,  # 혼합 정밀도 학습
            'fraction': 1.0,  # 전체 데이터 사용
            'profile': False,
            'freeze': None,  # 레이어 동결 없음
            'multi_scale': True,  # 다중 스케일 학습
            'overlap_mask': True,  # 마스크 오버랩 허용
            'mask_ratio': 4,  # 마스크 다운샘플링 비율
            'dropout': 0.0,  # 드롭아웃 비율
            'val': True,
            'plots': True,
            'verbose': True,
            
            # 옵티마이저 설정
            'optimizer': 'AdamW',
            'lr0': lr0,
            'lrf': lrf,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'warmup_epochs': warmup_epochs,
            'warmup_momentum': warmup_momentum,
            'warmup_bias_lr': warmup_bias_lr,
            
            # 손실 함수 가중치 (세그멘테이션 성능 향상)
            'box': box_loss_gain,
            'cls': cls_loss_gain,
            'dfl': dfl_loss_gain,
            
            # 증강 설정 (추가 증강 - 이미 증강된 데이터에 가벼운 추가 증강)
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,  # 회전은 이미 적용했으므로 0
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            
            # 기타 설정
            'cos_lr': True,  # 코사인 학습률 스케줄러
            'close_mosaic': 10,  # 마지막 10 에포크에서 모자이크 비활성화
        }
        
        # 학습 시작
        print("YOLOv8 세그멘테이션 학습 시작...")
        print(f"에포크: {epochs}, 이미지 크기: {imgsz}, 배치 크기: {batch_size}")
        
        results = model.train(**train_args)
        
        return model, results
    
    def validate_model(self, model_path=None):
        """모델 검증"""
        if model_path:
            model = YOLO(model_path)
        else:
            model = YOLO('runs/segment/train/weights/best.pt')
        
        # 검증 실행
        results = model.val(data=self.create_data_yaml(), 
                           imgsz=640, 
                           batch=16,
                           conf=0.25,
                           iou=0.6,
                           device='0' if torch.cuda.is_available() else 'cpu')
        
        # 클래스별 성능 출력
        print("\n=== 클래스별 성능 ===")
        for i, class_name in enumerate(self.class_names):
            if hasattr(results, 'box') and hasattr(results.box, 'map50'):
                map50 = results.box.map50[i] if i < len(results.box.map50) else 0
                print(f"{class_name}: mAP@0.5 = {map50:.3f}")
        
        return results
    
    def export_model(self, model_path=None, format='onnx'):
        """모델 내보내기"""
        if model_path:
            model = YOLO(model_path)
        else:
            model = YOLO('runs/segment/train/weights/best.pt')
        
        # 모델 내보내기
        model.export(format=format, imgsz=640, half=False, int8=False)
        print(f"모델이 {format} 형식으로 내보내졌습니다.")

# 고급 학습 설정 클래스
class AdvancedTrainingConfig:
    @staticmethod
    def get_high_performance_config():
        """85% 이상 성능을 위한 고급 설정"""
        return {
            'epochs': 400,
            'imgsz': 736,  # 더 큰 이미지 크기
            'batch_size': 12,  # GPU 메모리에 따라 조정
            'lr0': 0.001,  # 더 낮은 초기 학습률
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 5.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box_loss_gain': 7.5,
            'cls_loss_gain': 0.5,
            'dfl_loss_gain': 1.5,
        }
    
    @staticmethod
    def get_balanced_config():
        """균형 잡힌 설정"""
        return {
            'epochs': 300,
            'imgsz': 640,
            'batch_size': 16,
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box_loss_gain': 7.5,
            'cls_loss_gain': 0.5,
            'dfl_loss_gain': 1.5,
        }

# 앙상블 학습 클래스
class EnsembleTrainer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.models = []
    
    def train_multiple_models(self, num_models=3):
        """여러 모델을 다른 설정으로 학습"""
        configs = [
            {'lr0': 0.01, 'imgsz': 640, 'batch': 16},
            {'lr0': 0.005, 'imgsz': 736, 'batch': 12},
            {'lr0': 0.015, 'imgsz': 608, 'batch': 20},
        ]
        
        for i in range(num_models):
            print(f"\n=== 모델 {i+1} 학습 시작 ===")
            trainer = YOLOv8SegmentationTrainer(self.data_dir)
            config = configs[i % len(configs)]
            
            model, results = trainer.train_model(**config)
            self.models.append(f'runs/segment/train{i+1}/weights/best.pt')
    
    def ensemble_predict(self, image_path):
        """앙상블 예측"""
        predictions = []
        
        for model_path in self.models:
            model = YOLO(model_path)
            results = model(image_path)
            predictions.append(results)
        
        # 앙상블 로직 구현 (가중 평균 등)
        return predictions

# 사용 예시
if __name__ == "__main__":
    # 데이터 디렉토리 설정
    data_dir = "C:/Users/dadab/Desktop/augmented_dataset"
    
    # 트레이너 초기화
    trainer = YOLOv8SegmentationTrainer(data_dir)
    
    # 고성능 설정 사용
    config = AdvancedTrainingConfig.get_high_performance_config()
    
    # 학습 실행
    print("YOLOv8n-seg 모델 학습 시작...")
    model, results = trainer.train_model(**config)
    
    # 검증 실행
    print("\n모델 검증 중...")
    val_results = trainer.validate_model()
    
    # 모델 내보내기
    print("\n모델 내보내기...")
    trainer.export_model(format='onnx')
    
    print("\n학습 완료!")
    print("최적의 모델은 'runs/segment/train/weights/best.pt'에 저장되었습니다.")
    
    # 추가적인 성능 향상을 위한 앙상블 학습 (선택사항)
    # ensemble_trainer = EnsembleTrainer(data_dir)
    # ensemble_trainer.train_multiple_models(3)
