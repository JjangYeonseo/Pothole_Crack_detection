import os
import yaml
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import cv2
import time
from datetime import datetime

class YOLOv8SegmentationTrainer:
    def __init__(self, data_dir, model_name="yolov8s-seg.pt"):
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.class_names = ['ac', 'lctc', 'pc', 'ph']
        
        # GPU 메모리 최적화 및 정보 출력
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🚀 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("⚠️ CPU 모드로 실행")

    def create_data_yaml(self, output_path="dataset.yaml"):
        data_config = {
            'path': str(self.data_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        with open(output_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        return output_path

    def get_optimal_settings(self):
        """RTX 3070 + 도로 결함 감지에 최적화된 설정 (속도 개선)"""
        return {
            # 기본 학습 설정
            'epochs': 200,          # 300 → 200 (조기 종료 고려)
            'imgsz': 640,
            'batch': 16,            # 12 → 16 (GPU 활용도 증가)
            'workers': 2,           # 0 → 2 (안전한 멀티프로세싱)
            'patience': 20,         # 25 → 20 (빠른 수렴)
            
            # 메모리 및 성능 최적화
            'cache': False,         # 디스크 공간 부족으로 캐시 비활성화
            'amp': True,
            'multi_scale': False,   # True → False (속도 향상)
            'overlap_mask': True,
            'mask_ratio': 4,
            'close_mosaic': 15,     # 10 → 15 (안정성)
            
            # 옵티마이저 (85% mAP 목표)
            'optimizer': 'AdamW',
            'lr0': 0.002,
            'lrf': 0.0001,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Loss 가중치 (세그멘테이션 최적화)
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            
            # 데이터 증강 (속도 최적화)
            'hsv_h': 0.005,         # 0.008 → 0.005 (연산 감소)
            'hsv_s': 0.2,           # 0.3 → 0.2
            'hsv_v': 0.1,           # 0.15 → 0.1
            'degrees': 2.0,         # 3.0 → 2.0
            'translate': 0.02,      # 0.03 → 0.02
            'scale': 0.15,          # 0.2 → 0.15
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 0.7,          # 0.9 → 0.7 (속도 향상)
            'mixup': 0.0,           # 0.05 → 0.0 (비활성화)
            'copy_paste': 0.0,
            
            # 스케줄러
            'cos_lr': True,
            
            # 저장 및 검증
            'save': True,
            'save_period': 20,
            'val': True,
            'plots': True,
            'verbose': True
        }

    def check_dataset_info(self):
        """데이터셋 정보 확인"""
        train_dir = self.data_dir / 'train' / 'images'
        val_dir = self.data_dir / 'val' / 'images'
        
        train_count = len(list(train_dir.glob("*.jpg"))) + len(list(train_dir.glob("*.png")))
        val_count = len(list(val_dir.glob("*.jpg"))) + len(list(val_dir.glob("*.png")))
        
        print(f"📊 데이터셋 정보:")
        print(f"   학습: {train_count}장")
        print(f"   검증: {val_count}장")
        print(f"   클래스: {self.class_names}")
        
        return train_count, val_count

    def train_model(self):
        print(f"🎯 모델: {self.model_name}")
        self.check_dataset_info()
        
        model = YOLO(self.model_name)
        data_yaml = self.create_data_yaml()
        args = self.get_optimal_settings()
        
        args.update({
            'data': data_yaml,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'project': 'runs/segment',
            'name': f'train_{self.data_dir.name}_{datetime.now().strftime("%m%d_%H%M")}'
        })
        
        print(f"\n🚀 학습 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   이미지 크기: {args['imgsz']}")
        print(f"   배치 크기: {args['batch']}")
        print(f"   학습률: {args['lr0']}")
        print(f"   워커 수: {args['workers']} (Windows 호환 모드)")
        
        start_time = time.time()
        results = model.train(**args)
        training_time = time.time() - start_time
        
        print(f"⏱️ 학습 완료 - 소요시간: {training_time/3600:.1f}시간")
        
        return model, results

    def validate_model(self, model_path=None):
        if model_path:
            model = YOLO(model_path)
        else:
            # 가장 최근 학습된 모델 찾기
            runs_dir = Path('runs/segment')
            if runs_dir.exists():
                latest_run = max([d for d in runs_dir.iterdir() if d.is_dir()], 
                               key=os.path.getctime)
                model = YOLO(latest_run / 'weights' / 'best.pt')
                print(f"🔍 모델 로드: {latest_run / 'weights' / 'best.pt'}")
            else:
                print("❌ 학습된 모델을 찾을 수 없습니다.")
                return None

        data_yaml = self.create_data_yaml()
        results = model.val(
            data=data_yaml,
            imgsz=640,
            batch=12,
            conf=0.25,
            iou=0.6,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        print("\n📊 세그멘테이션 성능 결과:")
        print("=" * 55)
        
        try:
            # 세그멘테이션 결과 우선 확인
            if hasattr(results, 'seg') and results.seg is not None:
                maps_50 = results.seg.map50
                maps_5095 = results.seg.map
                print("🎯 세그멘테이션 메트릭 사용")
            else:
                maps_50 = results.box.map50
                maps_5095 = results.box.map
                print("📦 바운딩박스 메트릭 사용")

            # 클래스별 성능 출력
            success_count = 0
            for i, name in enumerate(self.class_names):
                if isinstance(maps_50, (list, np.ndarray)) and len(maps_50) > i:
                    m50 = maps_50[i]
                    m5095 = maps_5095[i] if isinstance(maps_5095, (list, np.ndarray)) else maps_5095
                else:
                    m50 = maps_50 if not isinstance(maps_50, (list, np.ndarray)) else 0
                    m5095 = maps_5095 if not isinstance(maps_5095, (list, np.ndarray)) else 0
                
                status = "✅" if m50 >= 0.85 else "❌"
                if m50 >= 0.85:
                    success_count += 1
                    
                print(f"{name:>4} → mAP@0.5: {m50:.3f} | mAP@0.5:0.95: {m5095:.3f} {status}")

            # 전체 평균 및 목표 달성 여부
            avg_map50 = np.mean(maps_50) if isinstance(maps_50, (list, np.ndarray)) else maps_50
            avg_map5095 = np.mean(maps_5095) if isinstance(maps_5095, (list, np.ndarray)) else maps_5095
            
            print("-" * 55)
            print(f"📈 전체 평균 mAP@0.5    : {avg_map50:.3f}")
            print(f"📈 전체 평균 mAP@0.5:0.95: {avg_map5095:.3f}")
            
            if success_count == len(self.class_names):
                print(f"\n🎉 목표 달성! 모든 클래스가 85% 이상 달성 ({success_count}/{len(self.class_names)})")
            elif avg_map50 >= 0.85:
                print(f"\n🎯 평균 목표 달성! 평균 mAP@0.5: {avg_map50:.3f}")
                print(f"   개별 달성: {success_count}/{len(self.class_names)} 클래스")
            else:
                print(f"\n📊 현재 상태: 평균 mAP@0.5: {avg_map50:.3f} (목표: 0.85)")
                print(f"   개별 달성: {success_count}/{len(self.class_names)} 클래스")
                
        except Exception as e:
            print("⚠️ 평가 지표 추출 실패:", e)
            print("결과 객체 구조:", dir(results))

        return results

    def visualize_predictions(self, model_path=None, num_samples=10):
        if model_path:
            model = YOLO(model_path)
        else:
            runs_dir = Path('runs/segment')
            if runs_dir.exists():
                latest_run = max([d for d in runs_dir.iterdir() if d.is_dir()], 
                               key=os.path.getctime)
                model = YOLO(latest_run / 'weights' / 'best.pt')
            else:
                print("❌ 학습된 모델을 찾을 수 없습니다.")
                return

        image_dir = self.data_dir / 'val' / 'images'
        save_dir = Path("inference_samples")
        save_dir.mkdir(parents=True, exist_ok=True)

        images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        selected_images = images[:num_samples] if len(images) >= num_samples else images

        print(f"🖼️ {len(selected_images)}장 이미지 추론 중...")
        
        for i, img_path in enumerate(selected_images):
            results = model(img_path, save=False, stream=False, conf=0.25, iou=0.6)
            
            # 결과 이미지에 클래스 정보 추가
            result_img = results[0].plot()
            
            # 파일명에 원본 이름 포함
            original_name = img_path.stem
            save_path = save_dir / f"result_{i:02d}_{original_name}.jpg"
            cv2.imwrite(str(save_path), result_img)
            
            # 간단한 진행상황 표시
            if (i + 1) % 5 == 0 or i == len(selected_images) - 1:
                print(f"   진행: {i+1}/{len(selected_images)}")

        print(f"✅ 예측 결과 저장 완료: {save_dir}")

    def run_full_pipeline(self):
        """전체 파이프라인 실행"""
        print("🚀 YOLOv8 세그멘테이션 전체 파이프라인 시작")
        print("=" * 60)
        
        # 1. 학습
        print("\n1️⃣ 학습 단계")
        model, train_results = self.train_model()
        
        # 2. 검증
        print("\n2️⃣ 검증 단계")
        val_results = self.validate_model()
        
        # 3. 시각화
        print("\n3️⃣ 시각화 단계")
        self.visualize_predictions(num_samples=15)
        
        print("\n✅ 전체 파이프라인 완료!")
        return model, train_results, val_results

if __name__ == "__main__":
    # Windows에서 multiprocessing 오류 방지
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    data_dir = "C:/Users/dadab/Desktop/augmented_dataset"
    trainer = YOLOv8SegmentationTrainer(data_dir, model_name="yolov8s-seg.pt")
    
    # 전체 파이프라인 실행
    trainer.run_full_pipeline()
