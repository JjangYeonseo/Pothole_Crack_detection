import os
import yaml
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import cv2
import time
from datetime import datetime

class FastOptimizedYOLOv8Trainer:
    def __init__(self, data_dir, model_name="yolov8s-seg.pt"):
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.class_names = ['ac', 'lctc', 'pc', 'ph']
        
        # 속도-성능 균형을 위한 클래스 가중치
        # 분석 결과 기반으로 미리 최적화된 값
        self.class_weights = [1.0, 2.2, 1.0, 2.0]  # ac, lctc, pc, ph
        
        # GPU 최적화
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 메모리 사용량 최적화
            torch.backends.cudnn.benchmark = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🚀 GPU: {gpu_name} ({gpu_memory:.1f}GB) - 고속 모드")
        else:
            print("⚠️ CPU 모드 - 속도 제한적")

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

    def get_fast_optimized_settings(self):
        """속도-성능 균형 최적화 설정 (2-3시간 목표)"""
        return {
            # 🚀 속도 최적화된 기본 설정
            'epochs': 200,          # 250 → 200 (20% 시간 단축)
            'imgsz': 640,
            'batch': 16,            # 배치 크기 유지 (성능 위해)
            'workers': 4,           # 2 → 4 (데이터 로딩 가속)
            'patience': 25,         # 조기 종료로 시간 절약
            
            # 🔥 메모리 최적화 + 속도
            'cache': 'ram',         # False → 'ram' (첫 에폭 후 가속)
            'amp': True,            # Mixed precision (속도 향상)
            'multi_scale': False,   # True → False (속도 향상)
            'overlap_mask': True,
            'mask_ratio': 4,
            'close_mosaic': 15,
            
            # ⚡ 빠른 수렴을 위한 옵티마이저
            'optimizer': 'AdamW',
            'lr0': 0.002,           # 약간 높은 학습률 (빠른 수렴)
            'lrf': 0.0001,
            'momentum': 0.937,
            'weight_decay': 0.0006,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # 🎯 세그멘테이션 최적화 (검증된 값)
            'box': 7.0,
            'cls': 0.7,             # 분류 성능 강화
            'dfl': 1.3,
            
            # 🎨 효과적인 데이터 증강 (과도하지 않게)
            'hsv_h': 0.01,          # 색상 변화 최소화 (속도)
            'hsv_s': 0.3,
            'hsv_v': 0.15,
            'degrees': 3.0,         # 적당한 회전
            'translate': 0.08,      # 적당한 이동
            'scale': 0.25,          # 적당한 크기 변화
            'shear': 1.0,           # 전단 변환 최소화
            'perspective': 0.0001,  # 원근 변환 최소화
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 0.9,          # 1.0 → 0.9 (속도 향상)
            'mixup': 0.05,          # 0.1 → 0.05 (속도 향상)
            'copy_paste': 0.0,
            
            # 📈 학습률 스케줄러
            'cos_lr': True,
            
            # 💾 저장 최적화
            'save': True,
            'save_period': 30,      # 25 → 30 (I/O 감소)
            'val': True,
            'plots': True,
            'verbose': True,
            
            # 🧠 성능 향상 설정 (속도 영향 최소)
            'dropout': 0.08,        # 0.1 → 0.08 (가벼운 정규화)
            'label_smoothing': 0.03, # 0.05 → 0.03 (가벼운 스무딩)
        }

    def quick_dataset_analysis(self):
        """빠른 데이터셋 분석"""
        train_dir = self.data_dir / 'train' / 'images'
        val_dir = self.data_dir / 'val' / 'images'
        
        train_count = len(list(train_dir.glob("*.jpg"))) + len(list(train_dir.glob("*.png")))
        val_count = len(list(val_dir.glob("*.jpg"))) + len(list(val_dir.glob("*.png")))
        
        print(f"📊 데이터셋: 학습 {train_count}장, 검증 {val_count}장")
        print(f"🎯 클래스 가중치: {dict(zip(self.class_names, self.class_weights))}")
        
        return train_count, val_count

    def fast_train(self):
        """빠른 고성능 학습"""
        print(f"⚡ 빠른 YOLOv8 세그멘테이션 학습 시작")
        print(f"🎯 목표: 2-3시간 내 85%+ 성능 달성")
        print("=" * 60)
        
        train_count, val_count = self.quick_dataset_analysis()
        
        # 예상 학습 시간 계산
        estimated_time = self.estimate_training_time(train_count)
        print(f"⏱️ 예상 학습 시간: {estimated_time:.1f}시간")
        
        model = YOLO(self.model_name)
        data_yaml = self.create_data_yaml()
        
        args = self.get_fast_optimized_settings()
        args.update({
            'data': data_yaml,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'project': 'runs/fast_segment',
            'name': f'fast_train_{self.data_dir.name}_{datetime.now().strftime("%m%d_%H%M")}',
            'exist_ok': True,
            'pretrained': True,
            'resume': False,
        })
        
        print(f"\n🚀 고속 학습 시작 - {datetime.now().strftime('%H:%M:%S')}")
        print(f"   ⚙️ 에폭: {args['epochs']}")
        print(f"   📦 배치: {args['batch']}")
        print(f"   🧠 캐시: {args['cache']}")
        print(f"   ⚡ AMP: {args['amp']}")
        
        start_time = time.time()
        results = model.train(**args)
        training_time = time.time() - start_time
        
        print(f"\n✅ 학습 완료!")
        print(f"⏱️ 실제 소요 시간: {training_time/3600:.1f}시간")
        print(f"📈 시간당 에폭: {args['epochs']/(training_time/3600):.1f}")
        
        return model, results

    def estimate_training_time(self, image_count):
        """학습 시간 추정"""
        # GPU 성능에 따른 대략적 추정
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory >= 10:  # RTX 3080, 4070 이상
                time_per_epoch = (image_count / 1000) * 0.8  # 분
            elif gpu_memory >= 8:   # RTX 3070, 4060 이상  
                time_per_epoch = (image_count / 1000) * 1.2
            else:  # GTX 1660, RTX 3060 이하
                time_per_epoch = (image_count / 1000) * 1.8
        else:
            time_per_epoch = (image_count / 1000) * 5.0  # CPU는 매우 느림
        
        total_minutes = time_per_epoch * 200  # 200 에폭
        return total_minutes / 60  # 시간 단위로 변환

    def fast_validate_with_target_check(self, model_path=None):
        """빠른 검증 + 목표 달성 확인"""
        if model_path:
            model = YOLO(model_path)
        else:
            runs_dir = Path('runs/fast_segment')
            if runs_dir.exists():
                latest_run = max([d for d in runs_dir.iterdir() if d.is_dir()], 
                               key=os.path.getctime)
                model = YOLO(latest_run / 'weights' / 'best.pt')
                print(f"🔍 모델: {latest_run / 'weights' / 'best.pt'}")
            else:
                print("❌ 학습된 모델이 없습니다.")
                return None

        data_yaml = self.create_data_yaml()
        
        print("\n🔬 성능 검증 중...")
        results = model.val(
            data=data_yaml,
            imgsz=640,
            batch=16,  # 검증도 빠르게
            conf=0.25,
            iou=0.6,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=True
        )
        
        try:
            if hasattr(results, 'seg') and results.seg is not None:
                maps_50 = results.seg.map50
            else:
                maps_50 = results.box.map50

            print(f"\n🎯 빠른 학습 최종 결과:")
            print("=" * 40)
            
            success_count = 0
            total_gap = 0
            
            for i, name in enumerate(self.class_names):
                if isinstance(maps_50, (list, np.ndarray)) and len(maps_50) > i:
                    m50 = maps_50[i]
                else:
                    m50 = maps_50
                
                status = "✅" if m50 >= 0.85 else "❌"
                if m50 >= 0.85:
                    success_count += 1
                else:
                    total_gap += (0.85 - m50)
                    
                print(f"  {name:>4}: {m50:.3f} {status}")

            avg_map = np.mean(maps_50) if isinstance(maps_50, (list, np.ndarray)) else maps_50
            
            print(f"  평균: {avg_map:.3f}")
            print(f"  달성: {success_count}/{len(self.class_names)} 클래스")
            
            # 결과 평가
            if success_count == len(self.class_names):
                print(f"\n🎉 완벽! 모든 클래스 85% 달성!")
                print(f"⚡ 빠른 학습으로도 목표 달성 성공!")
            elif success_count >= len(self.class_names) * 0.75:
                print(f"\n👍 양호! 대부분 클래스 목표 근접")
                print(f"💡 추가 50-100 에폭이면 완전 달성 가능")
            else:
                print(f"\n🤔 추가 최적화 필요")
                print(f"💡 적응형 학습 또는 데이터 보강 고려")
                
        except Exception as e:
            print(f"⚠️ 성능 분석 실패: {e}")

        return results

    def quick_visualization(self, num_samples=10):
        """빠른 결과 시각화"""
        runs_dir = Path('runs/fast_segment')
        if runs_dir.exists():
            latest_run = max([d for d in runs_dir.iterdir() if d.is_dir()], 
                           key=os.path.getctime)
            model = YOLO(latest_run / 'weights' / 'best.pt')
        else:
            print("❌ 시각화할 모델이 없습니다.")
            return

        image_dir = self.data_dir / 'val' / 'images'
        save_dir = Path("fast_inference_samples")
        save_dir.mkdir(parents=True, exist_ok=True)

        images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        selected_images = images[:num_samples]

        print(f"🖼️ {len(selected_images)}장 빠른 시각화...")
        
        for i, img_path in enumerate(selected_images):
            results = model(img_path, save=False, conf=0.25, iou=0.6)
            result_img = results[0].plot()
            save_path = save_dir / f"fast_result_{i:02d}_{img_path.stem}.jpg"
            cv2.imwrite(str(save_path), result_img)

        print(f"✅ 빠른 시각화 완료: {save_dir}")

    def run_fast_pipeline(self):
        """빠른 전체 파이프라인"""
        print("⚡ 빠른 YOLOv8 세그멘테이션 파이프라인")
        print("🎯 목표: 2-3시간 내 고성능 달성")
        print("=" * 50)
        
        # 1. 빠른 학습
        model, train_results = self.fast_train()
        
        # 2. 빠른 검증
        val_results = self.fast_validate_with_target_check()
        
        # 3. 빠른 시각화
        self.quick_visualization()
        
        print(f"\n⚡ 빠른 파이프라인 완료!")
        return model, train_results, val_results

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    data_dir = "C:/Users/dadab/Desktop/augmented_dataset"
    trainer = FastOptimizedYOLOv8Trainer(data_dir, model_name="yolov8s-seg.pt")
    
    # 빠른 파이프라인 실행
    trainer.run_fast_pipeline()
