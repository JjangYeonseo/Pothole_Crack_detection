import os
import yaml
from ultralytics import YOLO
import torch
from pathlib import Path

class YOLOv8SegmentationTrainer:
    def __init__(self, data_dir, model_name="yolov8n-seg.pt"):
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.class_names = ['ac', 'lctc', 'pc', 'ph']

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
        print(f"데이터셋 YAML 파일 생성: {output_path}")
        return output_path

    def train_model(self, 
                    epochs=300,
                    imgsz=640,
                    batch_size=8,
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

        data_yaml = self.create_data_yaml()
        model = YOLO(self.model_name)

        if torch.cuda.is_available():
            print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
            device = 'cuda'
        else:
            print("GPU 사용 불가, CPU 사용")
            device = 'cpu'

        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'device': device,
            'workers': 4,                # 메모리 절약
            'patience': 50,
            'save': True,
            'save_period': 50,
            'cache': 'disk',            # RAM 대신 디스크 사용
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': True,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True,
            'verbose': True,
            'optimizer': 'AdamW',
            'lr0': lr0,
            'lrf': lrf,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'warmup_epochs': warmup_epochs,
            'warmup_momentum': warmup_momentum,
            'warmup_bias_lr': warmup_bias_lr,
            'box': box_loss_gain,
            'cls': cls_loss_gain,
            'dfl': dfl_loss_gain,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'cos_lr': True,
            'close_mosaic': 10,
            'project': 'runs/segment',
            'name': f'train_{self.data_dir.name}'
        }

        print("YOLOv8 세그멘테이션 학습 시작...")
        print(f"에포크: {epochs}, 이미지 크기: {imgsz}, 배치 크기: {batch_size}")
        results = model.train(**train_args)
        return model, results

    def validate_model(self, model_path=None):
        if model_path:
            model = YOLO(model_path)
        else:
            model = YOLO(f'runs/segment/train_{self.data_dir.name}/weights/best.pt')

        results = model.val(data=self.create_data_yaml(), 
                            imgsz=640, 
                            batch=8,
                            conf=0.25,
                            iou=0.6,
                            device='0' if torch.cuda.is_available() else 'cpu')

        print("\n=== 클래스별 성능 ===")
        for i, class_name in enumerate(self.class_names):
            if hasattr(results, 'segment') and hasattr(results.segment, 'map50'):
                map50 = results.segment.map50[i] if i < len(results.segment.map50) else 0
                print(f"{class_name}: mAP@0.5 = {map50:.3f}")

        return results

    def export_model(self, model_path=None, format='onnx'):
        if model_path:
            model = YOLO(model_path)
        else:
            model = YOLO(f'runs/segment/train_{self.data_dir.name}/weights/best.pt')

        model.export(format=format, imgsz=640, half=False, int8=False)
        print(f"모델이 {format} 형식으로 내보내졌습니다.")

class AdvancedTrainingConfig:
    @staticmethod
    def get_high_performance_config():
        return {
            'epochs': 400,
            'imgsz': 736,
            'batch_size': 8,  # 메모리 사용 최적화
            'lr0': 0.001,
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

if __name__ == "__main__":
    data_dir = "C:/Users/dadab/Desktop/augmented_dataset"
    trainer = YOLOv8SegmentationTrainer(data_dir)
    config = AdvancedTrainingConfig.get_high_performance_config()

    print("YOLOv8n-seg 모델 학습 시작...")
    model, results = trainer.train_model(**config)

    print("\n모델 검증 중...")
    val_results = trainer.validate_model()

    print("\n모델 내보내기...")
    trainer.export_model(format='onnx')

    print("\n학습 완료!")
