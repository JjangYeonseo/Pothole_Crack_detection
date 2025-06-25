import os
import json
import shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image
import numpy as np
from datetime import datetime

class DataPreprocessor:
    def __init__(self):
        self.label_alias = {
            'ac': 'ac',
            'lc': 'lctc', 'tc': 'lctc', 'lctc': 'lctc',
            'pc': 'pc',
            'ph': 'ph', 'pothole': 'ph', '포트홀': 'ph', '노면파손_포트홀': 'ph'
        }
        self.class_names = ['ac', 'lctc', 'pc', 'ph']
        self.class_id_map = {name: idx for idx, name in enumerate(self.class_names)}

    def load_json_label(self, json_path):
        encodings = ['utf-8', 'utf-8-sig', 'euc-kr', 'cp949']
        for enc in encodings:
            try:
                with open(json_path, 'r', encoding=enc) as f:
                    return json.load(f)
            except Exception:
                continue
        raise ValueError(f"JSON 로드 실패: {json_path}")

    def convert_to_yolo_format(self, json_data, img_width, img_height, is_pothole=False):
        yolo_lines = []
        annotations = json_data.get('annotations') or json_data.get('shapes') or json_data.get('annotation') or []

        for ann in annotations:
            raw_label = ann.get('label') or ann.get('category') or ''
            label = self.label_alias.get(raw_label.lower().strip())
            if not label:
                continue
            if is_pothole and label != 'ph':
                continue

            class_id = self.class_id_map[label]

            if 'points' in ann:  # polygon
                points = ann['points']
                norm_points = []
                for x, y in points:
                    norm_points.extend([x / img_width, y / img_height])
                if len(norm_points) >= 6:
                    yolo_lines.append(f"{class_id} " + " ".join(map(str, norm_points)))

            elif 'segmentation' in ann:
                points = ann['segmentation']
                norm_points = []
                for x, y in points:
                    norm_points.extend([x / img_width, y / img_height])
                if len(norm_points) >= 6:
                    yolo_lines.append(f"{class_id} " + " ".join(map(str, norm_points)))

            elif 'bbox' in ann:
                x, y, w, h = ann['bbox']
                # Convert bbox to polygon
                polygon = [
                    (x, y),
                    (x + w, y),
                    (x + w, y + h),
                    (x, y + h)
                ]
                norm_points = []
                for px, py in polygon:
                    norm_points.extend([px / img_width, py / img_height])
                yolo_lines.append(f"{class_id} " + " ".join(map(str, norm_points)))

        return yolo_lines

    def process_images(self, img_dir, label_dir, output_img_dir, output_label_dir, is_pothole=False, tag=''):
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)

        success, fail = 0, 0
        success_log, fail_log = [], []

        for label_file in os.listdir(label_dir):
            if not label_file.endswith('.json'):
                continue

            json_path = os.path.join(label_dir, label_file)
            base_name = os.path.splitext(label_file)[0]
            img_name_jpg = base_name + '.jpg'
            img_name_png = base_name + '.png'

            img_path = None
            for candidate in [img_name_jpg, img_name_png]:
                if os.path.exists(os.path.join(img_dir, candidate)):
                    img_path = os.path.join(img_dir, candidate)
                    break

            if not img_path:
                fail += 1
                fail_log.append(f"[{label_file}] 이미지 없음")
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                w, h = img.size
                json_data = self.load_json_label(json_path)
                yolo_lines = self.convert_to_yolo_format(json_data, w, h, is_pothole)
            except Exception as e:
                fail += 1
                fail_log.append(f"[{label_file}] 오류: {e}")
                continue

            # 무조건 저장 (객체 없어도)
            out_img_name = base_name + '.jpg'
            img.save(os.path.join(output_img_dir, out_img_name))

            out_label_path = os.path.join(output_label_dir, base_name + '.txt')
            with open(out_label_path, 'w') as f:
                if yolo_lines:
                    f.write('\n'.join(yolo_lines))
            success += 1
            success_log.append(f"[{label_file}] 성공")

        print(f"[{tag}] 처리 완료: {success}/{success + fail} 성공, {fail} 실패")
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(output_img_dir).parent
        with open(out_dir / f"fail_log_{tag}_{now}.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(fail_log))
        with open(out_dir / f"success_log_{tag}_{now}.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(success_log))

    def analyze_dataset(self, label_dir):
        counts = defaultdict(int)
        total = 0
        for fname in os.listdir(label_dir):
            if not fname.endswith('.txt'):
                continue
            total += 1
            with open(os.path.join(label_dir, fname), 'r') as f:
                for line in f:
                    if line.strip():
                        cid = int(line.split()[0])
                        counts[cid] += 1

        print(f"총 이미지 수: {total}")
        for cid, count in sorted(counts.items()):
            print(f"  {self.class_names[cid]}: {count}")
        return counts

# 실행
if __name__ == "__main__":
    dp = DataPreprocessor()

    base_out = "C:/Users/dadab/Desktop/processed_dataset"

    print("[1] 원본 학습 데이터 처리 중...")
    dp.process_images(
        img_dir="C:/Users/dadab/Desktop/Final data and augmented datasets/train/images",
        label_dir="C:/Users/dadab/Desktop/Final data and augmented datasets/train/labels",
        output_img_dir=f"{base_out}/train/images",
        output_label_dir=f"{base_out}/train/labels",
        is_pothole=False,
        tag="Train"
    )

    print("[2] 원본 검증 데이터 처리 중...")
    dp.process_images(
        img_dir="C:/Users/dadab/Desktop/Final data and augmented datasets/val/images",
        label_dir="C:/Users/dadab/Desktop/Final data and augmented datasets/val/labels",
        output_img_dir=f"{base_out}/val/images",
        output_label_dir=f"{base_out}/val/labels",
        is_pothole=False,
        tag="Val"
    )

    print("[3] 포트홀 추가 데이터 처리 중...")
    dp.process_images(
        img_dir="C:/Users/dadab/Desktop/183.이륜자동차 안전 위험 시설물 데이터/01.데이터/1.Training/원천데이터_230222_add/TS_Bounding Box_26.포트홀",
        label_dir="C:/Users/dadab/Desktop/183.이륜자동차 안전 위험 시설물 데이터/01.데이터/1.Training/라벨링테이터_230222_add/TL_Bounding Box_26.포트홀",
        output_img_dir=f"{base_out}/train/images",
        output_label_dir=f"{base_out}/train/labels",
        is_pothole=True,
        tag="Pothole"
    )

    print("\n[4] 학습 데이터셋 분석:")
    dp.analyze_dataset(f"{base_out}/train/labels")

    print("\n[5] 검증 데이터셋 분석:")
    dp.analyze_dataset(f"{base_out}/val/labels")
