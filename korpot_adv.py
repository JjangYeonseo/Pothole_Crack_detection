import os
import json
from pathlib import Path
from collections import defaultdict
from PIL import Image
from datetime import datetime
import shutil

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
        
        # 포트홀 키워드들 (인코딩 문제 대응)
        self.pothole_keywords = ['포트홀', 'pothole', 'ph', '��ƮȦ', '포트홀', '26']

    def load_json_label(self, json_path):
        """JSON 라벨 파일 로드 (다양한 인코딩 시도)"""
        for enc in ['utf-8', 'utf-8-sig', 'euc-kr', 'cp949', 'latin-1']:
            try:
                with open(json_path, 'r', encoding=enc) as f:
                    data = json.load(f)
                    # 성공적으로 로드되면 인코딩 정보도 함께 반환
                    return data, enc
            except:
                continue
        raise ValueError(f"JSON 로드 실패: {json_path}")

    def detect_pothole_label(self, raw_label):
        """포트홀 라벨 감지 (인코딩 문제 대응)"""
        if not raw_label:
            return False
        
        raw_label_lower = raw_label.lower().strip()
        
        # 기존 매핑 체크
        if raw_label_lower in self.label_alias:
            return self.label_alias[raw_label_lower] == 'ph'
        
        # 포트홀 키워드 포함 여부 체크
        for keyword in self.pothole_keywords:
            if keyword in raw_label or keyword in raw_label_lower:
                return True
        
        # 숫자로 시작하는 라벨 중 26번 체크 (포트홀)
        if raw_label_lower.startswith('26'):
            return True
            
        return False

    def get_label_class(self, raw_label):
        """라벨에서 클래스 추출"""
        if not raw_label:
            return None
        
        raw_label_lower = raw_label.lower().strip()
        
        # 기존 매핑 체크
        if raw_label_lower in self.label_alias:
            return self.label_alias[raw_label_lower]
        
        # 포트홀 감지
        if self.detect_pothole_label(raw_label):
            return 'ph'
        
        return None

    def convert_to_yolo_format(self, json_data, img_width, img_height, allowed_classes=None, encoding='utf-8'):
        yolo_lines = []
        annotations = json_data.get('annotations') or json_data.get('shapes') or json_data.get('annotation') or []

        for ann in annotations:
            raw_label = ann.get('label') or ann.get('category') or ''
            label = self.get_label_class(raw_label)
            
            if not label or (allowed_classes and label not in allowed_classes):
                # 디버깅용 로그
                if raw_label and encoding != 'utf-8':
                    print(f"[라벨 감지 실패] '{raw_label}' (인코딩: {encoding})")
                continue
                
            class_id = self.class_id_map[label]

            if 'points' in ann:
                points = ann['points']
                flat_points = [p for point in points for p in point]
                norm_points = [p / img_width if i % 2 == 0 else p / img_height for i, p in enumerate(flat_points)]
                if len(norm_points) >= 6:
                    yolo_lines.append(f"{class_id} " + " ".join(map(str, norm_points)))

            elif 'segmentation' in ann:
                points = ann['segmentation']
                flat_points = [p for point in points for p in point]
                norm_points = [p / img_width if i % 2 == 0 else p / img_height for i, p in enumerate(flat_points)]
                if len(norm_points) >= 6:
                    yolo_lines.append(f"{class_id} " + " ".join(map(str, norm_points)))

            elif 'bbox' in ann:
                x, y, w, h = ann['bbox']
                polygon = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                norm_points = []
                for px, py in polygon:
                    norm_points.extend([px / img_width, py / img_height])
                yolo_lines.append(f"{class_id} " + " ".join(map(str, norm_points)))

        return yolo_lines

    def merge_labels_with_pothole_replacement(self, orig_img_dir, orig_label_dir, pothole_img_dir, pothole_label_dir, output_img_dir, output_label_dir):
        """기존 라벨과 포트홀 라벨을 병합하되, 포트홀은 새로운 것으로 대체"""
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)
        
        processed_files = set()
        
        # 1. 포트홀 데이터가 있는 파일들을 우선 처리 (기존 라벨과 병합)
        pothole_files = [f for f in os.listdir(pothole_label_dir) if f.endswith('.json')]
        print(f"포트홀 라벨 파일 개수: {len(pothole_files)}")
        
        for json_file in pothole_files:
            base_name = os.path.splitext(json_file)[0]
            processed_files.add(base_name)

            # 이미지 경로 확인 (기존 이미지 우선)
            img_path = None
            for ext in ['.jpg', '.png', '.jpeg']:
                orig_img_path = os.path.join(orig_img_dir, base_name + ext)
                pothole_img_path = os.path.join(pothole_img_dir, base_name + ext)
                
                if os.path.exists(orig_img_path):
                    img_path = orig_img_path
                    break
                elif os.path.exists(pothole_img_path):
                    img_path = pothole_img_path
                    break

            if not img_path:
                print(f"[이미지 없음] {base_name}")
                continue

            # 이미지 복사
            img = Image.open(img_path).convert("RGB")
            output_img_path = os.path.join(output_img_dir, base_name + ".jpg")
            img.save(output_img_path)
            w, h = img.size

            merged_lines = []

            # 기존 라벨에서 포트홀이 아닌 객체들만 가져오기
            orig_json_path = os.path.join(orig_label_dir, base_name + ".json")
            if os.path.exists(orig_json_path):
                try:
                    orig_json, orig_enc = self.load_json_label(orig_json_path)
                    # ac, lctc, pc만 허용 (포트홀 제외)
                    orig_lines = self.convert_to_yolo_format(orig_json, w, h, allowed_classes={'ac', 'lctc', 'pc'}, encoding=orig_enc)
                    merged_lines.extend(orig_lines)
                    print(f"[기존 라벨 병합] {base_name}: {len(orig_lines)}개 객체 (인코딩: {orig_enc})")
                except Exception as e:
                    print(f"[기존 라벨 로드 실패] {base_name}: {e}")

            # 새로운 포트홀 라벨 추가
            try:
                pothole_json, pothole_enc = self.load_json_label(os.path.join(pothole_label_dir, json_file))
                pothole_lines = self.convert_to_yolo_format(pothole_json, w, h, allowed_classes={'ph'}, encoding=pothole_enc)
                merged_lines.extend(pothole_lines)
                print(f"[포트홀 라벨 추가] {base_name}: {len(pothole_lines)}개 포트홀 (인코딩: {pothole_enc})")
            except Exception as e:
                print(f"[포트홀 라벨 로드 실패] {base_name}: {e}")

            # 최종 라벨 파일 저장
            output_label_path = os.path.join(output_label_dir, base_name + ".txt")
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(merged_lines))
            print(f"[완료] {base_name}: 총 {len(merged_lines)}개 객체")

        # 2. 포트홀 데이터에 없는 기존 데이터들 처리
        if os.path.exists(orig_label_dir):
            orig_files = [f for f in os.listdir(orig_label_dir) if f.endswith('.json')]
            for json_file in orig_files:
                base_name = os.path.splitext(json_file)[0]
                
                if base_name in processed_files:
                    continue  # 이미 처리됨
                
                # 기존 이미지 복사
                img_path = None
                for ext in ['.jpg', '.png', '.jpeg']:
                    test_path = os.path.join(orig_img_dir, base_name + ext)
                    if os.path.exists(test_path):
                        img_path = test_path
                        break
                
                if not img_path:
                    continue
                
                img = Image.open(img_path).convert("RGB")
                output_img_path = os.path.join(output_img_dir, base_name + ".jpg")
                img.save(output_img_path)
                w, h = img.size
                
                # 기존 라벨 그대로 사용
                try:
                    orig_json, orig_enc = self.load_json_label(os.path.join(orig_label_dir, json_file))
                    yolo_lines = self.convert_to_yolo_format(orig_json, w, h, encoding=orig_enc)
                    
                    output_label_path = os.path.join(output_label_dir, base_name + ".txt")
                    with open(output_label_path, 'w') as f:
                        f.write('\n'.join(yolo_lines))
                    print(f"[기존 데이터] {base_name}: {len(yolo_lines)}개 객체 (인코딩: {orig_enc})")
                except Exception as e:
                    print(f"[기존 데이터 처리 실패] {base_name}: {e}")

    def process_pothole_only(self, img_dir, label_dir, output_img_dir, output_label_dir):
        """포트홀만 있는 데이터 처리"""
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)

        json_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
        print(f"포트홀 전용 데이터 파일 개수: {len(json_files)}")

        for json_file in json_files:
            base_name = os.path.splitext(json_file)[0]

            img_path = None
            for ext in ['.jpg', '.png', '.jpeg']:
                test_path = os.path.join(img_dir, base_name + ext)
                if os.path.exists(test_path):
                    img_path = test_path
                    break
            
            if not img_path:
                print(f"[이미지 없음] {base_name}")
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                w, h = img.size
                json_data, encoding = self.load_json_label(os.path.join(label_dir, json_file))
                yolo_lines = self.convert_to_yolo_format(json_data, w, h, allowed_classes={'ph'}, encoding=encoding)

                if yolo_lines:  # 포트홀이 있는 경우만 저장
                    output_img_path = os.path.join(output_img_dir, base_name + '.jpg')
                    img.save(output_img_path)
                    
                    output_label_path = os.path.join(output_label_dir, base_name + '.txt')
                    with open(output_label_path, 'w') as f:
                        f.write('\n'.join(yolo_lines))
                    print(f"[포트홀 전용] {base_name}: {len(yolo_lines)}개 포트홀 (인코딩: {encoding})")
                else:
                    print(f"[포트홀 없음] {base_name} (인코딩: {encoding})")
                    
            except Exception as e:
                print(f"[처리 실패] {base_name}: {e}")

    def process_validation_data(self, val_img_dir, val_label_dir, output_img_dir, output_label_dir):
        """검증 데이터 처리 메서드"""
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)
        
        if not os.path.exists(val_label_dir):
            print("검증 데이터 라벨 디렉토리가 존재하지 않습니다.")
            return
        
        val_files = [f for f in os.listdir(val_label_dir) if f.endswith('.json')]
        print(f"검증 데이터 파일 개수: {len(val_files)}")
        
        for json_file in val_files:
            base_name = os.path.splitext(json_file)[0]
            
            # 이미지 찾기
            img_path = None
            for ext in ['.jpg', '.png', '.jpeg']:
                test_path = os.path.join(val_img_dir, base_name + ext)
                if os.path.exists(test_path):
                    img_path = test_path
                    break
            
            if not img_path:
                print(f"[검증 이미지 없음] {base_name}")
                continue
            
            try:
                img = Image.open(img_path).convert("RGB")
                w, h = img.size
                json_data, encoding = self.load_json_label(os.path.join(val_label_dir, json_file))
                yolo_lines = self.convert_to_yolo_format(json_data, w, h, encoding=encoding)
                
                # 이미지 저장
                img.save(os.path.join(output_img_dir, base_name + '.jpg'))
                
                # 라벨 저장
                with open(os.path.join(output_label_dir, base_name + '.txt'), 'w') as f:
                    f.write('\n'.join(yolo_lines))
                print(f"[검증 데이터] {base_name}: {len(yolo_lines)}개 객체 (인코딩: {encoding})")
            except Exception as e:
                print(f"[검증 데이터 처리 실패] {base_name}: {e}")

    def analyze_dataset(self, label_dir):
        """데이터셋 분석"""
        counts = defaultdict(int)
        total_files = 0
        
        for fname in os.listdir(label_dir):
            if not fname.endswith('.txt'):
                continue
            total_files += 1
            
            with open(os.path.join(label_dir, fname), 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            cid = int(line.split()[0])
                            counts[cid] += 1
                        except:
                            continue

        print(f"총 파일 수: {total_files}")
        for cid, count in sorted(counts.items()):
            if cid < len(self.class_names):
                print(f"{self.class_names[cid]}: {count}")
        return counts


if __name__ == "__main__":
    dp = DataPreprocessor()
    base_out = "C:/Users/dadab/Desktop/processed_dataset"

    print("="*60)
    print("[1] 학습 데이터: 포트홀 라벨 병합 처리 중...")
    print("="*60)
    dp.merge_labels_with_pothole_replacement(
        orig_img_dir="C:/Users/dadab/Desktop/Final data and augmented datasets/train/images",
        orig_label_dir="C:/Users/dadab/Desktop/Final data and augmented datasets/train/labels",
        pothole_img_dir="C:/Users/dadab/Desktop/Pothole data (originalset)/images",
        pothole_label_dir="C:/Users/dadab/Desktop/Pothole data (originalset)/labels",
        output_img_dir=f"{base_out}/train/images",
        output_label_dir=f"{base_out}/train/labels"
    )

    print("\n" + "="*60)
    print("[2] 추가 포트홀 전용 데이터 처리 중...")
    print("="*60)
    dp.process_pothole_only(
        img_dir="C:/Users/dadab/Desktop/183.이륜자동차 안전 위험 시설물 데이터/01.데이터/1.Training/원천데이터_230222_add/TS_Bounding Box_26.포트홀",
        label_dir="C:/Users/dadab/Desktop/183.이륜자동차 안전 위험 시설물 데이터/01.데이터/1.Training/라벨링테이터_230222_add/TL_Bounding Box_26.포트홀",
        output_img_dir=f"{base_out}/train/images",
        output_label_dir=f"{base_out}/train/labels"
    )

    print("\n" + "="*60)
    print("[3] 검증 데이터 처리 중...")
    print("="*60)
    print("검증 데이터는 기존 라벨을 그대로 사용 (데이터 리키지 방지)")
    
    dp.process_validation_data(
        val_img_dir="C:/Users/dadab/Desktop/Final data and augmented datasets/val/images",
        val_label_dir="C:/Users/dadab/Desktop/Final data and augmented datasets/val/labels",
        output_img_dir=f"{base_out}/val/images",
        output_label_dir=f"{base_out}/val/labels"
    )

    print("\n" + "="*60)
    print("[4] 학습 데이터 분석:")
    print("="*60)
    dp.analyze_dataset(f"{base_out}/train/labels")

    print("\n" + "="*60)
    print("[5] 검증 데이터 분석:")
    print("="*60)
    dp.analyze_dataset(f"{base_out}/val/labels")
    
    print("\n" + "="*60)
    print("데이터 전처리 완료!")
    print("="*60)
