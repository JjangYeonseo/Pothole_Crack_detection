import os, cv2, numpy as np, random, shutil
from collections import defaultdict
import albumentations as A

class FinalAugmenter:
    def __init__(self, target_counts):
        self.target_counts = target_counts
        self.class_names = ['ac', 'lctc', 'pc', 'ph']
        self.context_classes = [0, 1]  # ac, lctc만 copy-paste 컨텍스트로 사용

        # 각 클래스별 최적화된 증강 파이프라인
        self.transforms = {
            0: A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
                A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=8, p=0.4),
                A.OneOf([
                    A.MotionBlur(blur_limit=5, p=0.5),
                    A.GaussianBlur(blur_limit=3, p=0.3),
                ], p=0.3),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.3),
                A.HorizontalFlip(p=0.5),
                A.Affine(translate_percent=0.05, scale=1.1, rotate=10, p=0.4),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.2),
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.15),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)),

            1: A.Compose([
                A.Affine(translate_percent=0.08, scale=1.15, rotate=15, p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.5),
                A.Perspective(scale=(0.02, 0.05), p=0.3),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.ElasticTransform(alpha=50, sigma=5, p=0.2),
                A.CLAHE(clip_limit=2.0, p=0.2),
                A.RandomToneCurve(scale=0.1, p=0.2),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)),

            2: A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
                A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02, p=0.3),
                A.CLAHE(clip_limit=2.0, p=0.3),
                A.HorizontalFlip(p=0.5),
                A.Affine(translate_percent=0.03, scale=1.05, rotate=5, p=0.3),
                A.MultiplicativeNoise(multiplier=(0.95, 1.05), p=0.2),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)),

            3: A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
                A.OneOf([
                    A.ISONoise(color_shift=(0.01, 0.08), intensity=(0.15, 0.6), p=0.4),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.3),
                ], p=0.4),
                A.OneOf([
                    A.MotionBlur(blur_limit=7, p=0.4),
                    A.GaussianBlur(blur_limit=3, p=0.3),
                ], p=0.3),
                A.HorizontalFlip(p=0.5),
                A.Affine(translate_percent=0.05, scale=1.1, rotate=8, p=0.3),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.25),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)),
        }

    def parse_label(self, path):
        """YOLO 라벨 파싱 (오류 처리 강화)"""
        if not os.path.exists(path):
            return []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                anns = []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 7:  # 최소 클래스 + 3개 좌표점
                        cls = int(parts[0])
                        coords = list(map(float, parts[1:]))
                        # 좌표 쌍으로 변환
                        points = []
                        for i in range(0, len(coords)-1, 2):
                            x, y = coords[i], coords[i+1]
                            # 좌표 범위 검증
                            if 0 <= x <= 1 and 0 <= y <= 1:
                                points.append([x, y])
                        
                        if len(points) >= 3:  # 최소 삼각형
                            anns.append({'class_id': cls, 'points': points})
                return anns
        except Exception as e:
            print(f"라벨 파싱 오류 {path}: {e}")
            return []

    def save_label(self, anns, path):
        """YOLO 라벨 저장"""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                for ann in anns:
                    if len(ann['points']) >= 3:
                        cls = ann['class_id']
                        coords = []
                        for pt in ann['points']:
                            # 좌표 범위 재검증 및 클리핑
                            x = max(0, min(1, pt[0]))
                            y = max(0, min(1, pt[1]))
                            coords.extend([str(round(x, 6)), str(round(y, 6))])
                        f.write(f"{cls} {' '.join(coords)}\n")
        except Exception as e:
            print(f"라벨 저장 오류 {path}: {e}")

    def apply_transform(self, image, anns, transform):
        """증강 변환 적용"""
        h, w = image.shape[:2]
        keypoints, class_ids, ann_indices = [], [], []
        
        # 어노테이션을 키포인트로 변환
        for ann_idx, ann in enumerate(anns):
            for pt in ann['points']:
                keypoints.append([pt[0] * w, pt[1] * h])
                class_ids.append(ann['class_id'])
                ann_indices.append(ann_idx)
        
        if not keypoints:
            return image, anns
        
        try:
            result = transform(image=image, keypoints=keypoints)
            out_img = result['image']
            out_keypoints = result['keypoints']
            
            # 변환된 키포인트를 어노테이션으로 재구성
            ann_points = defaultdict(list)
            ann_classes = {}
            
            for (x, y), cls_id, ann_idx in zip(out_keypoints, class_ids, ann_indices):
                # 유효한 좌표만 유지
                if 0 <= x < w and 0 <= y < h:
                    ann_points[ann_idx].append([x/w, y/h])
                    ann_classes[ann_idx] = cls_id
            
            # 최종 어노테이션 생성
            final_anns = []
            for ann_idx, points in ann_points.items():
                if len(points) >= 3:  # 최소 3개 점 필요
                    final_anns.append({
                        'class_id': ann_classes[ann_idx],
                        'points': points
                    })
            
            return out_img, final_anns
            
        except Exception as e:
            print(f"증강 변환 오류: {e}")
            return image, anns

    def is_valid_paste_location(self, center_x, center_y, w, h, context_centers):
        """copy-paste 위치가 적절한지 검증"""
        # 이미지 경계 체크
        if center_x < 0.1 * w or center_x > 0.9 * w:
            return False
        if center_y < 0.1 * h or center_y > 0.9 * h:
            return False
        
        # 하늘 영역 체크 (상단 30% 영역 제한)
        if center_y < 0.3 * h:
            return False
        
        # 컨텍스트 객체와의 거리 체크
        min_distance = min(w, h) * 0.05  # 최소 거리
        max_distance = min(w, h) * 0.4   # 최대 거리
        
        for ctx_x, ctx_y in context_centers:
            distance = np.sqrt((center_x - ctx_x)**2 + (center_y - ctx_y)**2)
            if min_distance <= distance <= max_distance:
                return True
        
        return False

    def copy_paste_on_context(self, base_img, base_anns, source_img, source_anns, target_cls):
        """컨텍스트 기반 copy-paste (개선된 위치 제어)"""
        h, w = base_img.shape[:2]
        
        # 컨텍스트 객체 중심점 추출
        context_centers = []
        for ann in base_anns:
            if ann['class_id'] in self.context_classes:
                points = np.array([[x*w, y*h] for x, y in ann['points']], dtype=np.float32)
                if points.shape[0] >= 3:
                    center = np.mean(points, axis=0)
                    context_centers.append(center)
        
        if not context_centers:
            return base_img, base_anns

        # 소스에서 타겟 클래스 객체 찾기
        target_objects = [ann for ann in source_anns if ann['class_id'] == target_cls]
        if not target_objects:
            return base_img, base_anns
        
        target_obj = random.choice(target_objects)
        sh, sw = source_img.shape[:2]
        
        # 소스 객체의 바운딩 박스 계산
        src_points = np.array([[x*sw, y*sh] for x, y in target_obj['points']], dtype=np.int32)
        x_min, y_min = np.min(src_points, axis=0)
        x_max, y_max = np.max(src_points, axis=0)
        
        # 패치 추출
        patch = source_img[y_min:y_max, x_min:x_max]
        if patch.size == 0:
            return base_img, base_anns
        
        # 마스크 생성
        mask = np.zeros((sh, sw), dtype=np.uint8)
        cv2.fillPoly(mask, [src_points], 255)
        mask_patch = mask[y_min:y_max, x_min:x_max]
        
        ph, pw = patch.shape[:2]
        
        # 적절한 배치 위치 찾기 (최대 10번 시도)
        for attempt in range(10):
            ref_center = random.choice(context_centers)
            # 컨텍스트 주변에 랜덤 오프셋 적용
            offset_x = random.randint(-80, 80)
            offset_y = random.randint(-40, 40)  # y 오프셋을 더 제한적으로
            
            center_x = ref_center[0] + offset_x
            center_y = ref_center[1] + offset_y
            
            # 위치 검증
            if self.is_valid_paste_location(center_x, center_y, w, h, context_centers):
                # 패치 배치 좌표 계산
                paste_x = int(center_x - pw//2)
                paste_y = int(center_y - ph//2)
                
                # 경계 조정
                paste_x = max(0, min(paste_x, w - pw))
                paste_y = max(0, min(paste_y, h - ph))
                
                # 패치 크기 검증
                if (paste_y + ph <= h and paste_x + pw <= w and 
                    base_img[paste_y:paste_y+ph, paste_x:paste_x+pw].shape == patch.shape):
                    
                    # 부드러운 블렌딩
                    alpha = cv2.GaussianBlur(mask_patch, (9, 9), sigmaX=3).astype(np.float32) / 255.0
                    alpha = np.stack([alpha] * 3, axis=-1)
                    
                    roi = base_img[paste_y:paste_y+ph, paste_x:paste_x+pw]
                    blended = (roi * (1 - alpha) + patch * alpha).astype(np.uint8)
                    base_img[paste_y:paste_y+ph, paste_x:paste_x+pw] = blended
                    
                    # 새로운 어노테이션 추가
                    new_points = []
                    for x, y in target_obj['points']:
                        new_x = (x * sw - x_min + paste_x) / w
                        new_y = (y * sh - y_min + paste_y) / h
                        # 좌표 범위 검증
                        if 0 <= new_x <= 1 and 0 <= new_y <= 1:
                            new_points.append([new_x, new_y])
                    
                    if len(new_points) >= 3:
                        base_anns.append({'class_id': target_cls, 'points': new_points})
                    
                    break
        
        return base_img, base_anns

    def safe_imread(self, path):
        """한글 파일명 안전 읽기"""
        try:
            # OpenCV의 한글 경로 문제 해결
            img_array = np.fromfile(path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"이미지 읽기 오류 {path}: {e}")
            return None

    def safe_imwrite(self, path, img):
        """한글 파일명 안전 저장"""
        try:
            # OpenCV의 한글 경로 문제 해결
            ext = os.path.splitext(path)[1]
            is_success, img_encoded = cv2.imencode(ext, img)
            if is_success:
                img_encoded.tofile(path)
                return True
            return False
        except Exception as e:
            print(f"이미지 저장 오류 {path}: {e}")
            return False

    def augment_dataset(self, img_dir, lbl_dir, out_img_dir, out_lbl_dir):
        """데이터셋 증강 실행"""
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        # 원본 데이터 복사
        print("📁 원본 데이터 복사 중...")
        for img_file in os.listdir(img_dir):
            if not img_file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                continue
            
            base_name = os.path.splitext(img_file)[0]
            img_path = os.path.join(img_dir, img_file)
            lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")
            
            # 이미지 복사
            shutil.copy2(img_path, os.path.join(out_img_dir, img_file))
            
            # 라벨 복사 (존재하는 경우만)
            if os.path.exists(lbl_path):
                shutil.copy2(lbl_path, os.path.join(out_lbl_dir, f"{base_name}.txt"))

        # 모든 이미지와 어노테이션 로드
        all_images = []
        class_img_map = defaultdict(list)
        
        for img_file in os.listdir(img_dir):
            if not img_file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                continue
                
            base_name = os.path.splitext(img_file)[0]
            lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")
            
            if os.path.exists(lbl_path):
                anns = self.parse_label(lbl_path)
                if anns:
                    all_images.append((img_file, anns))
                    for ann in anns:
                        class_img_map[ann['class_id']].append(img_file)

        # 클래스별 증강 수행
        for cls_id, target_count in self.target_counts.items():
            cls_name = self.class_names[cls_id]
            cls_images = list(set(class_img_map[cls_id]))
            
            if not cls_images:
                print(f"⚠️  {cls_name} 클래스 이미지 없음")
                continue
                
            print(f"\n🔄 {cls_name} 클래스 증강 시작 (목표: {target_count}개)")
            transform = self.transforms[cls_id]
            count = 0
            attempts = 0
            max_attempts = target_count * 20
            
            while count < target_count and attempts < max_attempts:
                try:
                    # 베이스 이미지 선택
                    base_file = random.choice(cls_images)
                    base_path = os.path.join(img_dir, base_file)
                    base_name = os.path.splitext(base_file)[0]
                    
                    # 이미지와 라벨 로드
                    img = self.safe_imread(base_path)
                    anns = self.parse_label(os.path.join(lbl_dir, f"{base_name}.txt"))
                    
                    if img is None or not anns:
                        attempts += 1
                        continue

                    # 소수 클래스에 대해 copy-paste 적용
                    if cls_id in [2, 3] and len(all_images) > 10:
                        # 소스 이미지 선택 (해당 클래스 포함)
                        source_candidates = [(f, a) for f, a in all_images 
                                           if any(ann['class_id'] == cls_id for ann in a)]
                        if source_candidates:
                            src_file, src_anns = random.choice(source_candidates)
                            src_path = os.path.join(img_dir, src_file)
                            src_img = self.safe_imread(src_path)
                            
                            if src_img is not None:
                                img, anns = self.copy_paste_on_context(
                                    img.copy(), anns.copy(), src_img, src_anns, cls_id
                                )

                    # 일반 증강 적용
                    aug_img, aug_anns = self.apply_transform(img, anns, transform)
                    
                    # 타겟 클래스가 포함되어 있는지 확인
                    if not any(ann['class_id'] == cls_id for ann in aug_anns):
                        attempts += 1
                        continue

                    # 결과 저장
                    out_name = f"aug_{cls_name}_{count:05d}"
                    out_img_path = os.path.join(out_img_dir, f"{out_name}.jpg")
                    out_lbl_path = os.path.join(out_lbl_dir, f"{out_name}.txt")
                    
                    if self.safe_imwrite(out_img_path, aug_img):
                        self.save_label(aug_anns, out_lbl_path)
                        count += 1
                        
                        # 진행률 표시
                        if count % 200 == 0:
                            print(f"  → 진행률: {count}/{target_count}")
                    
                except Exception as e:
                    print(f"증강 처리 오류: {e}")
                
                attempts += 1
            
            success_rate = count / attempts * 100 if attempts > 0 else 0
            print(f"✅ {cls_name} 완료: {count}개 생성 (성공률: {success_rate:.1f}%)")

        print(f"\n🎉 모든 증강 작업 완료!")

if __name__ == "__main__":
    # 목표 카운트 설정
    target_counts = {
        0: 1400,  # ac
        1: 2100,  # lctc  
        2: 1700,  # pc - 가장 많이 증강 필요
        3: 500    # ph - 최소 증강
    }
    
    base_path = "C:/Users/dadab/Desktop/processed_dataset"
    output_path = "C:/Users/dadab/Desktop/augmented_dataset"

    print("🚀 최종 데이터 증강 시작...")
    print(f"목표: ac(+{target_counts[0]}), lctc(+{target_counts[1]}), pc(+{target_counts[2]}), ph(+{target_counts[3]})")

    augmenter = FinalAugmenter(target_counts)
    augmenter.augment_dataset(
        f"{base_path}/train/images", 
        f"{base_path}/train/labels", 
        f"{output_path}/train/images", 
        f"{output_path}/train/labels"
    )

    # 검증 데이터 복사
    print("\n📦 검증 데이터 복사 중...")
    val_dirs = [f"{output_path}/val/images", f"{output_path}/val/labels"]
    for dir_path in val_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # 검증 이미지 복사
    for file_name in os.listdir(f"{base_path}/val/images"):
        shutil.copy2(
            os.path.join(f"{base_path}/val/images", file_name),
            os.path.join(f"{output_path}/val/images", file_name)
        )
    
    # 검증 라벨 복사
    for file_name in os.listdir(f"{base_path}/val/labels"):
        shutil.copy2(
            os.path.join(f"{base_path}/val/labels", file_name),
            os.path.join(f"{output_path}/val/labels", file_name)
        )
    
    print("✅ 모든 작업 완료!")
    print("\n📊 예상 최종 분포:")
    print(f"ac: ~{1606 + target_counts[0]} ({target_counts[0]} 추가)")
    print(f"lctc: ~{1405 + target_counts[1]} ({target_counts[1]} 추가)")
    print(f"pc: ~{324 + target_counts[2]} ({target_counts[2]} 추가)")
    print(f"ph: ~{13850 + target_counts[3]} ({target_counts[3]} 추가)")
