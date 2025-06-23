import os
import json
import yaml
import time
import shutil
import random
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import cv2
from ultralytics import YOLO
import albumentations as A

# ───────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────
BASE_PATH        = Path(r"C:\Users\dadab\Desktop\Final data and augmented datasets")
TRAIN_IMG_DIR    = BASE_PATH / "train" / "images"
TRAIN_LABELS_DIR = BASE_PATH / "train" / "labels"
VAL_IMG_DIR      = BASE_PATH / "val" / "images"
VAL_LABELS_DIR   = BASE_PATH / "val" / "labels"
CLASS_NAMES      = ['ac','lc','tc','pc','ph']
NUM_CLASSES      = len(CLASS_NAMES)
AUG_TARGET_RATIO = 1.0    # minority-class target ratio relative to majority
MIN_AREA         = 1      # min bbox area (pixels)
MIN_VISIBILITY   = 0.1    # min bbox visibility

# Training params
epochs           = 200
img_size         = 640
batch_size       = 16
patience         = 50
CHECKPOINT_DIR   = BASE_PATH / 'checkpoints'
CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR      = BASE_PATH / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# ───────────────────────────────────────────────────────────
# 1. Convert JSON → YOLO TXT
# ───────────────────────────────────────────────────────────
def convert_json_to_yolo(json_path, img_w, img_h):
    data = json.load(open(json_path, 'r', encoding='utf-8'))
    shapes = data.get('shapes') or data.get('annotations') or []
    lines = []
    for obj in shapes:
        label = obj.get('label') or obj.get('class')
        if label not in CLASS_NAMES:
            continue
        cid = CLASS_NAMES.index(label)
        pts = obj.get('points') or obj.get('polygon')
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        x1, y1 = min(xs), min(ys)
        w, h = max(xs)-x1, max(ys)-y1
        if w <= 0 or h <= 0:
            continue
        cx, cy = (x1 + w/2)/img_w, (y1 + h/2)/img_h
        nw, nh = w/img_w, h/img_h
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def convert_all_labels():
    print("🔄 Converting JSON to YOLO TXT (if missing)...")
    for img_dir, lbl_dir in [(TRAIN_IMG_DIR, TRAIN_LABELS_DIR), (VAL_IMG_DIR, VAL_LABELS_DIR)]:
        for img_path in img_dir.glob('*.jpg'):
            txt_path = lbl_dir / f"{img_path.stem}.txt"
            if txt_path.exists():
                continue
            json_path = lbl_dir / f"{img_path.stem}.json"
            if not json_path.exists():
                txt_path.write_text('')
                continue
            img = cv2.imread(str(img_path)); h, w = img.shape[:2]
            lines = convert_json_to_yolo(json_path, w, h)
            txt_path.write_text("\n".join(lines))
    print("✅ JSON→TXT conversion done.")

# ───────────────────────────────────────────────────────────
# 2. Analyze dataset
# ───────────────────────────────────────────────────────────
def analyze_dataset(lbl_dir):
    cnt = Counter()
    for txt in lbl_dir.glob('*.txt'):
        for line in txt.read_text().splitlines():
            if not line.strip():
                continue
            cid = int(float(line.split()[0]))
            cnt[cid] += 1
    return cnt

# ───────────────────────────────────────────────────────────
# 3. Create YAML
# ───────────────────────────────────────────────────────────
def create_yaml():
    cfg = {
        'path': str(BASE_PATH),
        'train': 'train/images',
        'val': 'val/images',
        'nc': NUM_CLASSES,
        'names': CLASS_NAMES
    }
    yaml_path = BASE_PATH / 'dataset.yaml'
    yaml_path.write_text(yaml.dump(cfg))
    print(f"✅ dataset.yaml created: {yaml_path}")
    return yaml_path

# ───────────────────────────────────────────────────────────
# 4. Minority-class augmentation
# ───────────────────────────────────────────────────────────
def augment_minority_classes():
    print("🔄 Minority-class augmentation...")
    # clear previous augmented files to avoid exponential growth
    for f in TRAIN_IMG_DIR.glob('*_aug*.jpg'):
        os.remove(f)
    for f in TRAIN_LABELS_DIR.glob('*_aug*.txt'):
        os.remove(f)

    cnt = analyze_dataset(TRAIN_LABELS_DIR)
    maj = max(cnt.values(), default=0)
    target = int(maj * AUG_TARGET_RATIO)

    # only original stems (exclude already augmented)
    orig_stems = [p.stem for p in TRAIN_LABELS_DIR.glob('*.txt') if '_aug' not in p.stem]

    # augmentation pipeline
    aug = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7),
        A.CoarseDropout(p=0.5),  # default holes
        A.ElasticTransform(p=0.3),
        A.GaussNoise(p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', min_area=MIN_AREA,
                                min_visibility=MIN_VISIBILITY,
                                label_fields=['labels']))

    # group original files by class
    class_files = defaultdict(list)
    for stem in orig_stems:
        labels = [int(float(l.split()[0])) for l in (TRAIN_LABELS_DIR/f"{stem}.txt").read_text().splitlines() if l]
        for c in set(labels):
            class_files[c].append(stem)

    skipped, gen = 0, 0
    for cid in range(NUM_CLASSES):
        need = target - cnt.get(cid, 0)
        if need <= 0:
            continue
        stems = class_files.get(cid, [])
        if not stems:
            print(f"⚠️ No original samples for class '{CLASS_NAMES[cid]}', skipping.")
            continue
        for i in range(need):
            base = random.choice(stems)
            img = cv2.imread(str(TRAIN_IMG_DIR/f"{base}.jpg"))
            bboxes, labels = [], []
            for line in (TRAIN_LABELS_DIR/f"{base}.txt").read_text().splitlines():
                p = line.split(); labels.append(int(float(p[0]))); bboxes.append(list(map(float, p[1:5])))
            try:
                out = aug(image=img, bboxes=bboxes, labels=labels)
            except Exception:
                skipped += 1
                continue
            stem_aug = f"{base}_aug{cid}_{i}"
            cv2.imwrite(str(TRAIN_IMG_DIR/f"{stem_aug}.jpg"), out['image'])
            with open(TRAIN_LABELS_DIR/f"{stem_aug}.txt", 'w') as f:
                for l, bb in zip(out['labels'], out['bboxes']):
                    f.write(f"{l} {' '.join(map(str, bb))}\n")
            gen += 1
    print(f"⚠️ Skipped: {skipped}, Generated: {gen}")

# ───────────────────────────────────────────────────────────
# 5. Training (detection)
# ───────────────────────────────────────────────────────────
def train():
    yaml_path = create_yaml()
    model = YOLO('yolov8m.pt')  # use medium model
    ckpts = list(CHECKPOINT_DIR.glob('*.pt'))
    resume = str(max(ckpts, key=os.path.getctime)) if ckpts else False
    print(f"▶️ Training start: resume={resume}")
    t0 = time.time()
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        patience=patience,
        optimizer='AdamW',
        lr0=1e-3,
        lrf=0.01,
        weight_decay=1e-4,
        mosaic=1.0,
        mixup=0.5,
        project=str(RESULTS_DIR),
        name='yolov8m_training',
        exist_ok=True,
        resume=resume,
        val=True,
        plots=True
    )
    elapsed = time.time() - t0
    print(f"✅ Training done in {elapsed/60:.1f} min")
    last = RESULTS_DIR/'yolov8m_training'/'weights'/'last.pt'
    if last.exists():
        shutil.copy(last, CHECKPOINT_DIR/'last.pt')
    return results

# ───────────────────────────────────────────────────────────
# 6. Evaluation
# ───────────────────────────────────────────────────────────
def evaluate():
    best = RESULTS_DIR/'yolov8m_training'/'weights'/'best.pt'
    if not best.exists():
        print("❌ best.pt not found"); return
    model = YOLO(str(best))
    res = model.val(data=str(BASE_PATH/'dataset.yaml'), save_json=True)
    print("\n📊 Per-class mAP50:")
    for i, map50 in enumerate(res.box.map50s or []):
        print(f"  {CLASS_NAMES[i]}: {map50:.3f}")
    metrics_path = RESULTS_DIR/'yolov8m_training'/'val'/'metrics.json'
    if metrics_path.exists():
        metrics = json.load(open(metrics_path, 'r', encoding='utf-8'))
        per_cls = metrics.get('per_class_metrics', [])
        if per_cls:
            print("\n📈 Per-class Precision & Recall:")
            for i, cls in enumerate(per_cls):
                p = cls.get('precision', 0)
                r = cls.get('recall', 0)
                name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i)
                print(f"  {name}: Precision {p:.3f}, Recall {r:.3f}")
    else:
        print("⚠️ metrics.json not found; Precision/Recall 출력 불가")

# ───────────────────────────────────────────────────────────
# Main pipeline
# ───────────────────────────────────────────────────────────
def main():
    start = time.time()
    convert_all_labels()
    print(f"📝 Before: {analyze_dataset(TRAIN_LABELS_DIR)}")
    augment_minority_classes()
    print(f"🎯 After : {analyze_dataset(TRAIN_LABELS_DIR)}")
    train()
    evaluate()
    total = (time.time()-start)/60
    print(f"🏁 Total pipeline time: {total:.1f} min")

if __name__ == '__main__':
    main()
