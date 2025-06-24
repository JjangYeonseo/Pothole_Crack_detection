import os, json
from PIL import Image
from ultralytics import YOLO

CLASS_MAP = {"ac": 0, "lctc": 1, "pc": 2, "ph": 3}
CLASS_NAMES = list(CLASS_MAP.keys())

def convert_labelme_to_yolo(json_dir, image_dir, out_img_dir, out_lbl_dir):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    for file in os.listdir(json_dir):
        if file.endswith('.json'):
            with open(os.path.join(json_dir, file)) as f:
                data = json.load(f)
            img_path = os.path.join(image_dir, data['imagePath'])
            if not os.path.exists(img_path):
                continue

            image = Image.open(img_path).convert("RGB")
            w, h = image.size
            image.save(os.path.join(out_img_dir, os.path.basename(img_path)))

            with open(os.path.join(out_lbl_dir, file.replace('.json', '.txt')), 'w') as f_out:
                for s in data['shapes']:
                    label = s['label']
                    if label not in CLASS_MAP:
                        continue
                    points = s['points']
                    if s['shape_type'] == 'rectangle':
                        (x1, y1), (x2, y2) = points
                        points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    norm = [round(x/w, 6) if i%2==0 else round(x/h, 6) for pt in points for i,x in enumerate(pt)]
                    f_out.write(f"{CLASS_MAP[label]} " + " ".join(map(str, norm)) + "\n")

# 학습, 증강 데이터 변환
convert_labelme_to_yolo(
    r"C:\Users\dadab\Desktop\MergedDataset\train\labels",
    r"C:\Users\dadab\Desktop\MergedDataset\train\images",
    r"C:\Users\dadab\Desktop\YOLODataset\images\train",
    r"C:\Users\dadab\Desktop\YOLODataset\labels\train"
)
convert_labelme_to_yolo(
    r"C:\Users\dadab\Desktop\MergedDataset\augmented\labels",
    r"C:\Users\dadab\Desktop\MergedDataset\augmented\images",
    r"C:\Users\dadab\Desktop\YOLODataset\images\train",
    r"C:\Users\dadab\Desktop\YOLODataset\labels\train"
)
convert_labelme_to_yolo(
    r"C:\Users\dadab\Desktop\MergedDataset\val\labels",
    r"C:\Users\dadab\Desktop\MergedDataset\val\images",
    r"C:\Users\dadab\Desktop\YOLODataset\images\val",
    r"C:\Users\dadab\Desktop\YOLODataset\labels\val"
)

# YAML 파일 생성
def create_yaml():
    path = r"C:\Users\dadab\Desktop\YOLODataset\data.yaml"
    with open(path, 'w') as f:
        f.write("""\
train: C:/Users/dadab/Desktop/YOLODataset/images/train
val: C:/Users/dadab/Desktop/YOLODataset/images/val

nc: 4
names: ['ac', 'lctc', 'pc', 'ph']
""")
    return path

# 학습 실행
def train_and_evaluate(yaml_path):
    model = YOLO("yolov8n-seg.pt")
    model.train(
        data=yaml_path,
        epochs=200,
        imgsz=640,
        project=r"C:\Users\dadab\Desktop\yolo_train_results",
        name="exp1",
        save_period=10,
        patience=50,
        resume=False
    )
    metrics = model.val()
    print("\n📊 클래스별 정확도:")
    for idx, name in enumerate(CLASS_NAMES):
        try:
            p, r = metrics.box.p[idx], metrics.box.r[idx]
            m50, m95 = metrics.box.map50[idx], metrics.box.map[idx]
            print(f"{name}: P={p:.3f}, R={r:.3f}, mAP50={m50:.3f}, mAP50-95={m95:.3f}")
        except IndexError:
            print(f"{name}: (검출 없음)")

# 실행
yaml_path = create_yaml()
train_and_evaluate(yaml_path)
