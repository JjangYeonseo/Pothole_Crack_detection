import os, json
import numpy as np
from PIL import Image
import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3), A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.4), A.MotionBlur(p=0.2), A.GaussNoise(p=0.3),
    A.GridDistortion(p=0.2), A.ElasticTransform(p=0.2), A.RandomShadow(p=0.2),
    A.RandomFog(p=0.1), A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=20, p=0.6)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

def augment_labelme_json(img_path, json_path, out_img_dir, out_lbl_dir, num_aug):
    with open(json_path, 'r') as f:
        data = json.load(f)

    image = np.array(Image.open(img_path).convert("RGB"))
    h, w = image.shape[:2]
    bboxes, labels = [], []

    for s in data['shapes']:
        xs, ys = [p[0] for p in s['points']], [p[1] for p in s['points']]
        x1, y1, x2, y2 = max(0, min(xs)), max(0, min(ys)), min(w-1, max(xs)), min(h-1, max(ys))
        if x2 - x1 > 1 and y2 - y1 > 1:
            bboxes.append([x1, y1, x2, y2])
            labels.append(s['label'])

    base = os.path.splitext(os.path.basename(img_path))[0]
    for i in range(num_aug):
        result = transform(image=image, bboxes=bboxes, category_ids=labels)
        Image.fromarray(result['image']).save(os.path.join(out_img_dir, f"{base}_aug{i}.jpg"))

        new_shapes = [{
            "label": lbl,
            "points": [[x1, y1], [x2, y2]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        } for (x1, y1, x2, y2), lbl in zip(result['bboxes'], result['category_ids'])]

        aug_json = {
            "version": "4.5.6",
            "flags": {}, "shapes": new_shapes,
            "imagePath": f"{base}_aug{i}.jpg",
            "imageHeight": result['image'].shape[0],
            "imageWidth": result['image'].shape[1]
        }
        with open(os.path.join(out_lbl_dir, f"{base}_aug{i}.json"), 'w') as f:
            json.dump(aug_json, f, indent=2)

def augment_subset(img_dir, lbl_dir, out_img_dir, out_lbl_dir, target_labels=('ph', 'lctc'), num_aug_target=10, num_aug_other=4):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    for file in os.listdir(lbl_dir):
        if file.endswith('.json'):
            with open(os.path.join(lbl_dir, file), 'r') as f:
                labels = [s['label'] for s in json.load(f)['shapes']]
            num_aug = num_aug_target if any(l in target_labels for l in labels) else num_aug_other
            augment_labelme_json(
                os.path.join(img_dir, file.replace('.json', '.jpg')),  # 또는 .png 확장자 확인 필요
                os.path.join(lbl_dir, file),
                out_img_dir, out_lbl_dir,
                num_aug=num_aug
            )

# 실행
augment_subset(
    image_dir=r"C:\Users\dadab\Desktop\MergedDataset\train\images",
    label_dir=r"C:\Users\dadab\Desktop\MergedDataset\train\labels",
    output_img_dir=r"C:\Users\dadab\Desktop\MergedDataset\augmented\images",
    output_lbl_dir=r"C:\Users\dadab\Desktop\MergedDataset\augmented\labels"
)
