import os
import json

def merge_classes_in_json(label_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(label_dir):
        if file.endswith('.json'):
            with open(os.path.join(label_dir, file), 'r') as f:
                data = json.load(f)
            for obj in data['shapes']:
                if obj['label'] in ['lc', 'tc']:
                    obj['label'] = 'lctc'
            with open(os.path.join(output_dir, file), 'w') as f:
                json.dump(data, f, indent=2)

# 실행
merge_classes_in_json(
    r"C:\Users\dadab\Desktop\Final data and augmented datasets\train\labels",
    r"C:\Users\dadab\Desktop\Processed\train\labels"
)
merge_classes_in_json(
    r"C:\Users\dadab\Desktop\Final data and augmented datasets\val\labels",
    r"C:\Users\dadab\Desktop\Processed\val\labels"
)
