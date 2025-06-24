import os, json
import xml.etree.ElementTree as ET
from PIL import Image

def convert_voc_to_json(xml_dir, image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(xml_dir):
        if file.endswith('.xml'):
            tree = ET.parse(os.path.join(xml_dir, file))
            root = tree.getroot()

            image_name = root.find('filename').text
            img_path = os.path.join(image_dir, image_name)
            if not os.path.exists(img_path):
                continue

            img = Image.open(img_path)
            width, height = img.size

            shapes = []
            for obj in root.findall('object'):
                if obj.find('name').text.lower() != 'pothole':
                    continue
                bbox = obj.find('bndbox')
                x1, y1 = int(bbox.find('xmin').text), int(bbox.find('ymin').text)
                x2, y2 = int(bbox.find('xmax').text), int(bbox.find('ymax').text)
                shapes.append({
                    "label": "ph",
                    "points": [[x1, y1], [x2, y2]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                })

            json_data = {
                "version": "4.5.6",
                "flags": {},
                "shapes": shapes,
                "imagePath": image_name,
                "imageHeight": height,
                "imageWidth": width
            }

            json_name = file.replace(".xml", ".json")
            with open(os.path.join(output_dir, json_name), "w") as f:
                json.dump(json_data, f, indent=2)

# 실행
convert_voc_to_json(
    r"C:\Users\dadab\Desktop\archive\annotations",
    r"C:\Users\dadab\Desktop\archive\images",
    r"C:\Users\dadab\Desktop\Processed\external_ph_labels"
)
