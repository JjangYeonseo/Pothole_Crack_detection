import os, shutil

def copy_data(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    for file in os.listdir(src_img_dir):
        if file.endswith(('.jpg', '.png')):
            shutil.copy(os.path.join(src_img_dir, file), os.path.join(dst_img_dir, file))
    for file in os.listdir(src_lbl_dir):
        if file.endswith('.json'):
            shutil.copy(os.path.join(src_lbl_dir, file), os.path.join(dst_lbl_dir, file))

def run_merge_pipeline():
    base = r"C:\Users\dadab\Desktop\MergedDataset"
    copy_data(
        r"C:\Users\dadab\Desktop\Final data and augmented datasets\train\images",
        r"C:\Users\dadab\Desktop\Processed\train\labels",
        os.path.join(base, "train", "images"),
        os.path.join(base, "train", "labels")
    )
    copy_data(
        r"C:\Users\dadab\Desktop\Final data and augmented datasets\val\images",
        r"C:\Users\dadab\Desktop\Processed\val\labels",
        os.path.join(base, "val", "images"),
        os.path.join(base, "val", "labels")
    )
    copy_data(
        r"C:\Users\dadab\Desktop\archive\images",
        r"C:\Users\dadab\Desktop\Processed\external_ph_labels",
        os.path.join(base, "train", "images"),
        os.path.join(base, "train", "labels")
    )
    print("✅ 데이터 병합 완료")

# 실행
run_merge_pipeline()
