import os
import shutil
import random
import sys

def main():
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_path) if "src" in current_file_path else current_file_path

    source_path = os.path.join(project_root, "dataset", "images")
    target_path = os.path.join(project_root, "project_data")

    turtle_limit = 20    
    train_ratio = 0.6
    val_ratio = 0.3
    test_ratio = 0.1

    if not os.path.exists(source_path):
        print(f"❌ HATA: Kaynak klasör bulunamadı!")
        print(f"📍 Şuraya bakıldı: {source_path}")
        print("💡 Lütfen 'dataset/images' klasörünün proje ana dizininde olduğundan emin ol.")
        return

    if os.path.exists(target_path):
        print(f"♻️ Eski {target_path} klasörü temizleniyor...")
        shutil.rmtree(target_path)

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(target_path, split), exist_ok=True)

    all_turtles = sorted([d for d in os.listdir(source_path) 
                         if os.path.isdir(os.path.join(source_path, d)) and not d.startswith('.')])

    selected_turtles = all_turtles[:turtle_limit]

    print(f"🚀 Toplam {len(selected_turtles)} kaplumbağa (sınıf) için dağıtım işlemi başladı...")

    count = 0
    for turtle in selected_turtles:
        turtle_src_dir = os.path.join(source_path, turtle)
        
        images = [f for f in os.listdir(turtle_src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(images) < 3:
            continue

        random.shuffle(images)
        
        n = len(images)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }
        
        for split_name, split_files in splits.items():
            if not split_files:
                continue
                
            target_dir = os.path.join(target_path, split_name, turtle)
            os.makedirs(target_dir, exist_ok=True)
            
            for img_file in split_files:
                src_file = os.path.join(turtle_src_dir, img_file)
                dst_file = os.path.join(target_dir, img_file)
                shutil.copy(src_file, dst_file)
                count += 1

    print("\n✅ Veri seti hazırlama tamamlandı!")
    print(f"📍 Yeni Veri Yolu: {os.path.abspath(target_path)}")
    print(f"📊 Toplam {count} resim, {len(selected_turtles)} sınıfa dağıtıldı.")

if __name__ == "__main__":
    main()