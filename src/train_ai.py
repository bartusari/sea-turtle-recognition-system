import tensorflow as tf
import os
import sys
import pickle
import numpy as np
from sklearn.utils import class_weight

current_file_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_path) if "src" in current_file_path else current_file_path
sys.path.append(project_root)

from src.model_builder import create_turtle_model

TRAIN_DIR = os.path.join(project_root, "project_data", "train")
VAL_DIR = os.path.join(project_root, "project_data", "val")

WEIGHTS_SAVE_PATH = os.path.join(project_root, "turtle_weights.weights.h5") 
PKL_SAVE_PATH = os.path.join(project_root, "class_names.pkl")

IMG_SIZE = (224, 224)
BATCH_SIZE = 8 

def main():
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print(f"❌ HATA: Eğitim veya Doğrulama klasörü bulunamadı!")
        return

    print("\n[1/4] Veriler yükleniyor...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, 
        image_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        label_mode='int',
        shuffle=True
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR, 
        image_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        label_mode='int',
        shuffle=False
    )
    
    class_names = train_ds.class_names
    num_classes = len(class_names)

    print("\n[2/4] Sınıf ağırlıkları hesaplanıyor...")
    y_train = []
    for _, labels in train_ds.unbatch():
        y_train.append(labels.numpy())
    y_train = np.array(y_train)
    
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(weights))

    print("\n[3/4] Model inşa ediliyor...")
    model, _ = create_turtle_model(num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), # 5e-4 çok yüksek olabilir, 5e-5 daha stabil
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"\n[4/4] Eğitim Başlıyor (90 Epoch)...")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=90,
        class_weight=class_weight_dict,
    )

    print("\n[KAYIT] Ağırlıklar ve sınıf isimleri kaydediliyor...")
    
    model.save_weights(WEIGHTS_SAVE_PATH)
    
    with open(PKL_SAVE_PATH, "wb") as f:
        pickle.dump(class_names, f)

    print(f"\n✅ [TAMAMLANDI]")
    print(f"📍 Ağırlıklar: {WEIGHTS_SAVE_PATH}")
    print(f"📍 Sınıf İsimleri: {PKL_SAVE_PATH}")

if __name__ == "__main__":
    main()