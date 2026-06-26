import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, Callback

# manuel stopping: to stop and save create stop.txt in the file
class ManuelDurdurmaCallback(Callback):
    def __init__(self, dosya_adi="stop.txt"):
        super(ManuelDurdurmaCallback, self).__init__()
        self.dosya_adi = dosya_adi
        # Eğitim başlarken eğer eski bir stop.txt varsa siliyoruz
        if os.path.exists(self.dosya_adi):
            os.remove(self.dosya_adi)
            
    def on_epoch_end(self, epoch, logs=None):
        # Her epoch bitiminde klasörde stop.txt var mı diye bakar
        if os.path.exists(self.dosya_adi):
            print(f"\n🛑 '{self.dosya_adi}' dosyası algılandı! Eğitim {epoch+1}. epochta kullanıcı tarafından durduruluyor...")
            self.model.stop_training = True
            os.remove(self.dosya_adi)

data = tf.keras.utils.image_dataset_from_directory(
    'data', 
    image_size=(128, 128), 
    batch_size=32,
    shuffle=True, 
    seed=42
)

data = data.map(lambda x, y: (x / 255.0, y))

total_batches = len(data)
train_size = int(total_batches * 0.7)
validation_size = int(total_batches * 0.2) 

train = data.take(train_size)
val = data.skip(train_size).take(validation_size)
test = data.skip(train_size + validation_size) 

AUTOTUNE = tf.data.AUTOTUNE
train = train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val = val.cache().prefetch(buffer_size=AUTOTUNE)
test = test.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
])

model = Sequential()
model.add(tf.keras.layers.Input(shape=(128, 128, 3)))
model.add(data_augmentation)

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5)) 
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="adam", loss=tf.losses.BinaryCrossentropy(), metrics=["accuracy"])

logdir = "logs"
tensorboard_callback = TensorBoard(log_dir=logdir)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

manuel_durdur = ManuelDurdurmaCallback()

history = model.fit(
    train, 
    epochs=27, 
    validation_data=val, 
    callbacks=[tensorboard_callback, early_stopping_callback, manuel_durdur]
)

test_loss, test_accuracy = model.evaluate(test)
print(f"Test Kaybı (Loss): {test_loss:.4f}")
print(f"Test Doğruluğu (Accuracy): {test_accuracy:.4f}")

model.save("cat_dog_model.keras")
print("\nModel 'cat_dog_model.keras' adıyla başarıyla kaydedildi.")