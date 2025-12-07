import os, math, cv2, random, shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt


def gray_to_rgb(x):
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    return x.astype('float32')


def get_class_balanced_alpha_beta(counts, beta=0.9999):
    eff_nums = [(1 - beta**c) / (1 - beta) for c in counts]
    alpha = [1 - e/sum(eff_nums) for e in eff_nums]
    s = sum(alpha)
    alpha = [a/s for a in alpha]
    return alpha

def class_balanced_focal_loss(counts, beta=0.9999, gamma=2.0):
    alpha_list = get_class_balanced_alpha_beta(counts, beta)
    alpha = alpha_list[1]
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow((1.0 - p_t), gamma)
        return tf.reduce_mean(alpha_factor * modulating_factor * bce)
    return loss_fn

def create_train_val_split(data_dir, val_ratio=0.15):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    if not os.path.exists(val_dir):
        os.makedirs(val_dir, exist_ok=True)
    for class_name in os.listdir(train_dir):
        class_train_dir = os.path.join(train_dir, class_name)
        class_val_dir = os.path.join(val_dir, class_name)
        os.makedirs(class_val_dir, exist_ok=True)
        images = os.listdir(class_train_dir)
        n_val = max(1, int(len(images) * val_ratio))
        val_samples = random.sample(images, n_val)
        for img_file in val_samples:
            src = os.path.join(class_train_dir, img_file)
            dst = os.path.join(class_val_dir, img_file)
            shutil.move(src, dst)
    print("Validation split created.")

def oversample_minority_class(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    class_counts = {cls: len(os.listdir(os.path.join(train_dir, cls))) for cls in os.listdir(train_dir)}
    max_count = max(class_counts.values())
    for cls, count in class_counts.items():
        cls_dir = os.path.join(train_dir, cls)
        if count < max_count:
            images = os.listdir(cls_dir)
            n_to_add = max_count - count
            for i in range(n_to_add):
                src = os.path.join(cls_dir, random.choice(images))
                dst = os.path.join(cls_dir, f"copy_{i}_{random.randint(0,10000)}.jpeg")
                shutil.copy(src, dst)
    print("Oversampling done.")

def get_chestxray_generators(data_dir, img_size=(224,224), batch_size=16):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.02,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size,
        class_mode='binary', color_mode='rgb', shuffle=True
    )

    val_gen = test_datagen.flow_from_directory(
        val_dir, target_size=img_size, batch_size=batch_size,
        class_mode='binary', color_mode='rgb', shuffle=False
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size,
        class_mode='binary', color_mode='rgb', shuffle=False
    )

    return train_gen, val_gen, test_gen

def compute_class_counts(generator):
    classes = generator.classes
    values, counts = np.unique(classes, return_counts=True)
    counts_arr = [0,0]
    for v,c in zip(values, counts): counts_arr[int(v)] = int(c)
    return counts_arr


def build_model(input_shape=(224,224,3)):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=input_shape,
        weights=None
    )
    base.trainable = True
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs=base.input, outputs=outputs)

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            return layer.name
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    if img_tensor.ndim == 3:
        img_tensor = tf.expand_dims(img_tensor, axis=0)
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:,0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0] * pooled_grads
    heatmap = tf.reduce_mean(conv_outputs, axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)+1e-6
    return heatmap

def evaluate_model(model, test_gen, batch_size):
    steps = math.ceil(test_gen.samples / batch_size)
    probs = model.predict(test_gen, steps=steps).ravel()
    y_true = test_gen.classes
    y_pred = (probs>0.5).astype(int)
    print(classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys())))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    auc_score = roc_auc_score(y_true, probs) if len(np.unique(y_pred))>1 else 0.5
    print("Test AUC:", auc_score)
    fpr,tpr,_ = roc_curve(y_true, probs)
    plt.figure(); plt.plot(fpr,tpr,label=f'ROC (AUC={auc_score:.3f})'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.title('ROC Curve'); plt.savefig('roc_curve.png')
    precision,recall,_ = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall, precision)
    plt.figure(); plt.plot(recall, precision,label=f'PR (AUC={pr_auc:.3f})'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend(); plt.title('PR Curve'); plt.savefig('pr_curve.png')

    return y_true, probs, y_pred, auc_score

if __name__=="__main__":
    data_dir = r"D:\chest_ray" 
    img_size, batch_size, epochs, lr = 224, 16, 30, 1e-4  
    if not os.path.exists(os.path.join(data_dir, 'val')):
        create_train_val_split(data_dir, val_ratio=0.15)
    oversample_minority_class(data_dir)
    train_gen, val_gen, test_gen = get_chestxray_generators(data_dir,(img_size,img_size),batch_size)
    counts = compute_class_counts(train_gen)
    print("Training class counts:", counts)

    model = build_model(input_shape=(img_size,img_size,3))
    loss_fn = class_balanced_focal_loss(counts)
    optimizer = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    model.summary()
    total = sum(counts)
    class_weight = {0: total/(2*counts[0]), 1: total/(2*counts[1])}
    cb = [
        callbacks.ModelCheckpoint(
            'pneunet_best_val_acc.h5', 
            monitor='val_accuracy', 
            save_best_only=True, 
            mode='max', 
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            'pneunet_best_val_loss.h5', 
            monitor='val_loss', 
            save_best_only=True, 
            mode='min', 
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=6, 
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=3, 
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=cb, class_weight=class_weight)
    y_true, probs, y_pred, auc_score = evaluate_model(model, test_gen, batch_size)

    try:
        last_conv = find_last_conv_layer(model)
        sample_paths = test_gen.filepaths[:6]
        for i,p in enumerate(sample_paths):
            img = cv2.imread(p)
            if img is None: continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb,(img_size,img_size))
            img_input = np.expand_dims(img_resized.astype('float32')/255.0, axis=0)
            heatmap = make_gradcam_heatmap(img_input, model, last_conv)
            heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))
            heatmap = np.uint8(255*heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed = cv2.addWeighted(img_rgb,0.6,heatmap,0.4,0)
            cv2.imwrite(f'gradcam_{i}.jpg', cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
        print("Saved Grad-CAM images.")
    except Exception as e:
        print("Grad-CAM error:", e)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

test_datagen = ImageDataGenerator(rescale=1./255)
model = load_model("pneunet_best_val_acc.h5", compile=False)

test_generator = test_datagen.flow_from_directory(
    "D:/chest_ray/test",   
    target_size=(224, 224),
    batch_size=16,
    class_mode="binary",
    shuffle=False
)

y_pred_proba = model.predict(test_generator, verbose=1)   
y_pred = (y_pred_proba > 0.5).astype("int32").flatten()
y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Pneumonia"],
            yticklabels=["Normal", "Pneumonia"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

fpr, tpr, _ = roc_curve(y_true, y_pred_proba.ravel())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label="ROC curve (AUC = %0.3f)" % roc_auc)
plt.plot([0,1], [0,1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()

precision, recall, _ = precision_recall_curve(y_true, y_pred_proba.ravel())
plt.figure(figsize=(6,5))
plt.plot(recall, precision, color="green", lw=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

try:
    with open("trainHistoryDict.pkl", "rb") as file_pi:
        history = pickle.load(file_pi)

    plt.figure(figsize=(6,5))
    plt.plot(history["accuracy"], label="Train Acc")
    plt.plot(history["val_accuracy"], label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.show()

    plt.figure(figsize=(6,5))
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.show()
except FileNotFoundError:
    print("⚠️ trainHistoryDict.pkl not found. Skipping training history plots.")
