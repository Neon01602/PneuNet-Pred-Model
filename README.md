# **PneuNet — Pneumonia Detection from Chest X-rays (Overview & Usefulness)**

## **🔍 What This Project Covers**
- Pneumonia detection using medical X-ray imaging  
- DenseNet, EfficientNet, and a CNN+RNN hybrid architecture  
- Ensemble prediction strategy for better reliability  
- Explainability using Grad-CAM heatmaps  
- Model comparison using ROC curves, precision curves, and confusion matrices  
- Clinical usefulness and decision-support justification  

---

## **🧠 Approach (High-Level & Easy to Understand)**

PneuNet uses **three different deep learning models** — DenseNet, EfficientNet, and a custom CNN+RNN hybrid — each learning lung patterns differently.  
Instead of trusting one model, all three predictions are **combined (ensembled)** to produce a more stable and accurate output.

This helps reduce:
- False negatives (missing pneumonia cases)  
- Overfitting from any single model  
- Bias caused by a single architecture’s limitations  

The ensemble therefore acts like **multiple expert opinions**, improving overall reliability for medical screening.

---

## **🩺 Why This Project Is Useful in Healthcare**

### **1. More Reliable Diagnoses**
Using an ensemble ensures the system does not depend on one model’s weaknesses.  
This increases:
- Pneumonia detection accuracy  
- Sensitivity to subtle lung opacities  
- Confidence in predictions  

### **2. Explainable & Trustworthy**
With **Grad-CAM heatmaps**, clinicians can visually see *where* the model focuses:  
- Highlighting infected lung regions  
- Ensuring the AI isn't looking at irrelevant areas  
- Providing transparency for medical decision-making  

### **3. Supports Radiologists**
Instead of replacing human experts, PneuNet provides:
- Instant preliminary screening  
- Assistance during high workload  
- A second pair of (AI) eyes for safety  

### **4. Handles Imbalanced Medical Data**
Medical datasets often contain more “normal” patients than diseased ones.  
PneuNet tackles this using:
- Data augmentation  
- Oversampling  
- Class-aware loss functions  

This improves sensitivity toward pneumonia cases.

---

## **📊 Comparison Matrices**

The ensemble produces more stable predictions compared to individual models.  
A typical (representational) comparison looks like this:

| Model          | Accuracy | Precision | Recall | F1-Score | AUC |
|----------------|----------|-----------|--------|----------|-----|
| DenseNet       | 0.886     | 0.875      | 0.954 | 0.913     | 0.959 |
| EfficientNet   | 0.865     | 0.859    | 0.938   | 0.897     | 0.939 |
| CNN + RNN      | 0.830   | 0.841    | 0.897 | 0.868   | 0.897 |
| **Ensemble**   | **0.9054** | **0.9107** | **0.9410** | **0.9256** | **0.9603** |

This demonstrates why combining multiple architectures yields the strongest performance.

---

## **🧪 Confusion Matrices (How the Model Behaves)**

The confusion matrix provides insight into:
- True pneumonia cases correctly identified  
- Normal cases correctly identified  
- Misclassifications  

**Representation:**

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/Neon01602/PneuNet-Pred-Model/blob/main/DenseNet/Figure_3.png?raw=true" alt="DenseNet Confusion Matrix" width="250"/><br>
      <b>Figure 1 — DenseNet Confusion Matrix</b>
    </td>
    <td align="center">
      <img src="https://github.com/Neon01602/PneuNet-Pred-Model/blob/main/EfficientNet/cm_chest_xray.png?raw=true" alt="EfficientNet Confusion Matrix" width="250"/><br>
      <b>Figure 2 — EfficientNet Confusion Matrix</b>
    </td>
    <td align="center">
      <img src="https://github.com/Neon01602/PneuNet-Pred-Model/blob/main/cnn%2Brnn/Figure_3.png?raw=true" alt="CNN+RNN Confusion Matrix" width="250"/><br>
      <b>Figure 3 — CNN+RNN Confusion Matrix</b>
    </td>
  </tr>
</table>


