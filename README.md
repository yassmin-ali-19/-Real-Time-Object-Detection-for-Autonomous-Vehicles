# 🤖 Real-Time Object Detection for Autonomous Vehicles [cite: 289]

## 🌟 Project Motivation & Overview

Autonomous vehicles require robust and intelligent perception systems to navigate safely. This project is dedicated to building an **AI-powered, real-time object detection system** for self-driving cars[cite: 307].

The system is designed to detect and classify critical road elements—**pedestrians**, **vehicles**, **traffic signs**, and **obstacles**—across diverse driving conditions, including urban streets, highways, and low-light scenarios[cite: 308]. The work covers the full AI pipeline, from data collection and model development to real-time deployment and continuous monitoring using MLOps practices[cite: 310].

---

## ⚙️ Technical Stack & Key Achievements

### 1. Dataset & Preprocessing
* **Dataset:** The large-scale **BDD100K** dataset was used, selected for its high diversity in weather/lighting and rich annotations, which was a better fit than the initially considered KITTI dataset[cite: 315, 317, 319].
* **Preprocessing:** The dataset size was reduced from 70,000 images to approximately **45,000 images** by removing extremely rare classes to mitigate class imbalance and ensure balanced learning[cite: 327, 328].
* **Split:** The final dataset was split into 70% training, 20% validation, and 10% testing[cite: 357, 358, 359, 360].

### 2. Model Development & Performance

The **YOLO Large** model was selected after initial experiments showed the best trade-off between detection accuracy and inference speed[cite: 370, 372].

| Model Version | All Classes mAP@0.5 | Car mAP@0.5 |
| :--- | :---: | :---: |
| YOLO l (Large) | 0.058 | 0.002 |
| YOLO m (Medium) | 0.057 | 0.002 |
| YOLO s (Small) | 0.052 | 0.002 |

**Final Model Configuration:**
* **Architecture:** YOLOv11 (Transfer Learning from Medium weights, `yolollm.pt`)[cite: 376].
* **Best Training Run:** 150 epochs, AdamW optimizer, with augmentations (Aug)[cite: 382].
* **Final Selection:** This configuration was chosen for its highest mAP scores and fastest inference time, making it suitable for real-time scenarios[cite: 387, 388].

**Key Performance Metrics (Best Model):**
| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: |
| **Car** | 0.804 [cite: 382] | 0.741 [cite: 382] | 0.805 [cite: 382] | 0.510 [cite: 382] |
| **Traffic Sign** | 0.732 [cite: 382] | 0.635 [cite: 382] | 0.695 [cite: 382] | 0.378 [cite: 382] |
| **Person** | 0.741 [cite: 382] | 0.606 [cite: 382] | 0.679 [cite: 382] | 0.352 [cite: 382] |
| **Traffic Light** | 0.740 [cite: 382] | 0.572 [cite: 382] | 0.645 [cite: 382] | 0.259 [cite: 382] |

### 3. Deployment & MLOps

* **Interface (GUI):** A real-time web application was developed using **Streamlit**[cite: 526].
    * Supports input from a Laptop Camera, local file uploads (video/image), and Mobile Streaming (via DroidCam)[cite: 529, 531, 527].
* **MLOps:** **MLflow** was used for model management, versioning, and serving[cite: 535].
    * The model was logged, registered, and promoted to the **"Production"** stage[cite: 536, 539].
    * Inference is handled via a **REST API endpoint** automatically exposed by MLflow, which is integrated with the Streamlit GUI[cite: 541, 543].

---

## ⚠️ Challenges Faced

During development, several real-world challenges were encountered and partially mitigated:

* **Training Large-Scale Data:** Training the BDD100K dataset was challenging due to frequent runtime disconnections on Google Colab and inefficiency of local training without widespread GPU access for the team[cite: 551, 552, 554, 555].
* **Imbalanced Classes:** The inherent imbalance of classes in BDD100K negatively impacted performance for minority classes, leading to the decision to exclude some extremely rare classes to improve training stability[cite: 557, 558].
* **Real-Time Inference:** When processing high-speed video streams in the GUI, the model occasionally could not keep up with all frames, causing delays and skipped predictions[cite: 561, 562].
* **Mobile Streaming Instability:** The mobile camera streaming solution (DroidCam) proved to be unstable and unreliable for consistent real-time inference[cite: 559, 560].

---

## 📈 Future Work

Future efforts will focus on enhancing the model's robustness and the system's deployment stability:

1.  **Improving the Dataset:** Instead of removing underrepresented classes, **augment** them with additional images to mitigate imbalance and improve the model's ability to generalize to all categories[cite: 565, 566].
2.  **Continue Model Tuning:** Further hyperparameter tuning (e.g., learning rates, batch sizes, optimizers, and network architectures) is planned to achieve more robust accuracy[cite: 568].
3.  **Improving Real-Time GUI:** Optimize the Streamlit GUI to handle high-speed video streams more efficiently, and implement a more reliable mobile streaming solution (e.g., using buffering or dedicated protocols) to ensure consistent frame-by-frame inference[cite: 570, 571].

---

## 👥 Team Information

| Role | Name |
| :--- | :--- |
| **Team Members** | Yassmin Walaa, Bassant Mohamed Reda , Raneem Khaled Abdelmonem , Reem Mohamed Ali , Renad |
| **Supervisor** | Eng: Aya Hisham |