#  Real-Time Object Detection for Autonomous Vehicles

Autonomous vehicles are transforming the way we think about transportation, but for them to navigate safely, they need to “see” and understand their surroundings.

This project is all about building a real-time object detection system that allows self-driving cars to detect pedestrians, vehicles, traffic signs, and obstacles,so they can make smart, safe, and instantaneous decisions on the road.

## Project Motivation & Vision

Imagine a world where cars can perceive their environment like humans do. Autonomous vehicles have the potential to reduce accidents, save lives, and make commuting more efficient.

Achieving this requires robust and intelligent perception systems that can detect objects in real time across diverse conditions.

This project tackles that challenge by building a real-time object detection system designed to be fast, accurate, and reliable. By combining state-of-the-art object detection models, transfer learning, and real-world driving datasets, this project aims to empower autonomous vehicles with intelligence and reliability, making roads safer and driving smarter.

## Dataset

We use the BDD100K dataset, a large-scale driving dataset with labeled objects:
🔗 BDD100K on Kaggle

## YOLOv11 pretrained model sizes Performance Comparison 

| Model Version | Overall mAP@0.5 | Performance on 'person' Class | Trade-off & Recommendation |
| :--- | :---: | :---: | :--- |
| **YOLO l (Large)** | **0.058 (Highest)** | **0.541 (Best)** | **Maximum Accuracy.** Best at detecting the dominant 'person' class, but is the largest and slowest model. |
| **YOLO m (Medium)** | 0.057 | 0.529 (Very Good) | **Best Balance.** Near-identical overall accuracy to l but is significantly faster and smaller. Recommended for deployment. |
| **YOLO s (Small)** | 0.052 (Lowest) | 0.467 (Lowest) | **Highest Speed.** Fastest and smallest model, but shows a clear drop in performance, even on the best-performing class. |
