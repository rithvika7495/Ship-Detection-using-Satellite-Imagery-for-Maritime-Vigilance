# 🚢 Ship Detection using Synthetic Aperture Radar (SAR) Imagery

This repository contains the full implementation and research code for the project titled **"Ship Detection using SAR Images for Maritime Vigilance"**. The aim is to develop a deep learning-based framework optimized for accurate ship detection in **Synthetic Aperture Radar (SAR)** images — addressing key challenges like small targets, cluttered backgrounds, and high noise levels.

This project was presented at the **International Conference on Data Science, Agents, and Artificial Intelligence (ICDSAAI 2025)** and was conducted under the guidance of **Dr. Poonkodi M** from Vellore Institute of Technology, Chennai.

---

## 📚 Publication

- 📝 **Paper Title**: *Ship Detection using SAR Images for Maritime Vigilance*  
- 🧠 **Authors**: Rithvika T, Monish P, Dr. Poonkodi M  
- 🎓 **Guided & Contributed by**: *Dr. Poonkodi M*  
- 🏫 **Affiliation**: Vellore Institute of Technology, Chennai  
- 📍 **Conference**: International Conference on Data Science, Agents, and Artificial Intelligence (ICDSAAI 2025)  
- 📅 **Date**: March 29, 2025  
- 🔗 **DOI**: [10.1109/ICDSAAI65575.2025.11011861](https://doi.org/10.1109/ICDSAAI65575.2025.11011861)

---

## 🔬 Abstract

Maritime surveillance is crucial for national security, illegal shipping detection, and disaster response. Traditional detection techniques such as CFAR or threshold-based methods result in high false positives, especially in noisy SAR images. Deep learning models like Faster R-CNN and YOLO perform well on natural images but struggle with SAR's grayscale format and environmental noise.

To solve this, we propose a **custom-optimized ResNet50 framework** integrated with:
- Region of Interest (ROI) and corner detection  
- Feature extraction using ResNet50 + Feature Pyramid Networks (FPN)  
- Weight mapping and entropy-based filtering  
- Swish & Tanh activation functions  
- Hyperparameter tuning using Particle Swarm Optimization (PSO)

---

## 🧠 Model Architecture

> **Architecture Overview:**

![image](https://github.com/user-attachments/assets/32dcca1a-0fea-4d54-b4d6-160d18737897)

*The architecture utilizes ResNet50 and FPN for multi-scale feature extraction, enhanced by weight mapping and corner-aware ROIs. Activations are refined using Swish + Tanh, and the final layer detects and classifies ships in SAR scenes.*

---

## 🧪 Results and Discussion

> **Performance Visuals:**

![image](https://github.com/user-attachments/assets/5062078a-59c6-4b1e-9655-74ac6b13d984)

- **Detection Rate**: High accuracy for small and partially occluded ships  
- **Real-time Capability**: Moderate FPS for real-time monitoring tasks  
- **Comparative Advantage**: Outperforms traditional and basic deep learning models (Faster R-CNN, Mask R-CNN, YOLOv2, FCOS) in both accuracy and SAR-noise robustness

---

## ⚙️ Methodology

1. **Data Preprocessing**
   - Augmentation: Rotation, flipping, zooming  
   - Noise reduction: Median filtering  
   - Edge enhancement: Sobel filters

2. **Region of Interest & Corner Detection**
   - Highlights ship-likely regions  
   - Boosts shape awareness and boundary clarity

3. **Feature Extraction**
   - ResNet50 + FPN (U-Net-inspired) for multi-scale feature fusion  
   - Robust against cluttered backgrounds

4. **Weight Map & Activation Optimization**
   - Swish & Tanh activations for better gradient flow  
   - Weight maps emphasize ship regions, suppress noise

5. **Pixel Filtering & Invalid Region Removal**
   - Filters based on Euclidean distance and color difference  
   - Removes unlikely ship regions to reduce false positives

6. **Hyperparameter Optimization**
   - **Particle Swarm Optimization (PSO)** dynamically tunes the model for optimal balance between speed and accuracy

7. **Final Detection & Classification**
   - Detection: Ships located in refined ROIs  
   - Classification: Ships categorized by structure and size

---

## 🗃️ Dataset

- **Source**: Ship Detection Dataset 
- **Image Type**: SAR (Synthetic Aperture Radar)  
- **Labels**: Run-Length Encoded (RLE) binary segmentation masks  
- **Resolution**: 768 × 768 pixels per image

---

## 🛠️ Technologies Used

- 🧠 Deep Learning Frameworks: TensorFlow, Keras  
- 🖼️ Image Processing: OpenCV, scikit-image  
- 🧪 Model Backbone: ResNet50, Feature Pyramid Network (FPN)  
- ⚙️ Optimization: Particle Swarm Optimization (PSO)  
- 📊 Visualization: Matplotlib, Seaborn

---

## 🚀 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ship-detection-sar.git
cd ship-detection-sar
````

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Update dataset paths in the notebook**

```python
train_image_dir = "path/to/train_v2"
test_image_dir = "path/to/test_v2"
csv_path = "train_ship_segmentations_v2.csv"
```

4. **Run the notebook**

```bash
jupyter notebook "ship_detection using satellite imagery.ipynb"
```

---
## 🙏 Acknowledgments

We would like to express our sincere gratitude to **Dr. Poonkodi M** for her unwavering guidance and mentorship throughout the project. We also thank **VIT, Chennai** for supporting and providing a platform for research.

---

## 🧾 License

This project is intended for **academic and research use only**. For commercial use, please contact the authors.

---

## 🔗 Citation

```bibtex
R. Tiruveedhula and M. P, "Ship Detection using SAR Images for Maritime Vigilance," 2025 International Conference on Data Science, Agents & Artificial Intelligence (ICDSAAI), Chennai, India, 2025, pp. 1-6, doi: 10.1109/ICDSAAI65575.2025.11011861. keywords: {Surveillance;Image edge detection;Feature extraction;Real-time systems;Radar polarimetry;Security;Marine vehicles;Particle swarm optimization;Synthetic aperture radar;Synthetic aperture radar (SAR);Edge Detection;Feature Extraction Algorithm;Euclidian Distance},
```

