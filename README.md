# 🌿 Medicinal Plant Classification with CNNs — Paper Replication

This repository is a full PyTorch-based replication of the research article:

> **"Enhanced classification of medicinal plants using deep learning and optimized CNN architectures"**  
> _Hicham Bouakkaz et al., Heliyon 11 (2025) e42385_  
> [Read the Paper](https://doi.org/10.1016/j.heliyon.2025.e42385)

---

## 📌 Project Goals

The purpose of this project is to reproduce the deep learning framework proposed in the paper, which aims to:
- Classify medicinal plant images using hybrid CNN architectures (residual & inverted residual blocks)
- Apply serial feature fusion and feature optimization via Binary Chimp Optimization (BCO)
- Visualize predictions with Grad-CAM

---

## 🔧 Features to Implement

- [ ] Residual Block CNN  
- [ ] Inverted Residual Block CNN  
- [ ] Serial Feature Fusion  
- [ ] Binary Chimp Optimization (BCO)  
- [ ] Grad-CAM Visualization  
- [ ] Dataset augmentation pipeline  
- [ ] Training pipeline with k-fold cross-validation  
- [ ] Evaluation metrics: Accuracy, F1-Score, Confusion Matrix  

---

## 📁 Dataset

The model uses the **Medicinal Plant - 30 classes Dataset** dataset:
📦 [Kaggle Dataset](https://www.kaggle.com/datasets/sharvan123/medicinal-plant)

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/kitnew/medplant-deepclassifier.git
cd medplant-deepclassifier

# Install dependencies
pip install -r requirements.txt

# Check config in `config/default.yaml`

# Download the dataset
# 1. Download the dataset from Kaggle
# 2. Place it in the `data/raw` directory
# 3. Run files preparation.
python -m src.preprocessing.file_prep

# Run the training pipeline
python -m src.training.train

# Run the evaluating pipeline
python -m src.evaluating.evaluate

# Enjoy visualizations!
```

---

## 📊 Results & Benchmarks

You can find results in `data/visualizations` directory.

---

🧠 Reference
- Bouakkaz, H., Bouakkaz, M., Kerrache, C. A., & Dhelim, S. (2025). Enhanced classification of medicinal plants using deep learning and optimized CNN architectures. Heliyon. 11(3), e42385. [PDF](https://www.sciencedirect.com/science/article/pii/S2405844025007650/pdfft?md5=90820734ff4baab581e7782b49d6cbe4&pid=1-s2.0-S2405844025007650-main.pdf)

---

## 🧑‍💻 Contributors

- [Nikita Chernysh](https://github.com/kitnew)
- Oleksandr Holovatyi

---

## 📜 License

MIT License