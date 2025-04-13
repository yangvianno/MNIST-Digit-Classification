# 🔢 MNIST Digit Classification with Deep Learning

This project applies deep learning to the **MNIST handwritten digit dataset**, using the **TensorFlow/Keras** framework to build, train, and evaluate neural networks. The goal is to recognize digits (0–9) from image data and optimize model accuracy using best practices in neural network design.

---

## 🧠 Model Architecture & Techniques

- 📦 **Framework**: TensorFlow & Keras
- 🧱 **Model Types**: Feedforward Neural Networks (FFNNs)
- 🧹 **Data Preprocessing**:
  - Normalization of pixel values
  - One-hot encoding of labels
- 🔐 **Regularization**:
  - Dropout layers to prevent overfitting
- 🔍 **Model Evaluation**:
  - Train/validation/test split (50,000/10,000/10,000)
  - Confusion matrix visualization
  - Error analysis on misclassified digits

---

## 📈 Results

- Achieved up to **98.2% test accuracy**
- Implemented and compared **multiple deep learning architectures**
- Evaluated performance with confusion matrices and visual inspection

---

## 📁 Project Structure

```
MNIST-Digit-Classification/
├── mnist_model.py         # Training and evaluation script
├── requirements.txt       # Python dependencies (tensorflow, numpy, matplotlib)
├── data/                  # MNIST dataset (can be loaded via Keras)
├── results/               # Saved confusion matrix, sample misclassifications
```

---

## 🚀 How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python mnist_model.py
```

3. Output:
- Training logs and final accuracy
- Plots of training/validation accuracy & loss
- Confusion matrix image

---

## 🔬 Future Improvements

- Add CNN (Convolutional Neural Network) model
- Use data augmentation for generalization
- Integrate TensorBoard for training visualization

---

## ✍️ Author

**Alex Vo**  
📧 [vodanghongphat@gmail.com](mailto:vodanghongphat@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/vodanghongphat)  
🐙 [GitHub](https://github.com/yangvianno)

---

## 📌 License

This project is for educational and demonstration purposes only, using the publicly available MNIST dataset from Yann LeCun et al.
