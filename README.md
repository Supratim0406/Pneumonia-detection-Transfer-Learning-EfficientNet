
# 🫁 Pneumonia X-Ray Classification System

A full-stack **medical image classification project** that detects **Pneumonia from Chest X-ray images** using **EfficientNetB3 + TensorFlow + Flask + Bootstrap**. Built as a portfolio project to demonstrate **end‑to‑end machine learning engineering**, including model training, inference API, and web deployment.

---

## 🚀 Project Highlights
- **EfficientNetB3 model** fine‑tuned on Pneumonia dataset
- **Binary classification:** `PNEUMONIA` vs `NORMAL`
- **Flask backend** for real‑time prediction
- **Bootstrap frontend** to upload chest X‑ray images
- **CORS support** for deployment compatibility (Netlify + Render)
- **Model served locally or via HuggingFace Hub (optional)**
- Designed as a **portfolio project** for data science / ML roles

---

## 📌 Tech Stack
| Component | Technology |
|----------|-----------|
| Model | TensorFlow, EfficientNetB3 |
| Backend | Flask, Python |
| Frontend | HTML, CSS, Bootstrap |
| Deployment Options | Render (backend), Netlify (frontend) |
| Version Control | Git + GitHub |

---

## 🧠 Model Summary
- Backbone: **EfficientNetB3** pretrained on ImageNet
- Input Size: **300x300**
- Loss: **Binary Crossentropy**
- Activation: **Sigmoid**
- Metric: **ROC‑AUC** (important for medical tasks)

> ROC‑AUC used because it evaluates performance across thresholds and handles class imbalance — crucial in medical diagnostics.

---

## 📂 Folder Structure
```
Pneumonia Detection/
│
├─ backend/
│   ├─ app.py
│   ├─ models/
│   │   └─ efficientNet_model_tuned.h5
│   └─ uploads/
│
├─ templates/
│   └─ index.html
│
└─ README.md
```

---

## 🧪 Run Locally
### 1️⃣ Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Start backend
```bash
cd backend
python app.py
```
➡ Open: http://127.0.0.1:5000

---

## 🌍 Deployment (Optional)
| Service | Purpose | Status |
|---------|---------|-------|
| **Render** | Backend hosting | ✔ Recommended |
| **Netlify** | Frontend hosting | ✔ Recommended |
| **HuggingFace Hub** | Store model | Optional |

---

## 🎯 Skills Demonstrated
- Deep Learning & Transfer Learning
- Model serving & inference pipelines
- API development & CORS handling
- Version control & documentation
- Frontend‑backend integration

---

## 📸 Demo
> 

![Demo Screenshot](https://github.com/user-attachments/assets/bfd42d22-aa94-480d-9dbe-6bcec55fa01b)

---

## 🧩 Future Improvements
- Deploy model via **Render / Netlify** for faster inference
- Add user authentication for clinical use
- CI/CD with GitHub Actions

---

## 🤝 Contributions
Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

---

## 📜 License
MIT — free to use for learning and portfolio building.

---

## 💼 Contact
If you'd like to collaborate or have questions:
```
👤 Name: Supratim Saha
📧 Email: supratimsaha.ds@gmail.com
🔗 LinkedIn: 
🌐 Portfolio: 
```

> ⭐ *If you found this helpful, consider giving the repo a star — it helps showcase the work to recruiters!*
