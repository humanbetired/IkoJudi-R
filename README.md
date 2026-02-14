
# 🧠 NoJudol

**NoJudol** adalah aplikasi berbasis web yang memungkinkan pengguna mengunggah gambar dan menjalankan analisis berbasis AI/ML untuk mendeteksi dan mengklasifikasikan konten gambar — tanpa *judgement*, tanpa ribet. Fokus utama kami adalah kemudahan, akurasi, dan pengalaman pengguna yang menyenangkan.

> ⚡ *Upload. Analyze. Understand. Simple.*

---

## 🚀 Fitur Unggulan

- 📤 **Drag and Drop Upload** — Upload gambar semudah menyeret dan melepas
- 🖼️ **Real-time Image Feedback** — Notifikasi langsung setelah gambar berhasil diunggah
- 🤖 **AI-powered Analysis** — Analisis otomatis menggunakan model Machine Learning
- 📊 **Hasil Visualisasi** — Tampilkan hasil analisis dalam bentuk yang menarik dan informatif
- 🌐 **Antarmuka Responsif** — Desain modern dan responsif untuk berbagai perangkat

---

## 🛠️ Teknologi yang Digunakan

- **Frontend**: HTML, Tailwind CSS, JavaScript
- **Backend**: Python, Flask
- **AI/ML**: TensorFlow / PyTorch (sesuai model yang dipakai)
- **Deployment**: (opsional) Vercel / Render / Heroku

---

## ⚙️ Cara Menjalankan

1. **Clone repositori ini**
   ```bash
   git clone https://github.com/username/NoJudol.git
   cd NoJudol
   ```

2. **Aktifkan virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan aplikasi**
   ```bash
   flask run
   ```

5. Buka browser ke: [http://localhost:5000](http://localhost:5000)

---

## 📁 Struktur Proyek

```
NoJudol/
│
├── static/               # File statis (JS, CSS, gambar)
│   └── js/
│       └── upload.js
│
├── templates/            # File HTML (Jinja2)
│   └── index.html
│
├── model/                # Model ML (jika ada)
│   └── classifier.pkl
│
├── app.py                # Entry point Flask
├── requirements.txt      # Daftar dependensi
└── README.md             # Dokumentasi ini
```

---

## 🤝 Kontribusi

Kami sangat terbuka untuk kontribusi! Silakan fork repositori ini dan buat pull request. Kamu juga bisa:

- Buat [Issue](https://github.com/username/NoJudol/issues) untuk bug atau fitur baru
- Tambahkan dokumentasi atau contoh gambar untuk demo

---

## 📄 Lisensi

MIT License. Silakan digunakan, dimodifikasi, dan dibagikan untuk tujuan apapun.

---

## ❤️ Dukungan

Kalau kamu suka proyek ini, bantu dengan ⭐ di GitHub.  
Atau, follow akun pembuat untuk proyek menarik lainnya!

---

> Dibuat dengan 💻, ☕, dan semangat #NoJudgement oleh Tim NoJudol.