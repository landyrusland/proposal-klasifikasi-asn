# Aplikasi Pendukung Seminar Proposal Tesis

**Model Klasifikasi Pegawai ASN Menggunakan Machine Learning untuk Identifikasi Kesenjangan Kompetensi dan Penentuan Prioritas Pengembangan Berbasis Matriks Kompetensi**

Landi Ruslandi — NPM 072925025 — Magister Ilmu Komputer, Universitas Pakuan

---

## Isi Folder

| Berkas | Keterangan |
|---|---|
| `app.py` | Aplikasi Streamlit |
| `dataset_publik.csv` | Dataset anonim, 1.796 baris × 46 kolom |
| `requirements.txt` | Daftar pustaka yang dibutuhkan |

Ketiga berkas harus berada dalam satu folder yang sama.

---

## Cara Menjalankan

**1. Pasang Python 3.10 atau lebih baru**, lalu buka terminal (Command Prompt di Windows) dan masuk ke folder aplikasi:

```
cd path/menuju/folder-aplikasi
```

**2. Buat lingkungan virtual** (disarankan, agar tidak mengganggu instalasi Python lain):

```
python -m venv venv
```

Aktifkan — Windows:
```
venv\Scripts\activate
```
macOS atau Linux:
```
source venv/bin/activate
```

**3. Pasang pustaka:**

```
pip install -r requirements.txt
```

**4. Jalankan aplikasi:**

```
streamlit run app.py
```

Peramban akan terbuka otomatis pada `http://localhost:8501`. Bila tidak, salin alamat tersebut ke peramban.

**Menghentikan aplikasi:** tekan `Ctrl + C` pada terminal.

---

## Isi Aplikasi

**Tab 1 — Profil Dataset.** Asal data, hasil integrasi dua generasi sistem, kelengkapan indikator per tahun, dan pratinjau dataset. Kolom nama dan NIP disembunyikan.

**Tab 2 — EDA Dasar.** Profil pegawai, sebaran variabel terikat, perbandingan indikator menurut predikat kinerja, korelasi antarvariabel, dan kesenjangan kompetensi hasil asesmen.

**Tab 3 — Uji Kelayakan Model.** Tiga model dengan pengaturan bawaan sebagai pemeriksaan awal apakah data layak dimodelkan. **Bukan hasil penelitian.**

---

## Catatan Penting

Bagian pemodelan sengaja dibatasi pada tiga model tanpa penalaan hiperparameter. Tujuannya menunjukkan bahwa data **layak dimodelkan**, bukan menyajikan hasil akhir. Perbandingan sepuluh model, penalaan, uji signifikansi, analisis SHAP, dan prioritisasi pengembangan dilaksanakan pada tahap penelitian.

Dua kotak centang pada Tab 3 sebaiknya dipahami sebelum seminar:

- **Pemisahan berbasis pegawai.** Sebanyak 103 pegawai muncul pada tahun 2024 sekaligus 2025. Tanpa pengelompokan, seorang pegawai dapat berada di data latih dan data uji sekaligus sehingga kinerja model tampak lebih baik daripada sebenarnya. Nonaktifkan sesekali untuk melihat perbedaannya.
- **Variabel kelompok pegawai.** Dinonaktifkan secara bawaan karena kelompok Fungsional mendominasi predikat Sangat Baik (36 dari 47 kasus pada 2025), sehingga berisiko membuat model sekadar membedakan jenis jabatan.
