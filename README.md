# Dashboard Visualisasi Hasil Penelitian

**Model Klasifikasi dan Interpretasi Kinerja Pegawai ASN Menggunakan Machine Learning dan AHP-TOPSIS untuk Penentuan Prioritas Pengembangan Kompetensi Berbasis Matriks Kompetensi**

Landi Ruslandi — NPM 072925025 — Magister Ilmu Komputer, Universitas Pakuan
Studi Kasus pada BBPPMPV Bidang Pertanian

---

## Isi folder

| Berkas | Keterangan |
|---|---|
| `app.py` | Aplikasi dashboard Streamlit |
| `pemodelan.py` | Modul pemodelan: penyiapan data, pelatihan, validasi, uji Nadeau–Bengio, LIME |
| `dataset_penelitian_publik.csv` | Dataset anonim bawaan, 2.898 baris × 45 kolom |
| `peringkat_prioritas_topsis.csv` | Peringkat prioritas 115 pegawai beserta nilai kriteria |
| `hasil_model.json` | Metrik skenario A, B, C dan uji kebermaknaan (hasil penelitian) |
| `hasil_skenarioD.json` | Metrik skenario D, kepentingan fitur, odds ratio, LIME (hasil penelitian) |
| `hasil_interpretasi.json` | Kepentingan fitur dan matriks konfusi skenario A (hasil penelitian) |
| `hasil_prioritas.json` | Matriks kompetensi, bobot AHP, hasil TOPSIS (hasil penelitian) |
| `requirements.txt` | Daftar pustaka yang dibutuhkan |
| `data_80_20/data_latih_80.csv` | 80% pegawai (165 pegawai, 2.310 baris) untuk melatih model |
| `data_80_20/data_uji_20.csv` | 20% pegawai (42 pegawai, 588 baris) untuk menguji model |

Seluruh berkas harus berada dalam satu folder yang sama dengan `app.py`.

---

## Cara menjalankan

**1. Pasang Python 3.10 atau lebih baru**, lalu buka terminal dan masuk ke folder aplikasi:

```
cd path/menuju/aplikasi_dashboard
```

**2. Buat lingkungan virtual** (disarankan):

```
python -m venv venv
```

Aktifkan — Windows: `venv\Scripts\activate` · macOS atau Linux: `source venv/bin/activate`

**3. Pasang pustaka** (versi baru menambahkan scikit-learn, imbalanced-learn, XGBoost, dan LIME):

```
pip install -r requirements.txt
```

**4. Jalankan:**

```
streamlit run app.py
```

Peramban terbuka otomatis pada `http://localhost:8501`. Hentikan dengan `Ctrl + C`.

---

## Alur pemakaian (skema 80:20)

1. **Unggah data latih:** `data_80_20/data_latih_80.csv` ke kotak *Data latih — 80%*.
2. **Unggah data uji:** `data_80_20/data_uji_20.csv` ke kotak *Data uji — 20%*.
3. **Opsional, agar bisa dicari dengan nama:** unggah `peta_identitas_v3.csv` (berkas lokal Anda, berkolom `kode_pegawai`, `nip`, `nama`) ke kotak *Peta identitas*.
4. **Tab “Latih & Uji Model”:** pilih skenario, lalu tekan **Latih dan validasi model**.
5. **Tab “Interpretasi (LIME)”:** ketik nama atau kode pegawai.

### Cara pembagian 80:20

Pembagian dilakukan **per pegawai**: semua triwulan seorang pegawai berada di satu sisi, sehingga model diuji pada pegawai yang benar-benar belum pernah dilihatnya. Pembagian juga **berstrata** menurut kelompok pegawai dan banyaknya predikat Sangat Baik, sehingga proporsinya seimbang (Sangat Baik 26,9% pada data latih dan 28,0% pada data uji). Bibit acaknya 42, jadi hasilnya selalu sama.

Pembagian baru untuk dataset lain dapat dibuat dari panel kiri, menu **Bagi dataset 80:20 per pegawai**.

Evaluasi pada data uji hanya memakai **label triwulanan asli**. Label tahunan 2024–2025 yang disebar ke empat triwulan ditampilkan dengan tanda “(tahunan)” dan tidak dihitung.

## Tampilan hasil yang mudah dibaca

Tab **Latih & Uji Model**:

- **Alur empat langkah:** data latih, data uji, latih, lalu uji dan jelaskan. Langkah yang selesai bertanda ✓.
- **Tampilan Sederhana (bawaan):**
  - ringkasan hasil dalam kalimat;
  - empat angka utama beserta artinya;
  - matriks konfusi dalam bahasa sehari-hari;
  - grafik F1 validasi dibanding F1 uji, untuk memeriksa apakah model hanya menghafal data latih.
- **Tampilan Rinci:** seluruh tabel statistik seperti di laporan, yaitu rerata ± simpangan baku, uji Nadeau–Bengio, matriks konfusi, dan odds ratio.
- **Kamus istilah:** penjelasan singkat setiap ukuran.
- **Prediksi per pegawai:** peluang dalam persen beserta status ✅ tepat atau ⚠️ keliru.

Tab **Interpretasi (LIME)**:

- **Kartu ringkasan:** peluang, tebakan model, predikat aktual, dan kesesuaian.
- **Narasi otomatis:** faktor penekan dan penaik, serta perubahan dibanding triwulan sebelumnya.
- **Grafik faktor:** setiap faktor ditampilkan dengan nilai pegawai dalam satuan asli (pukul 07.22, 3 kegiatan, 9 hari).
- **Rincian faktor:** arah pengaruh, kelompok nilai menurut LIME, posisi dibanding data latih, besar pengaruh, dan jenis faktor (pengembangan kompetensi, kedisiplinan dan presensi, atau atribut tetap).
- **Catatan dan tindak lanjut:** disusun berdasarkan temuan Bab IV, misalnya pelatihan sebagai faktor yang dapat diintervensi dan kewaspadaan terhadap mutu pencatatan presensi.
- **Tombol periksa kestabilan:** menghitung indeks Jaccard dari lima ulangan LIME.
- **Lintasan peluang per triwulan.**
- **Unduh penjelasan:** berkas CSV untuk arsip.

## Format CSV

Pemisah koma atau titik koma, dengan desimal titik atau koma (ekspor Excel berbahasa Indonesia), keduanya terbaca.

| Kelompok | Kolom |
|---|---|
| Kunci pegawai | `kode_pegawai` atau `nip` |
| Periode | `tw` (contoh `2026-TW3`) atau `tahun` dan `triwulan` |
| Label (wajib untuk data latih) | `skp_predikat` (`Sangat Baik`, `Baik`, …) atau `skp_sangat_baik` (1/0) |
| Penanda replikasi (opsional) | `skp_jenis_penilaian` (`Triwulan` / `Tahunan (replikasi)`) atau `skp_bobot` (1 / 0,25) |
| Fitur skenario D | `pelatihan_kumulatif`, `masa_kerja_tahun`, `apel_alpa`, `absensi_rata_jam_masuk`, `apel_persen_hadir`, `absensi_tanpa_checkout`, `absensi_rata_durasi`, `apel_dinas_luar` |
| Tambahan skenario A | `usia`, `pendidikan`, `kelompok_pegawai`, `status_pegawai` |
| Tambahan skenario C | sembilan kolom `asesmen_…` dan `asesmen_tersedia` |
| Tampilan (opsional) | `nama`, `jabatan` |

Sel kosong dibiarkan kosong; jangan diisi nol. Indikator yang kosong diisi median data latih pada tahap pemodelan, dan dashboard memberi tanda bila hal itu terjadi.

---

## Privasi data

- Berkas yang diunggah hanya diproses di memori selama sesi berjalan dan tidak disimpan oleh aplikasi.
- Nama pegawai hanya tampil bila berkas yang Anda unggah (data atau peta identitas) memuat kolom `nama`. Dataset bawaan tetap anonim.
- Jalankan aplikasi secara lokal bila memakai data beridentitas. Jangan mengunggah data beridentitas ke layanan hosting publik.
- Keluaran dashboard bersifat rekomendatif dan tidak menggantikan kewenangan pejabat penilai.
