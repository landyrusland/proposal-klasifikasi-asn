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
| `kualifikasi.py` | Modul deteksi kesesuaian kualifikasi pendidikan dengan jabatan dan unit kerja |
| `dataset_penelitian_publik_v2.csv` | Dataset anonim bawaan, 2.898 baris × 55 kolom (versi 2: frekuensi keterlambatan, syarat jabatan, unit 2026) |
| `dataset_penelitian_publik.csv` | Dataset anonim versi 1 (cadangan; dipakai bila versi 2 tidak ada) |
| `syarat_jabatan.csv` | Jenjang pendidikan minimal 24 jabatan, diturunkan dari tabel `tugas_pegawai` institusi |
| `acuan_gelar.csv` | Acuan gelar akademik → bidang ilmu (dapat diubah) |
| `acuan_rumpun_unit.csv` | Acuan unit kerja → bidang ilmu yang relevan (dapat diubah) |
| `peta_unit_2026.csv` | Padanan nama unit lama pada dataset dengan struktur unit 2026 (tabel `unit`) |
| `matriks_kompetensi_terisi.csv` | Matriks kompetensi 207 pegawai (anonim), dasar perhitungan ulang K-formal dan K1 |
| `peringkat_prioritas_topsis.csv` | Peringkat prioritas 115 pegawai beserta nilai kriteria |
| `hasil_model.json` | Metrik skenario A, B, C dan uji kebermaknaan (hasil penelitian) |
| `hasil_skenarioD.json` | Metrik skenario D, kepentingan fitur, odds ratio, LIME (hasil penelitian) |
| `hasil_interpretasi.json` | Kepentingan fitur dan matriks konfusi skenario A (hasil penelitian) |
| `hasil_prioritas.json` | Matriks kompetensi, bobot AHP, hasil TOPSIS (hasil penelitian) |
| `requirements.txt` | Daftar pustaka yang dibutuhkan |

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

## Alur pemakaian

1. **Unggah satu berkas dataset utuh (100%)** pada panel kiri. Bila kosong, dipakai `dataset_penelitian_publik_v2.csv`. Bila berkas memuat kolom `nama` (misalnya `dataset_penelitian_triwulan.csv` milik Anda), pegawai dapat dicari berdasarkan nama.
2. **Tab “Persiapan Data”:** periksa mutu data dan pembersihan otomatis.
3. **Tab “Latih & Uji Model”:**
   - Langkah 1 menampilkan pembagian otomatis 80% data latih dan 20% data uji.
   - Langkah 2: pilih skenario, lalu tekan **Latih dan validasi model**.
4. **Tab “Interpretasi (LIME)”:** ketik nama atau kode pegawai.
5. **Tab “Kesesuaian Kualifikasi”:** periksa kesesuaian jenjang dan bidang pendidikan tiap pegawai dengan jabatan dan unit kerjanya.

### Tab Kesesuaian Kualifikasi

- **Jenjang:** pendidikan pegawai dibanding jenjang minimal jabatan pada kolom `pendidikan_formal` tabel `tugas_pegawai`. Jenjang minimal adalah jenjang terendah yang disebut (misalnya “SLTA – D-1 – D-3” berarti SLTA). Status: *di bawah*, *sesuai*, atau *di atas syarat*.
- **Bidang ilmu terhadap unit kerja:** bidang ilmu diperkirakan dari gelar akademik pada nama (S.Pi. → perikanan, S.Kom. → komputer, dan seterusnya) atau dari kolom `bidang_pendidikan`. Bidang itu lalu dicocokkan dengan rumpun bidang yang relevan bagi unit kerja. Hasil ini **indikatif** dan perlu dicocokkan dengan ijazah. Pegawai berpendidikan SLTA tidak dinilai bidangnya.
- Grafik per unit kerja, saringan unit dan status, pencarian nama, serta unduhan CSV.
- Tabel acuan dapat diperiksa di bagian *Acuan yang dipakai*. Bila analisis jabatan diperbarui, unggah ekspor terbaru tabel `tugas_pegawai` di bagian itu.

### Acuan syarat pendidikan pada matriks kompetensi

Tab **Matriks Kompetensi** memiliki pilihan acuan:

- **Tabel tugas jabatan institusi** (bawaan). Nilai pendidikan formal, total matriks, dan K1 pada AHP-TOPSIS dihitung ulang memakai jenjang minimal resmi.
- **Aturan laporan penelitian.** Angka sama persis dengan Bab IV laporan.

### Frekuensi keterlambatan dalam LIME

- Kolom `terlambat_frekuensi` berisi jumlah hari terlambat per triwulan. Hari dihitung terlambat bila jam datang melewati 07.30 (08.00 pada Ramadan 2026) tanpa keterangan izin. Kolom ini tersedia sejak Februari 2025.
- Pada tab Latih & Uji Model, kotak **Sertakan frekuensi keterlambatan** aktif secara bawaan dan menambahkan fitur ini ke skenario yang dipilih (misalnya D + T).
- Pada tab LIME, faktor ini **selalu ditampilkan** (tanda ⏰), walaupun pengaruhnya di luar faktor teratas, lengkap dengan peringkat pengaruhnya.
- Panel khusus membandingkan keterlambatan pegawai dengan sebaran keterlambatan pegawai Sangat Baik dan bukan Sangat Baik.

### Pembagian 80:20 di dalam aplikasi

Aplikasi membagi dataset **per pegawai**: semua triwulan seorang pegawai berada di satu sisi, sehingga model diuji pada pegawai yang belum pernah dilihatnya. Pembagian juga **berstrata** menurut kelompok pegawai dan banyaknya predikat Sangat Baik.

Pada dataset penelitian, hasilnya 165 pegawai latih (2.310 baris, Sangat Baik 26,9%) dan 42 pegawai uji (588 baris, Sangat Baik 28,0%). Bibit acak bawaannya 42, jadi pembagian selalu sama.

Porsi uji, bibit acak, dan unduhan kedua bagian ada di Langkah 1, pada bagian *Pengaturan dan unduhan pembagian data*.

Evaluasi pada data uji hanya memakai **label triwulanan asli**. Label tahunan 2024–2025 yang disebar ke empat triwulan ditampilkan dengan tanda “(tahunan)” dan tidak dihitung.

### Tab Persiapan Data

1. **Pemeriksaan mutu dan pembersihan otomatis:** membuang baris ganda pegawai–triwulan, mengosongkan nilai di luar batas wajar (misalnya durasi kerja lebih dari 16 jam), dan mengosongkan label yang tidak sah. Sel kosong tidak diisi nol.
2. **Kelengkapan indikator:** persentase sel kosong per indikator.
3. **Transformasi untuk model:** imputasi median, one-hot, standardisasi, dan SMOTE, dengan contoh satu baris sebelum dan sesudah transformasi, serta jumlah kelas sebelum dan sesudah SMOTE.
4. **Ringkasan pembersihan saat penyusunan dataset** dari data mentah.

## Tampilan hasil yang mudah dibaca

Tab **Latih & Uji Model**:

- **Alur empat langkah:** pembagian data, latih dan validasi, uji, lalu jelaskan. Langkah yang selesai bertanda ✓.
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
| Keterlambatan (opsional, dianjurkan) | `terlambat_frekuensi`; tampilan tambahan `terlambat_persen`, `terlambat_rata_menit` |
| Kesesuaian kualifikasi | `jabatan`, `pendidikan`, `unit_kerja` (opsional `unit_kerja_2026`, `bidang_pendidikan`) |
| Tampilan (opsional) | `nama`, `jabatan` |

Sel kosong dibiarkan kosong; jangan diisi nol. Indikator yang kosong diisi median data latih pada tahap pemodelan, dan dashboard memberi tanda bila hal itu terjadi.

---

## Privasi data

- Berkas yang diunggah hanya diproses di memori selama sesi berjalan dan tidak disimpan oleh aplikasi.
- Nama pegawai hanya tampil bila dataset yang Anda unggah memuat kolom `nama`. Dataset bawaan tetap anonim.
- Jalankan aplikasi secara lokal bila memakai data beridentitas. Jangan mengunggah data beridentitas ke layanan hosting publik.
- Keluaran dashboard bersifat rekomendatif dan tidak menggantikan kewenangan pejabat penilai.
