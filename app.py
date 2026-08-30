"""
Aplikasi pendukung Seminar Proposal Tesis
Model Klasifikasi Pegawai ASN Menggunakan Machine Learning
Landi Ruslandi — 072925025 — Magister Ilmu Komputer, Universitas Pakuan

Menjalankan:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Dataset & Uji Kelayakan Model — Proposal Tesis",
                   page_icon="📊", layout="wide")

BERKAS = "dataset_publik.csv"

KOL_ASESMEN = ['asesmen_integritas', 'asesmen_kerjasama', 'asesmen_komunikasi',
               'asesmen_orientasi_hasil', 'asesmen_pelayanan_publik',
               'asesmen_pengembangan_diri', 'asesmen_mengelola_perubahan',
               'asesmen_pengambilan_keputusan', 'asesmen_perekat_bangsa']

LABEL_RINGKAS = {c: c.replace('asesmen_', '').replace('_', ' ').title() for c in KOL_ASESMEN}


@st.cache_data
def muat_data():
    d = pd.read_csv(BERKAS, dtype={'kode_pegawai': str})
    return d.rename(columns={'kode_pegawai': 'nip'})   # kode semu sebagai kunci pengelompokan


try:
    ds = muat_data()
except FileNotFoundError:
    st.error(f"Berkas **{BERKAS}** tidak ditemukan. "
             "Letakkan berkas dataset pada folder yang sama dengan app.py.")
    st.stop()

aktif = ds[ds.status_aktif == 'Aktif'].copy()
berlabel = aktif[aktif.skp_predikat.notna()].copy()

# ─────────────────────────── SIDEBAR ───────────────────────────
with st.sidebar:
    st.markdown("### Proposal Tesis")
    st.caption("Model Klasifikasi Pegawai ASN Menggunakan Machine Learning "
               "untuk Identifikasi Kesenjangan Kompetensi dan Penentuan "
               "Prioritas Pengembangan Berbasis Matriks Kompetensi")
    st.markdown("---")
    st.markdown("**Landi Ruslandi**  \nNPM 072925025  \nMagister Ilmu Komputer  \n"
                "Universitas Pakuan")
    st.markdown("---")
    st.metric("Baris dataset", f"{len(ds):,}")
    st.metric("Pegawai aktif", aktif.nip.nunique())
    st.metric("Baris berlabel SKP", len(berlabel))

st.title("Dataset Penelitian dan Uji Kelayakan Model")
st.caption("Aplikasi pendukung seminar proposal — menampilkan profil data yang akan "
           "digunakan serta pemeriksaan awal kelayakan pemodelan.")

tab1, tab2, tab3 = st.tabs(["📊 Profil Dataset", "🔍 EDA Dasar", "🤖 Uji Kelayakan Model"])

# ═══════════════════════ TAB 1 — PROFIL DATASET ═══════════════════════
with tab1:
    st.subheader("Asal dan Hasil Integrasi Data")
    st.markdown(
        "Dataset dibentuk dari **dua generasi sistem informasi kepegawaian** yang "
        "memiliki kunci penghubung berbeda, ditambah hasil asesmen kompetensi dan "
        "dokumen Monitoring Evaluasi Akhir SKP. Seluruh sumber dipetakan ke **NIP** "
        "sebagai kunci baku."
    )

    sumber = pd.DataFrame([
        ["absensi", "baru + lama", "80.578", "26.118 baris ganda dibuang"],
        ["apel", "baru + lama", "50.271", "16.898 ganda; 8.595 hanya ada di sistem lama"],
        ["riwayat_penugasan", "lama", "5.848", "satu-satunya sumber bertanggal kegiatan"],
        ["riwayat_penugasan", "baru", "930", "penempatan unit/jabatan, tanpa tahun"],
        ["riwayat_pelatihan", "baru", "1.481", "39 baris tergeser; 1.447 tahun dipulihkan"],
        ["rekap asesmen", "—", "2 lembar", "9 indikator kompetensi, skala 1–3"],
        ["Monev SKP 2024 & 2025", "—", "4 berkas", "predikat Baik / Sangat Baik"],
    ], columns=["Tabel Sumber", "Sistem", "Baris Mentah", "Keterangan"])
    st.dataframe(sumber, use_container_width=True, hide_index=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baris akhir", f"{len(ds):,}")
    c2.metric("Kolom", ds.shape[1])
    c3.metric("Pegawai (seluruhnya)", ds.nip.nunique())
    c4.metric("Pegawai aktif", aktif.nip.nunique())

    st.markdown("---")
    st.subheader("Kelengkapan Indikator per Tahun")
    st.caption("Sel kosong berarti sumber data belum tersedia pada tahun tersebut — "
               "bukan bernilai nol. Pembedaan ini dipertahankan sejak tahap integrasi.")

    kelengkapan = []
    for th in sorted(aktif.tahun.unique()):
        s = aktif[aktif.tahun == th]
        kelengkapan.append({
            "Tahun": int(th),
            "Absensi": int((s.absensi_jumlah_hari.fillna(0) > 0).sum()),
            "Apel": int((s.apel_jumlah_undangan.fillna(0) > 0).sum()),
            "Penugasan": int((s.penugasan_jumlah.fillna(0) > 0).sum()),
            "Pelatihan": int((s.pelatihan_jumlah.fillna(0) > 0).sum()),
            "Predikat SKP": int(s.skp_predikat.notna().sum()),
        })
    kl = pd.DataFrame(kelengkapan)
    st.dataframe(kl, use_container_width=True, hide_index=True)

    fig = px.imshow(kl.set_index("Tahun").T, text_auto=True, aspect="auto",
                    color_continuous_scale="Blues",
                    labels=dict(x="Tahun", y="Indikator", color="Jumlah pegawai"))
    fig.update_layout(height=320, margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.info("**Konsekuensi bagi rancangan penelitian.** Hanya tahun 2025 yang memuat "
            "seluruh indikator sekaligus predikat SKP. Karena itu pemodelan dijalankan "
            "dalam tiga skenario yang saling melengkapi sebagai uji ketegaran, bukan "
            "memilih salah satu.")

    st.markdown("---")
    st.subheader("Pratinjau Dataset")
    kol_tampil = st.multiselect(
        "Pilih kolom yang ditampilkan",
        options=[c for c in ds.columns if c != 'nip'],
        default=['tahun', 'kelompok_pegawai', 'apel_persen_hadir', 'penugasan_jumlah',
                 'pelatihan_kumulatif_sd_tahun', 'asesmen_total', 'skp_predikat'])
    st.dataframe(aktif[kol_tampil].head(50), use_container_width=True, height=300)
    st.caption("Dataset telah dianonimkan: nama, NIP, tanggal lahir, dan jabatan "
               "dihapus, usia dikelompokkan. Sesuai UU No. 27 Tahun 2022 tentang "
               "Pelindungan Data Pribadi.")

# ═══════════════════════ TAB 2 — EDA DASAR ═══════════════════════
with tab2:
    st.subheader("Profil Pegawai Aktif")
    u = aktif.drop_duplicates('nip')

    c1, c2 = st.columns(2)
    with c1:
        vc = u.kelompok_pegawai.value_counts(dropna=False).reset_index()
        vc.columns = ['Kelompok', 'Jumlah']
        fig = px.bar(vc, x='Jumlah', y='Kelompok', orientation='h',
                     title="Kelompok Pegawai", text='Jumlah')
        fig.update_layout(height=280, margin=dict(t=40, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        vp = u.pendidikan.value_counts(dropna=False).head(7).reset_index()
        vp.columns = ['Pendidikan', 'Jumlah']
        fig = px.bar(vp, x='Jumlah', y='Pendidikan', orientation='h',
                     title="Tingkat Pendidikan", text='Jumlah')
        fig.update_layout(height=280, margin=dict(t=40, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Sebaran Variabel Terikat — Predikat Kinerja SKP")

    baris = []
    for th in [2024, 2025]:
        s = berlabel[berlabel.tahun == th]
        if len(s):
            pos = int(s.skp_sangat_baik.sum())
            baris.append({"Tahun": th, "Jumlah Pegawai": len(s), "Sangat Baik": pos,
                          "Baik": len(s) - pos,
                          "Proporsi Kelas Positif": f"{pos/len(s)*100:.1f}%"})
    sb = pd.DataFrame(baris)
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.dataframe(sb, use_container_width=True, hide_index=True)
        st.warning("Kelas tidak seimbang, terutama pada 2025. Penanganan SMOTE "
                   "diterapkan **hanya pada lipatan latih** di dalam validasi silang.")
    with c2:
        pv = berlabel.groupby(['tahun', 'skp_predikat']).size().reset_index(name='Jumlah')
        fig = px.bar(pv, x='tahun', y='Jumlah', color='skp_predikat', barmode='group',
                     title="Predikat Kinerja per Tahun",
                     labels={'tahun': 'Tahun', 'skp_predikat': 'Predikat'})
        fig.update_layout(height=300, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Perbandingan Indikator menurut Predikat Kinerja")
    th_pilih = st.radio("Tahun", [2025, 2024], horizontal=True)
    sp = berlabel[berlabel.tahun == th_pilih]

    var_num = ['penugasan_jumlah', 'pelatihan_kumulatif_sd_tahun', 'apel_persen_hadir',
               'absensi_jumlah_hari', 'asesmen_total']
    var_num = [v for v in var_num if sp[v].notna().any()]

    rr = sp.groupby('skp_predikat')[var_num].mean().T
    rr.columns = [str(c) for c in rr.columns]
    if 'Sangat Baik' in rr.columns and 'Baik' in rr.columns:
        rr['Selisih'] = (rr['Sangat Baik'] - rr['Baik'])
    st.dataframe(rr.round(2), use_container_width=True)

    pilih_var = st.selectbox("Tampilkan sebaran variabel", var_num, index=0)
    fig = px.box(sp, x='skp_predikat', y=pilih_var, color='skp_predikat', points='all',
                 labels={'skp_predikat': 'Predikat Kinerja'})
    fig.update_layout(height=380, margin=dict(t=30, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    if th_pilih == 2025:
        st.error("**Temuan yang perlu ditelaah.** Persentase kehadiran apel pegawai "
                 "berpredikat Sangat Baik justru **lebih rendah** sekitar 22 poin. "
                 "Kehadiran apel berkorelasi negatif dengan jumlah penugasan "
                 "(r = −0,306). Dugaan awal: pegawai yang banyak ditugaskan lebih "
                 "sering berada di luar kantor. Bila terkonfirmasi, kehadiran apel "
                 "tidak dapat ditafsirkan lugas sebagai indikator disiplin.")

    st.markdown("---")
    st.subheader("Korelasi Antarvariabel")
    kor_var = [v for v in ['apel_persen_hadir', 'penugasan_jumlah', 'pelatihan_jumlah',
                           'pelatihan_kumulatif_sd_tahun', 'absensi_jumlah_hari',
                           'asesmen_total'] if sp[v].notna().any()]
    korr = sp[kor_var].corr().round(3)
    fig = px.imshow(korr, text_auto=True, aspect="auto", zmin=-1, zmax=1,
                    color_continuous_scale="RdBu_r")
    fig.update_layout(height=420, margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Kesenjangan Kompetensi Hasil Asesmen")
    sa = u[u.asesmen_tersedia == 1]
    if len(sa):
        lemah = (sa[KOL_ASESMEN] == 1).sum().sort_values(ascending=False)
        dfl = pd.DataFrame({
            "Indikator": [LABEL_RINGKAS[c] for c in lemah.index],
            "Jumlah Pegawai": lemah.values,
            "Persentase": (lemah.values / len(sa) * 100).round(1)})
        c1, c2 = st.columns([1.2, 1])
        with c1:
            fig = px.bar(dfl, x="Jumlah Pegawai", y="Indikator", orientation='h',
                         text="Jumlah Pegawai",
                         title=f"Pegawai dengan Skor 1 (Kelemahan), n = {len(sa)}")
            fig.update_layout(height=380, margin=dict(t=40, b=10),
                              yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            rata = (sa[KOL_ASESMEN] == 1).sum(axis=1).mean()
            bersih = int(((sa[KOL_ASESMEN] == 1).sum(axis=1) == 0).sum())
            st.metric("Pegawai dengan data asesmen", len(sa))
            st.metric("Rata-rata kelemahan per pegawai", f"{rata:.1f} dari 9")
            st.metric("Pegawai tanpa kelemahan", bersih)
            st.info("Kelemahan tersebar hampir merata. Bila seluruhnya ditindaklanjuti, "
                    "nyaris semua pegawai menjadi sasaran intervensi sekaligus. "
                    "**Inilah yang menjadikan prioritisasi berbasis dampak sebagai "
                    "kebutuhan nyata**, bukan sekadar penyempurnaan metodologis.")

# ═══════════════════════ TAB 3 — UJI KELAYAKAN MODEL ═══════════════════════
with tab3:
    st.subheader("Uji Kelayakan Awal Pemodelan")
    st.markdown(
        "Bagian ini **bukan hasil penelitian**, melainkan pemeriksaan apakah data yang "
        "tersedia layak dimodelkan. Yang diuji hanya tiga model dengan pengaturan "
        "bawaan, tanpa penalaan hiperparameter. Perbandingan sepuluh model, penalaan, "
        "uji signifikansi, dan analisis SHAP dilaksanakan pada tahap penelitian."
    )

    c1, c2 = st.columns(2)
    with c1:
        skenario = st.selectbox(
            "Skenario data",
            ["A. 2024–2025 (tanpa absensi)", "B. 2025 saja (indikator lengkap)",
             "C. 2024 saja (tanpa absensi)"])
    with c2:
        pakai_grup = st.checkbox(
            "Pemisahan berbasis pegawai (StratifiedGroupKFold)", value=True,
            help="Mencegah pegawai yang sama muncul di data latih dan uji sekaligus. "
                 "Wajib aktif untuk skenario A.")

    ikut_kelompok = st.checkbox(
        "Sertakan variabel kelompok pegawai", value=False,
        help="Kelompok Fungsional mendominasi predikat Sangat Baik. Menyertakan "
             "variabel ini berisiko membuat model hanya membedakan jenis jabatan.")

    if st.button("Jalankan uji kelayakan", type="primary"):
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import (StratifiedGroupKFold,
                                             RepeatedStratifiedKFold, cross_validate)
        from sklearn.metrics import make_scorer, matthews_corrcoef
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE

        num_dasar = ['apel_jumlah_undangan', 'apel_persen_hadir', 'penugasan_jumlah',
                     'penugasan_kegiatan_unik', 'pelatihan_jumlah',
                     'pelatihan_kumulatif_sd_tahun'] + KOL_ASESMEN
        num_abs = ['absensi_jumlah_hari', 'absensi_rata_durasi_jam']

        if skenario.startswith("A"):
            data, num = berlabel, num_dasar
        elif skenario.startswith("B"):
            data, num = berlabel[berlabel.tahun == 2025], num_dasar + num_abs
        else:
            data, num = berlabel[berlabel.tahun == 2024], num_dasar

        kat = ['jenis_kelamin', 'pendidikan', 'kelompok_usia'] + (['kelompok_pegawai'] if ikut_kelompok else [])

        X, y, g = data[num + kat], data.skp_sangat_baik.astype(int), data.nip

        pra = ColumnTransformer([
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                              ('sc', StandardScaler())]), num),
            ('kat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                              ('oh', OneHotEncoder(handle_unknown='ignore',
                                                   min_frequency=5))]), kat)])

        model = {
            'Regresi Logistik (L2)': LogisticRegression(max_iter=2000),
            'Linear Discriminant Analysis': LinearDiscriminantAnalysis(
                solver='lsqr', shrinkage='auto'),
            'Random Forest': RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                    random_state=42, n_jobs=-1),
        }
        skor = {'f1': 'f1', 'roc_auc': 'roc_auc', 'recall': 'recall',
                'mcc': make_scorer(matthews_corrcoef)}
        cv = (StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) if pakai_grup
              else RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42))

        hasil = []
        bar = st.progress(0.0, text="Menjalankan validasi silang…")
        for i, (nm, mdl) in enumerate(model.items(), 1):
            pipe = ImbPipeline([('pra', pra),
                                ('smote', SMOTE(random_state=42, k_neighbors=3)),
                                ('clf', mdl)])
            r = cross_validate(pipe, X, y, cv=cv, scoring=skor, n_jobs=-1,
                               groups=g if pakai_grup else None)
            hasil.append({'Model': nm, 'Kelompok': 'Linier' if i <= 2 else 'Ensemble',
                          'F1': r['test_f1'].mean(), 'SD F1': r['test_f1'].std(),
                          'AUC-ROC': r['test_roc_auc'].mean(),
                          'Recall': r['test_recall'].mean(),
                          'MCC': r['test_mcc'].mean()})
            bar.progress(i / len(model), text=f"Selesai: {nm}")
        bar.empty()

        h = pd.DataFrame(hasil).sort_values('F1', ascending=False)
        st.markdown(f"**n = {len(y)} baris, {g.nunique()} pegawai, "
                    f"{int(y.sum())} berpredikat Sangat Baik ({y.mean()*100:.1f}%)**")
        st.dataframe(h.round(3), use_container_width=True, hide_index=True)

        fig = go.Figure()
        for kel, warna in [('Linier', '#4C78A8'), ('Ensemble', '#F58518')]:
            sub = h[h.Kelompok == kel]
            fig.add_bar(x=sub.Model, y=sub.F1, name=kel, marker_color=warna,
                        error_y=dict(type='data', array=sub['SD F1']))
        fig.update_layout(title="F1-Score menurut Kelompok Model (batang galat = simpangan baku)",
                          height=380, yaxis_title="F1-Score", margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

        teratas = h.iloc[0]
        st.success(f"**Kesimpulan sementara.** Model terbaik pada uji ini adalah "
                   f"**{teratas.Model}** ({teratas.Kelompok}) dengan F1 = "
                   f"{teratas.F1:.3f} dan AUC-ROC = {teratas['AUC-ROC']:.3f}. "
                   "Data layak dimodelkan. Peringkat ini masih dapat berubah setelah "
                   "penalaan hiperparameter dan pengujian signifikansi.")

        with st.expander("Catatan metodologis — penting untuk sesi tanya jawab"):
            st.markdown("""
- **Angka di atas bersifat indikatif.** Tanpa penalaan hiperparameter dan tanpa uji
  signifikansi, selisih antarmodel belum dapat dinyatakan nyata secara statistik.
- **Pemisahan berbasis pegawai wajib untuk skenario A.** Sebanyak 103 pegawai muncul
  pada tahun 2024 sekaligus 2025. Tanpa pengelompokan, seorang pegawai dapat berada di
  data latih dan data uji sekaligus sehingga kinerja model tampak lebih baik daripada
  sebenarnya. Nonaktifkan kotak centang tersebut untuk melihat perbedaannya.
- **Data asesmen merupakan potret satu waktu** yang diberlakukan pada seluruh tahun.
  Untuk baris tahun 2024, hal ini berarti informasi yang lebih baru dipakai menjelaskan
  kejadian yang lebih lama. Keterbatasan ini dinyatakan terbuka pada naskah proposal.
- **Variabel kelompok pegawai dinonaktifkan secara bawaan** karena kelompok Fungsional
  mendominasi predikat Sangat Baik (36 dari 47 kasus pada 2025). Menyertakannya
  berisiko membuat model sekadar membedakan jenis jabatan, bukan kinerja.
            """)
    else:
        st.info("Tekan tombol di atas untuk menjalankan uji kelayakan. "
                "Proses memerlukan beberapa detik.")
