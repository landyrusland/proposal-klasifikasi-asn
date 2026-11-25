"""
Dashboard Visualisasi Hasil Penelitian
Model Klasifikasi dan Interpretasi Kinerja Pegawai ASN Menggunakan Machine Learning
dan AHP-TOPSIS untuk Penentuan Prioritas Pengembangan Kompetensi

Landi Ruslandi - NPM 072925025 - Magister Ilmu Komputer, Universitas Pakuan

Menjalankan:  streamlit run app.py
"""
import io
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix

import pemodelan as pm

NAVY, EMAS, ABU = '#263C92', '#C08A00', '#8A8F9A'
HIJAU, MERAH, BIRUM = '#2E6E4E', '#A4303F', '#ECEFF8'
URUT = [NAVY, EMAS, ABU, HIJAU, MERAH]
BAWAAN = 'dataset_penelitian_publik.csv'
W = dict(width='stretch')

st.set_page_config(page_title='Dashboard Kinerja Pegawai ASN', page_icon='📊',
                   layout='wide', initial_sidebar_state='expanded')

st.markdown(f"""<style>
.block-container {{padding-top: 2.2rem; max-width: 1500px;}}
h1,h2,h3 {{color: {NAVY};}}
div[data-testid="stMetricValue"] {{color: {NAVY}; font-size: 1.7rem;}}
div[data-testid="stMetric"] {{background:{BIRUM}; border-left:4px solid {NAVY};
  padding:12px 14px; border-radius:6px;}}
.kecil {{color:#6b7280; font-size:0.85rem;}}
</style>""", unsafe_allow_html=True)


def rp(x, d=3):
    """Angka desimal gaya Indonesia."""
    return f'{x:.{d}f}'.replace('.', ',') if pd.notna(x) else '–'


# ----------------------------------------------------------------- pemuatan
@st.cache_data(show_spinner=False)
def baca_csv(isi: bytes):
    """Membaca CSV berpemisah koma atau titik koma, UTF-8 atau Latin-1."""
    for enc in ('utf-8-sig', 'latin-1'):
        try:
            return pd.read_csv(io.BytesIO(isi), sep=None, engine='python', encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError('Berkas tidak dapat dibaca sebagai CSV.')


@st.cache_data(show_spinner=False)
def muat_json(nama):
    return json.load(open(nama, encoding='utf-8')) if os.path.exists(nama) else None


def pasang_peta(d, peta):
    """Menambahkan kolom nama dari peta identitas (hanya di memori, tidak disimpan)."""
    if peta is None:
        return d
    p = peta.copy()
    p.columns = [c.strip().lower() for c in p.columns]
    kunci = 'kode_pegawai' if 'kode_pegawai' in p.columns else ('nip' if 'nip' in p.columns else None)
    if kunci is None or 'nama' not in p.columns:
        return d
    peta_nama = dict(zip(p[kunci].astype(str), p['nama'].astype(str).str.strip()))
    d = d.copy()
    nama_baru = d['kode'].map(peta_nama)
    if 'nip' in d.columns:
        nama_baru = nama_baru.fillna(d['nip'].astype(str).map(peta_nama))
    d['nama'] = nama_baru.fillna(d['nama'] if 'nama' in d.columns else '').fillna('')
    return d


def sidik(df):
    return int(pd.util.hash_pandas_object(df, index=False).sum() % (2 ** 61))


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown('### Dashboard Hasil Penelitian')
    st.caption('Landi Ruslandi · NPM 072925025 · Magister Ilmu Komputer UNPAK')
    st.divider()
    st.markdown('**Sumber data**')
    up_latih = st.file_uploader('Data latih — 80% (CSV)', type=['csv'], key='up_latih',
                                help='Data berlabel untuk melatih dan memvalidasi model. Bila kosong, dipakai '
                                     'dataset penelitian bawaan (seluruh pegawai).')
    up_olah = st.file_uploader('Data uji — 20% (CSV)', type=['csv'], key='up_olah',
                               help='Pegawai yang tidak ikut dilatih, untuk menguji model. Bila berlabel, kinerja '
                                    'model pada data ini ikut dihitung.')
    up_peta = st.file_uploader('Peta identitas — opsional', type=['csv'], key='up_peta',
                               help='CSV berkolom kode_pegawai (atau nip) dan nama, agar pegawai dapat dicari '
                                    'berdasarkan nama. Diproses di memori dan tidak disimpan.')

d0 = None
try:
    if up_latih is not None:
        d0 = baca_csv(up_latih.getvalue()); nama_latih = up_latih.name
    elif os.path.exists(BAWAAN):
        d0 = pd.read_csv(BAWAAN); nama_latih = f'{BAWAAN} (bawaan)'
except Exception as e:
    st.error(f'Data latih gagal dibaca: {e}')
    st.stop()
if d0 is None:
    st.error(f'Berkas **{BAWAAN}** tidak ditemukan dan belum ada data latih yang diunggah. '
             'Unggah data latih melalui panel kiri atau salin berkas dataset ke folder app.py.')
    st.stop()

peta = None
if up_peta is not None:
    try:
        peta = baca_csv(up_peta.getvalue())
    except Exception as e:
        st.sidebar.error(f'Peta identitas gagal dibaca: {e}')

d = pasang_peta(pm.normalkan(d0), peta)
if 'tw' not in d.columns:
    st.error('Data latih harus memuat kolom **tw** (misalnya 2026-TW1) atau kolom **tahun** dan **triwulan**.')
    st.stop()

do, nama_olah = None, None
if up_olah is not None:
    try:
        do = pasang_peta(pm.normalkan(baca_csv(up_olah.getvalue())), peta); nama_olah = up_olah.name
        if 'tw' not in do.columns:
            do['tw'] = 'tanpa periode'
    except Exception as e:
        st.sidebar.error(f'Data uji gagal dibaca: {e}')

HM = muat_json('hasil_model.json')
HI = muat_json('hasil_interpretasi.json')
HD = muat_json('hasil_skenarioD.json')
HP = muat_json('hasil_prioritas.json')

TW = sorted(d.tw.unique())
peg = d.drop_duplicates('kode')
KEL = sorted(peg.kelompok_pegawai.unique()) if 'kelompok_pegawai' in d else []
PEND = sorted(peg.pendidikan.unique()) if 'pendidikan' in d else []

with st.sidebar:
    contoh = (pd.read_csv(BAWAAN).head(20) if os.path.exists(BAWAAN) else d0.head(20))
    st.download_button('Unduh contoh format CSV', contoh.to_csv(index=False).encode('utf-8'),
                       'contoh_format_dataset.csv', 'text/csv')
    st.caption(f'Data latih: `{nama_latih}` · {len(d):,} baris · {d.kode.nunique()} pegawai'.replace(',', '.'))
    st.caption(f'Data uji: `{nama_olah}` · {len(do):,} baris · {do.kode.nunique()} pegawai'.replace(',', '.')
               if do is not None else 'Data uji: belum diunggah')
    with st.expander('Bagi dataset 80:20 per pegawai'):
        st.caption('Membagi data latih yang sedang dipakai menjadi 80% data latih dan 20% data uji. Semua baris '
                   'seorang pegawai masuk ke satu sisi, dan proporsi kelompok pegawai serta predikat Sangat Baik '
                   'dijaga seimbang. Unduh kedua berkas, lalu unggah kembali ke kotak di atas.')
        bibit = st.number_input('Bibit acak', 0, 9999, 42, help='Bibit yang sama selalu menghasilkan pembagian yang sama.')
        if st.button('Buat pembagian 80:20'):
            st.session_state['bagi'] = pm.bagi_latih_uji(d0, 0.2, int(bibit))
        if 'bagi' in st.session_state:
            lt, uj, inf = st.session_state['bagi']
            st.caption(f"Latih: {inf['latih']['pegawai']} pegawai, {inf['latih']['baris']} baris, "
                       f"SB {rp(inf['latih']['persen_sb'], 1)}% · Uji: {inf['uji']['pegawai']} pegawai, "
                       f"{inf['uji']['baris']} baris, SB {rp(inf['uji']['persen_sb'], 1)}%")
            st.download_button('Unduh data latih 80%', lt.to_csv(index=False).encode('utf-8'), 'data_latih_80.csv',
                               'text/csv')
            st.download_button('Unduh data uji 20%', uj.to_csv(index=False).encode('utf-8'), 'data_uji_20.csv',
                               'text/csv')
    if peta is not None:
        st.caption('Peta identitas terpasang: pencarian nama aktif.')
    st.divider()
    st.markdown('**Penyaring data**')
    st.caption('Berlaku untuk tab Ringkasan dan Profil Data.')
    f_tw = st.multiselect('Triwulan', TW, default=TW)
    f_kel = st.multiselect('Kelompok pegawai', KEL, default=KEL) if KEL else []
    f_pend = st.multiselect('Pendidikan', PEND, default=PEND) if PEND else []

ds = d[d.tw.isin(f_tw)]
if KEL:
    ds = ds[ds.kelompok_pegawai.isin(f_kel)]
if PEND:
    ds = ds[ds.pendidikan.isin(f_pend)]

st.title('Klasifikasi dan Prioritas Pengembangan Kompetensi Pegawai ASN')
st.markdown('<span class="kecil">Studi Kasus pada BBPPMPV Bidang Pertanian · '
            'identitas pegawai disamarkan menjadi kode berurut</span>', unsafe_allow_html=True)

T = st.tabs(['Ringkasan', 'Profil Data', 'Matriks Kompetensi', 'Perbandingan Model',
             'Latih & Uji Model', 'Interpretasi (LIME)', 'Prioritas AHP-TOPSIS'])


def ada(df, *kolom):
    return all(k in df.columns for k in kolom)


# ============================================================== 1. RINGKASAN
with T[0]:
    if ds.empty:
        st.warning('Tidak ada baris yang cocok dengan penyaring. Longgarkan pilihan pada panel kiri.')
    else:
        lab = ds.skp_predikat if 'skp_predikat' in ds else pd.Series(dtype=object)
        c = st.columns(5)
        c[0].metric('Baris pegawai–triwulan', f'{len(ds):,}'.replace(',', '.'))
        c[1].metric('Pegawai', f'{ds.kode.nunique()}')
        c[2].metric('Triwulan', f'{ds.tw.nunique()}')
        berlabel = int(lab.notna().sum())
        c[3].metric('Baris berlabel', f'{berlabel:,}'.replace(',', '.'))
        sb = int((lab == 'Sangat Baik').sum())
        c[4].metric('Predikat Sangat Baik', f"{sb} ({rp(100 * sb / berlabel, 1)}%)" if berlabel else '0')

        st.subheader('Kelengkapan indikator per triwulan')
        st.caption('Sel kosong berarti sumber data belum tersedia pada periode itu — bukan bernilai nol.')
        INDI = [('Absensi', 'absensi_hari'), ('Apel', 'apel_undangan'), ('Pelatihan', 'pelatihan_jumlah')]
        rek = ds.groupby('tw').apply(lambda g: pd.Series(
            {**{n: int((g[k].fillna(0) > 0).sum()) for n, k in INDI if k in g},
             'Berlabel': int(g.skp_predikat.notna().sum()) if 'skp_predikat' in g else 0}),
            include_groups=False).reset_index()
        f = go.Figure()
        for k, w in zip([c for c in rek.columns if c != 'tw'], URUT):
            f.add_bar(x=rek.tw, y=rek[k], name=k, marker_color=w)
        f.update_layout(barmode='group', height=330, template='plotly_white',
                        margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation='h', y=1.14, x=0),
                        yaxis_title='Jumlah pegawai yang punya catatan')
        st.plotly_chart(f, **W)

        k1, k2 = st.columns(2)
        with k1:
            st.subheader('Kehadiran apel per triwulan')
            if 'apel_persen_hadir' in ds:
                ap = ds[ds.apel_persen_hadir.notna()].groupby('tw').apel_persen_hadir.mean().reset_index()
                g = go.Figure(go.Scatter(x=ap.tw, y=ap.apel_persen_hadir, mode='lines+markers',
                                         line=dict(color=NAVY, width=3)))
                g.update_layout(height=300, template='plotly_white', yaxis_title='Rerata persentase hadir',
                                margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(g, **W)
        with k2:
            st.subheader('Proporsi predikat Sangat Baik')
            if berlabel:
                pr = (ds[ds.skp_predikat.notna()].groupby('tw').skp_predikat
                      .apply(lambda s: 100 * (s == 'Sangat Baik').mean()).reset_index(name='persen'))
                g = px.bar(pr, x='tw', y='persen', color_discrete_sequence=[NAVY])
                g.update_layout(height=300, template='plotly_white', yaxis_title='Persen',
                                xaxis_title='', margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(g, **W)

# ============================================================ 2. PROFIL DATA
with T[1]:
    pg = ds.drop_duplicates('kode')
    c = st.columns(3)
    with c[0]:
        st.subheader('Kelompok pegawai')
        if 'kelompok_pegawai' in pg:
            v = pg.kelompok_pegawai.value_counts()
            st.plotly_chart(px.pie(values=v.values, names=v.index, hole=.55, color_discrete_sequence=URUT)
                            .update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0)), **W)
    with c[1]:
        st.subheader('Pendidikan')
        if 'pendidikan' in pg:
            v = pg.pendidikan.value_counts()
            st.plotly_chart(px.bar(x=v.index, y=v.values, color_discrete_sequence=[NAVY])
                            .update_layout(height=280, template='plotly_white', xaxis_title='',
                                           yaxis_title='Pegawai', margin=dict(l=0, r=0, t=10, b=0)), **W)
    with c[2]:
        st.subheader('Masa kerja')
        if 'masa_kerja_tahun' in pg:
            st.plotly_chart(px.histogram(pg, x='masa_kerja_tahun', nbins=20, color_discrete_sequence=[EMAS])
                            .update_layout(height=280, template='plotly_white', xaxis_title='Tahun',
                                           yaxis_title='Pegawai', margin=dict(l=0, r=0, t=10, b=0)), **W)

    st.subheader('Indikator perilaku menurut predikat kinerja')
    st.caption('Temuan utama penelitian: arah hubungannya berlawanan dengan dugaan awal.')
    if 'skp_predikat' in ds:
        b = ds[ds.skp_predikat.isin(['Baik', 'Sangat Baik'])]
        kol = [k for k in ['apel_persen_hadir', 'absensi_rata_jam_masuk', 'absensi_rata_durasi',
                           'absensi_tanpa_checkout', 'pelatihan_kumulatif'] if k in b]
        if len(b) and kol:
            rk = b.groupby('skp_predikat')[kol].mean().round(2).rename(columns=pm.LABEL).T
            st.dataframe(rk, **W)
            pil = st.selectbox('Sebaran rinci', kol, format_func=lambda x: pm.LABEL.get(x, x))
            st.plotly_chart(px.box(b, x='skp_predikat', y=pil, color='skp_predikat',
                                   color_discrete_sequence=[NAVY, EMAS], points='outliers')
                            .update_layout(height=340, template='plotly_white', showlegend=False, xaxis_title='',
                                           yaxis_title=pm.LABEL.get(pil, pil), margin=dict(l=10, r=10, t=10, b=10)),
                            **W)
    with st.expander('Pratinjau data'):
        st.dataframe(ds.drop(columns=['nama', 'nip'], errors='ignore').head(300), height=320, **W)

# ===================================================== 3. MATRIKS KOMPETENSI
with T[2]:
    if HP is None:
        st.info('Berkas `hasil_prioritas.json` tidak ditemukan, sehingga bagian ini kosong.')
    else:
        K = HP['kompetensi']
        c = st.columns(4)
        c[0].metric('Matriks lengkap', f"{K['lengkap']} / {K['total_pegawai']}",
                    f"{100 * K['lengkap'] / K['total_pegawai']:.1f}% cakupan")
        c[1].metric('Sudah sesuai (total > 8)', K['sesuai'])
        c[2].metric('Belum sesuai', K['belum'])
        c[3].metric('Mencapai nilai 12', 0)
        nm = {'k_formal': 'Pendidikan formal', 'k_nonformal': 'Pendidikan non-formal',
              'k_terampil': 'Keterampilan', 'k_alaman': 'Pengalaman'}
        asp = {a: {str(int(float(k))): v for k, v in dd.items()} for a, dd in K['aspek'].items()}
        f = go.Figure()
        for nilai, warna, lab in [('1', MERAH, '1 — kurang'), ('2', EMAS, '2 — sesuai'), ('3', HIJAU, '3 — lebih')]:
            f.add_bar(y=[nm[a] for a in nm], x=[asp[a].get(nilai, 0) for a in nm], name=lab, orientation='h',
                      marker_color=warna, text=[asp[a].get(nilai, 0) for a in nm], textposition='inside')
        f.update_layout(barmode='stack', height=330, template='plotly_white',
                        xaxis_title=f"Jumlah pegawai (n = {K['lengkap']})", legend=dict(orientation='h', y=1.14, x=0),
                        margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(f, **W)
        st.warning('**Kesenjangan terbesar: pendidikan non-formal** — '
                   f"{asp['k_nonformal'].get('1', 0)} dari {K['lengkap']} pegawai pada nilai terendah. "
                   'Sebagian besar mencerminkan rendahnya mutu pencatatan riwayat pelatihan, '
                   'bukan semata rendahnya keikutsertaan.')
        tot = {int(float(k)): v for k, v in K['sebaran_total'].items()}
        f2 = px.bar(x=list(tot), y=list(tot.values()), color_discrete_sequence=[NAVY])
        f2.add_vline(x=8.5, line_dash='dash', line_color=MERAH, annotation_text='ambang kesesuaian',
                     annotation_position='top right')
        f2.update_layout(height=300, template='plotly_white', xaxis_title='Total nilai (maksimum 12)',
                         yaxis_title='Jumlah pegawai', margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(f2, **W)

# ====================================================== 4. PERBANDINGAN MODEL (hasil penelitian)
with T[3]:
    st.caption('Hasil tersimpan dari penelitian (Bab IV). Untuk melatih ulang pada data Anda sendiri, '
               'gunakan tab **Latih & Uji Model**.')
    if HM is None or HD is None:
        st.info('Berkas hasil pemodelan tidak ditemukan, sehingga bagian ini kosong.')
    else:
        SK = {'A': 'A — label triwulanan asli (616 baris)', 'B': 'B — ditambah 2025 berlabel replikasi (1.432 baris)',
              'C': 'C — ditambah indikator asesmen (348 baris)', 'D': 'D — hanya indikator perilaku (616 baris)'}
        NMET = {'f1': 'F1-Score', 'auc': 'AUC-ROC', 'mcc': 'MCC', 'recall': 'Recall', 'presisi': 'Presisi',
                'akurasi': 'Akurasi'}
        met = st.radio('Metrik', list(NMET), horizontal=True, format_func=NMET.get)
        MODEL = ['Regresi Logistik', 'Random Forest', 'XGBoost']
        f = go.Figure()
        for m, w in zip(MODEL, URUT):
            nilai = [(HM[k]['metrik'] if k in HM else HD['metrik'])[m][met] for k in SK]
            f.add_bar(x=list(SK), y=[v[0] for v in nilai], name=m, marker_color=w,
                      error_y=dict(type='data', array=[v[1] for v in nilai], color='#555'),
                      text=[f'{v[0]:.3f}' for v in nilai], textposition='outside')
        f.update_layout(barmode='group', height=400, template='plotly_white', yaxis_title=NMET[met],
                        yaxis_range=[0, 1.05], legend=dict(orientation='h', y=1.12, x=0),
                        margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(f, **W)
        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader('Uji kebermaknaan selisih F1-Score (skenario A)')
            ut = pd.DataFrame([(k, v[0], v[1], 'Bermakna' if v[1] < .05 else 'Tidak bermakna')
                               for k, v in HM['A']['uji_t'].items()],
                              columns=['Pasangan model', 'Statistik t', 'Nilai p', 'Simpulan'])
            st.dataframe(ut, hide_index=True, **W)
            st.error('**Keunggulan model ensemble atas model linier tidak terbukti** pada ukuran data ini. '
                     'Selisih Random Forest terhadap Regresi Logistik tidak bermakna (p = 0,307), sedangkan pada '
                     'skenario D Regresi Logistik justru mengungguli kedua model ensemble.')
        with c2:
            st.subheader('Matriks konfusi')
            pilih = st.selectbox('Skenario', ['A — seluruh variabel', 'D — perilaku saja'])
            cm = np.array((HI if pilih.startswith('A') else HD)['konfusi'])
            f2 = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', labels=dict(x='Prediksi', y='Aktual'),
                           x=['Bukan SB', 'Sangat Baik'], y=['Bukan SB', 'Sangat Baik'])
            f2.update_layout(height=300, coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(f2, **W)
            tn, fp, fn, tp = cm.ravel()
            st.caption(f'TN {tn} · FP {fp} · FN {fn} · TP {tp}')


# ====================================================== 5. LATIH & UJI MODEL
def gambar_konfusi(cm, tinggi=280):
    f = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', labels=dict(x='Prediksi', y='Aktual'),
                  x=['Bukan SB', 'Sangat Baik'], y=['Bukan SB', 'Sangat Baik'])
    f.update_layout(height=tinggi, coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
    return f


def kelengkapan(df, kolom):
    ada_k = [k for k in kolom if k in df.columns]
    return df[ada_k].notna().sum(axis=1).astype(int).astype(str) + f'/{len(kolom)}'


def persen(a, b):
    return f'{100 * a / b:.0f}%' if b else '–'


def kotak_langkah(kol, no, judul, ket, selesai):
    warna = HIJAU if selesai else '#C9CED8'
    tanda = '✓' if selesai else no
    kol.markdown(f"""<div style="border:2px solid {warna};border-radius:10px;padding:10px 12px;height:96px">
<div style="display:flex;gap:10px;align-items:center"><div style="width:28px;height:28px;border-radius:50%;
background:{warna};color:white;font-weight:700;display:flex;align-items:center;justify-content:center">{tanda}</div>
<b style="color:{NAVY}">{judul}</b></div><div class="kecil" style="margin-top:6px">{ket}</div></div>""",
                 unsafe_allow_html=True)


def kotak_konfusi_kata(cm, judul):
    """Matriks konfusi dalam bahasa sehari-hari."""
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    st.markdown(f'**{judul}**')
    k = st.columns(4)
    isi = [(f'✅ {tp}', 'Sangat Baik dan dikenali', f'{persen(tp, tp + fn)} dari pegawai Sangat Baik', HIJAU_M),
           (f'⚠️ {fn}', 'Sangat Baik tetapi terlewat', 'disebut bukan Sangat Baik', MERAH_M),
           (f'⚠️ {fp}', 'Dikira Sangat Baik padahal bukan', 'paling merugikan untuk prioritas', MERAH_M),
           (f'✅ {tn}', 'Bukan Sangat Baik dan dikenali', f'{persen(tn, tn + fp)} dari yang bukan SB', HIJAU_M)]
    for kol, (angka, lab, ket, bg) in zip(k, isi):
        kol.markdown(f"""<div style="background:{bg};border-radius:8px;padding:10px 12px;height:110px">
<div style="font-size:1.5rem;font-weight:700;color:{NAVY}">{angka}</div><div><b>{lab}</b></div>
<div class="kecil">{ket}</div></div>""", unsafe_allow_html=True)


HIJAU_M, MERAH_M = '#EAF4EE', '#FBECEE'
KAMUS = [
    ('Data latih (80%)', 'Data yang dipakai model untuk belajar. Di dalamnya model juga divalidasi silang per pegawai.'),
    ('Data uji (20%)', 'Pegawai yang sama sekali tidak dilihat model saat belajar. Hasil pada data ini paling jujur '
                       'menggambarkan kinerja model pada pegawai baru.'),
    ('Recall', 'Dari semua pegawai yang benar-benar Sangat Baik, berapa persen yang ditemukan model.'),
    ('Presisi', 'Dari semua yang ditebak Sangat Baik oleh model, berapa persen yang benar.'),
    ('F1-Score', 'Gabungan recall dan presisi dalam satu angka (0–1). Dipakai sebagai ukuran utama karena kelas '
                 'Sangat Baik lebih sedikit daripada kelas lain.'),
    ('AUC', 'Kemampuan model mengurutkan pegawai dari yang paling berpeluang Sangat Baik. 0,5 = menebak acak, '
            '1 = sempurna.'),
    ('MCC', 'Ukuran keseluruhan yang memakai keempat isi matriks konfusi. 0 = setara tebakan acak, 1 = sempurna.'),
    ('Peluang Sangat Baik', 'Keyakinan model (0–100%) bahwa pegawai pada triwulan itu berpredikat Sangat Baik. '
                            'Di atas 50% ditebak Sangat Baik.'),
    ('p (uji Nadeau–Bengio)', 'Bila p ≥ 0,05, selisih dua model tidak bermakna: keduanya dianggap setara.'),
    ('LIME', 'Penjelasan per pegawai: faktor mana yang menaikkan atau menurunkan peluang pada satu triwulan.'),
]


with T[4]:
    cfg = None
    L = st.session_state.get('latih')
    # evaluasi hanya memakai label triwulanan asli (label tahunan replikasi tidak dihitung)
    y_uji = np.where(pm.apakah_replikasi(do), np.nan, pm.label_biner(do)) if do is not None else None
    uji_berlabel = do is not None and np.isfinite(y_uji).sum() >= 10
    st.subheader('Alur pengolahan machine learning')
    k = st.columns(4)
    kotak_langkah(k[0], '1', 'Data latih', f'{d.kode.nunique()} pegawai · {len(d):,} baris'.replace(',', '.'), True)
    kotak_langkah(k[1], '2', 'Data uji',
                  (f'{do.kode.nunique()} pegawai · {len(do):,} baris'.replace(',', '.') +
                   ('' if uji_berlabel else ' · tanpa label')) if do is not None else 'unggah di panel kiri',
                  do is not None)
    kotak_langkah(k[2], '3', 'Latih dan validasi', 'selesai' if L else 'tekan tombol Latih di bawah', L is not None)
    kotak_langkah(k[3], '4', 'Uji dan jelaskan', 'hasil uji di bawah · LIME di tab Interpretasi'
                  if (L and uji_berlabel) else 'menunggu langkah 2 dan 3', bool(L and uji_berlabel))
    if do is not None:
        tumpang = set(do.kode) & set(d.kode)
        if tumpang:
            st.error(f'{len(tumpang)} pegawai ada di data latih **dan** data uji. Hasil uji menjadi terlalu optimistis. '
                     'Gunakan berkas hasil pembagian 80:20 per pegawai (panel kiri → Bagi dataset 80:20).')
    mode = st.radio('Tampilan hasil', ['Sederhana', 'Rinci'], horizontal=True, key='mode_ml',
                    help='Sederhana: ringkasan dalam kalimat. Rinci: seluruh tabel statistik seperti di laporan.')
    sederhana = mode == 'Sederhana'
    st.divider()

    st.markdown('#### Langkah 3 — Latih dan validasi')
    if 'skp_predikat' not in d.columns and 'skp_sangat_baik' not in d.columns:
        st.error('Data latih tidak memuat kolom label **skp_predikat** (atau **skp_sangat_baik**).')
    else:
        c = st.columns([1.5, 2.1])
        sk = c[0].radio('Himpunan fitur', list(pm.SKENARIO), index=2, format_func=lambda k: pm.SKENARIO[k]['nama'],
                        help='Skenario D direkomendasikan karena tidak memuat atribut yang tidak dapat diubah '
                             '(pendidikan, kelompok pegawai, status, usia).')
        model_dipilih = c[1].multiselect('Model yang dibandingkan', pm.MODEL_TERSEDIA, default=pm.MODEL_TERSEDIA)
        rep_tahun = []
        with st.expander('Pengaturan lanjutan', expanded=not sederhana):
            a = st.columns(3)
            ulang = a[0].selectbox('Pengulangan validasi', [1, 3], index=1,
                                   help='3 kali seperti penelitian (15 penaksiran); 1 kali lebih cepat.')
            smote = a[1].checkbox('SMOTE pada lipatan latih', value=True)
            rep = pm.apakah_replikasi(d)
            if rep.any() and 'tahun' in d.columns:
                th = sorted(d.loc[rep, 'tahun'].dropna().astype(int).unique())
                rep_tahun = a[2].multiselect('Label tahunan replikasi (bobot 0,25)', th, default=[])
        kurang = pm.kolom_kurang(d, sk)
        if kurang:
            st.error('Kolom berikut tidak ada pada data latih: ' + ', '.join(f'`{k}`' for k in kurang))
        else:
            dl = pm.siapkan_latih(d, sk, rep_tahun)
            npos = int(dl._y.sum())
            st.caption(f'Dipakai untuk belajar: {len(dl)} baris berlabel dari {dl.kode.nunique()} pegawai, '
                       f'{npos} di antaranya Sangat Baik ({rp(100 * npos / max(len(dl), 1), 1)}%).')
            layak = len(dl) >= 30 and 5 <= npos <= len(dl) - 5 and dl.kode.nunique() >= 5
            if not layak:
                st.error('Data latih terlalu sedikit (minimal 30 baris berlabel, masing-masing kelas lebih dari 5, '
                         'dan 5 pegawai).')
            cfg = (sidik(d), sk, tuple(model_dipilih), ulang, smote, tuple(rep_tahun))
            if st.button('Latih dan validasi model', type='primary', disabled=not (layak and model_dipilih)):
                bar = st.progress(0.0, text='Memvalidasi model…')
                ringkas, oof = pm.validasi(dl, sk, model_dipilih, ulang=ulang, pakai_smote=smote,
                                           progres=lambda x: bar.progress(x, text=f'Memvalidasi model… {x:.0%}'))
                bar.progress(1.0, text='Melatih model akhir pada seluruh data latih…')
                paket = {nm: pm.latih_akhir(dl, sk, nm, smote) for nm in model_dipilih}
                bar.empty()
                st.session_state['latih'] = dict(cfg=cfg, sk=sk, ringkas=ringkas, oof=oof, paket=paket, dl=dl,
                                                 ulang=ulang, smote=smote, sumber=nama_latih, lime={})
                L = st.session_state['latih']

    if L:
        if cfg is not None and L.get('cfg') is not None and L['cfg'] != cfg:
            st.info('Pengaturan atau data latih berubah sejak pelatihan terakhir. Hasil di bawah masih milik '
                    'pelatihan sebelumnya; tekan **Latih dan validasi model** untuk memperbarui.')
        R = L['ringkas']
        nm_list = list(L['paket'])

        # ---- hasil pada data uji untuk semua model
        HU = {}
        if uji_berlabel:
            ok = np.isfinite(y_uji)
            for nm, pk in L['paket'].items():
                if all(c in do.columns for c in pk['num'] + pk['kat']):
                    pr = pm.peluang(pk, do[ok]); yy = y_uji[ok].astype(int)
                    HU[nm] = dict(m=pm._metrik(yy, (pr >= .5).astype(int), pr),
                                  cm=confusion_matrix(yy, (pr >= .5).astype(int), labels=[0, 1]))

        # ---- ringkasan dalam kalimat
        st.markdown('#### Ringkasan hasil')
        if R is not None:
            urut = sorted(R, key=lambda m: -R[m]['rerata']['f1'])
            terbaik = urut[0]
            kal = [f"Model dilatih dengan **{L['dl'].kode.nunique()} pegawai** "
                   f"({len(L['dl'])} baris berlabel) dan divalidasi silang per pegawai."]
            kal.append(f"Pada validasi, model dengan F1 tertinggi adalah **{terbaik}** "
                       f"(F1 {rp(R[terbaik]['rerata']['f1'])}).")
            if len(urut) > 1 and len(R[terbaik]['lipatan']) > 1:
                t, p = pm.uji_nadeau_bengio(R[terbaik]['lipatan'].f1, R[urut[1]]['lipatan'].f1)
                if p >= .05:
                    kal.append(f"Selisihnya dengan **{urut[1]}** **tidak bermakna** (p = {rp(p)}), sehingga keduanya "
                               'dapat dianggap setara' + ('; Regresi Logistik lebih mudah dijelaskan.'
                                                          if 'Regresi Logistik' in (terbaik, urut[1]) else '.'))
                else:
                    kal.append(f"Keunggulannya atas **{urut[1]}** **bermakna** (p = {rp(p)}).")
        else:
            terbaik = nm_list[0]
            kal = ['Model dilatih cepat tanpa validasi silang (dari tab Interpretasi).']
        nm_r = st.selectbox('Model yang diringkas', nm_list, index=nm_list.index(terbaik), key='ringkas_model')
        if nm_r in HU:
            m_ = HU[nm_r]['m']; tn, fp, fn, tp = [int(x) for x in HU[nm_r]['cm'].ravel()]
            kal.append(f"Pada **data uji** ({do.kode.nunique()} pegawai yang tidak ikut dilatih, "
                       f"{int(np.isfinite(y_uji).sum())} baris berlabel triwulanan), **{nm_r}** menemukan "
                       f"**{tp} dari {tp + fn}** kasus Sangat Baik ({persen(tp, tp + fn)}), dan **{tp} dari {tp + fp}** "
                       f"tebakan Sangat Baiknya benar ({persen(tp, tp + fp)}).")
            if R is not None:
                sel = m_['f1'] - R[nm_r]['rerata']['f1']
                kal.append(f"F1 uji {rp(m_['f1'])} dibanding F1 validasi {rp(R[nm_r]['rerata']['f1'])}: " +
                           ('sebanding, jadi model **tidak sekadar menghafal** data latih.' if abs(sel) <= .08 else
                            ('lebih tinggi; wajar karena data uji kecil, tetapi tafsirkan dengan hati-hati.'
                             if sel > 0 else 'jauh lebih rendah; model kemungkinan **terlalu menyesuaikan** data latih.')))
        elif do is None:
            kal.append('Unggah **data uji (20%)** pada panel kiri untuk melihat kinerja pada pegawai yang belum pernah '
                       'dilihat model.')
        st.info('\n\n'.join(kal))

        # kartu metrik dengan arti
        sumber_m = HU[nm_r]['m'] if nm_r in HU else (R[nm_r]['rerata'] if R is not None else None)
        if sumber_m is not None:
            st.caption('Angka pada data uji.' if nm_r in HU else 'Angka rerata validasi silang (data uji belum ada).')
            k = st.columns(4)
            for kol, (kk, lab, arti) in zip(k, [
                    ('recall', 'Recall', 'pegawai Sangat Baik yang ditemukan'),
                    ('presisi', 'Presisi', 'tebakan Sangat Baik yang benar'),
                    ('f1', 'F1-Score', 'keseimbangan keduanya (ukuran utama)'),
                    ('auc', 'AUC', 'kemampuan mengurutkan (0,5 = acak)')]):
                kol.metric(lab, rp(sumber_m[kk]), help=arti)
                kol.caption(arti)
        if nm_r in HU:
            kotak_konfusi_kata(HU[nm_r]['cm'], f'Isi matriks konfusi {nm_r} pada data uji')
        elif R is not None and nm_r in L['oof']:
            prob = L['oof'][nm_r]; ok = ~np.isnan(prob)
            kotak_konfusi_kata(confusion_matrix(L['dl']._y.values[ok], (prob[ok] >= .5).astype(int), labels=[0, 1]),
                               f'Isi matriks konfusi {nm_r} pada validasi (luar-lipatan)')

        # perbandingan validasi vs uji
        if R is not None:
            f = go.Figure()
            f.add_bar(x=list(R), y=[R[m]['rerata']['f1'] for m in R], name='Validasi (data latih)',
                      marker_color=NAVY, text=[rp(R[m]['rerata']['f1']) for m in R], textposition='outside')
            if HU:
                f.add_bar(x=list(R), y=[HU[m]['m']['f1'] if m in HU else None for m in R], name='Uji (20%)',
                          marker_color=EMAS, text=[rp(HU[m]['m']['f1']) if m in HU else '' for m in R],
                          textposition='outside')
            f.update_layout(barmode='group', height=300, template='plotly_white', yaxis_range=[0, 1.05],
                            yaxis_title='F1-Score', title='F1-Score per model',
                            legend=dict(orientation='h', y=1.18, x=0), margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(f, **W)

        with st.expander('Kamus istilah — cara membaca angka di atas'):
            for a_, b_ in KAMUS:
                st.markdown(f'**{a_}** — {b_}')

        # ---- tampilan rinci
        if not sederhana and R is not None:
            st.markdown('#### Rincian statistik')
            tb = pd.DataFrame([{'Model': nm, **{lab: f"{rp(v['rerata'][k])} ± {rp(v['sb'][k])}"
                                                for k, lab in [('f1', 'F1-Score'), ('auc', 'AUC'), ('mcc', 'MCC'),
                                                               ('akurasi', 'Akurasi'), ('presisi', 'Presisi'),
                                                               ('recall', 'Recall')]}} for nm, v in R.items()])
            st.markdown('**Validasi silang pada data latih**')
            st.dataframe(tb, hide_index=True, **W)
            n_est = len(next(iter(R.values()))['lipatan'])
            st.caption(f'Rerata ± simpangan baku dari {n_est} penaksiran. Angka dapat sedikit berbeda dari laporan '
                       'karena pembagian lipatan bersifat acak.')
            if HU:
                st.markdown('**Kinerja pada data uji**')
                st.dataframe(pd.DataFrame([{'Model': nm, **{lab: rp(v['m'][k]) for k, lab in
                                                            [('f1', 'F1-Score'), ('auc', 'AUC'), ('mcc', 'MCC'),
                                                             ('akurasi', 'Akurasi'), ('presisi', 'Presisi'),
                                                             ('recall', 'Recall')]}} for nm, v in HU.items()]),
                             hide_index=True, **W)
            c1, c2 = st.columns([3, 2])
            with c1:
                if len(R) > 1 and n_est > 1:
                    nm_l = list(R); baris = []
                    for i in range(len(nm_l)):
                        for j in range(i + 1, len(nm_l)):
                            t, p = pm.uji_nadeau_bengio(R[nm_l[i]]['lipatan'].f1, R[nm_l[j]]['lipatan'].f1)
                            baris.append([f'{nm_l[i]} vs {nm_l[j]}', rp(t), rp(p),
                                          'Bermakna' if p < .05 else 'Tidak bermakna'])
                    st.markdown('**Uji t terkoreksi Nadeau–Bengio atas F1-Score**')
                    st.dataframe(pd.DataFrame(baris, columns=['Pasangan model', 't', 'p', 'Simpulan (α = 0,05)']),
                                 hide_index=True, **W)
            with c2:
                pm_k = st.selectbox('Matriks konfusi', list(R), key='cm_model')
                if pm_k in HU:
                    cm = HU[pm_k]['cm']; ket = 'data uji'
                else:
                    prob = L['oof'][pm_k]; ok = ~np.isnan(prob); ket = 'validasi luar-lipatan'
                    cm = confusion_matrix(L['dl']._y.values[ok], (prob[ok] >= .5).astype(int), labels=[0, 1])
                st.plotly_chart(gambar_konfusi(cm), **W)
                tn, fp, fn, tp = cm.ravel()
                st.caption(f'TN {tn} · FP {fp} · FN {fn} · TP {tp} ({ket})')
            with st.expander('Kepentingan fitur atau odds ratio model akhir'):
                pk_nm = st.selectbox('Model', nm_list, key='imp_model')
                imp, jenis = pm.kepentingan(L['paket'][pk_nm])
                imp = imp.head(12)
                if jenis == 'odds':
                    f = go.Figure(go.Bar(x=imp.Nilai, y=imp.Fitur, orientation='h',
                                         marker_color=[HIJAU if x > 1 else MERAH for x in imp.Nilai],
                                         text=[f'{x:.3f}' for x in imp.Nilai], textposition='outside'))
                    f.add_vline(x=1, line_dash='dash', line_color='#333')
                    judul = 'Odds ratio per kenaikan satu simpangan baku (> 1 menaikkan peluang Sangat Baik)'
                else:
                    f = go.Figure(go.Bar(x=100 * imp.Nilai, y=imp.Fitur, orientation='h', marker_color=NAVY,
                                         text=[f'{100 * x:.1f}%' for x in imp.Nilai], textposition='outside'))
                    judul = 'Kontribusi terhadap pemisahan kelas (%)'
                f.update_layout(height=60 + 28 * len(imp), template='plotly_white', title=judul,
                                yaxis=dict(autorange='reversed'), margin=dict(l=10, r=50, t=40, b=10))
                st.plotly_chart(f, **W)

        # ---- prediksi per baris
        st.divider()
        st.markdown('#### Langkah 4 — Prediksi per pegawai pada data uji')
        sumber_olah = do if do is not None else d
        if do is None:
            st.caption('Data uji belum diunggah, sehingga prediksi dijalankan pada data latih (bukan pengujian '
                       'independen).')
        nm_pred = st.selectbox('Model untuk prediksi', nm_list, key='pred_model', index=nm_list.index(nm_r))
        paket = L['paket'][nm_pred]
        kurang = [k for k in paket['num'] + paket['kat'] if k not in sumber_olah.columns]
        if kurang:
            st.error('Data uji tidak memuat kolom: ' + ', '.join(f'`{k}`' for k in kurang))
        else:
            hasil = sumber_olah.copy()
            hasil['peluang_sb'] = pm.peluang(paket, hasil)
            hasil['prediksi'] = np.where(hasil.peluang_sb >= .5, 'Sangat Baik', 'Bukan SB')
            yo = pm.label_biner(hasil); rp_ = pm.apakah_replikasi(hasil)
            hasil['aktual'] = np.where(np.isnan(yo), '–', np.where(yo == 1, 'Sangat Baik', 'Bukan SB'))
            hasil.loc[rp_ & ~np.isnan(yo), 'aktual'] += ' (tahunan)'
            hasil['status'] = np.where(np.isnan(yo) | rp_, '–', np.where((hasil.peluang_sb >= .5) == (yo == 1),
                                                                         '✅ tepat', '⚠️ keliru'))
            hasil['fitur_terisi'] = kelengkapan(hasil, paket['num'])
            pil_tw = st.multiselect('Triwulan', sorted(hasil.tw.unique()),
                                    default=[sorted(hasil.tw.unique())[-1]], key='olah_tw')
            tampil = hasil[hasil.tw.isin(pil_tw)] if pil_tw else hasil
            kol = (['nama'] if 'nama' in tampil and tampil.nama.str.len().gt(0).any() else []) + \
                  ['kode'] + [k for k in ['jabatan'] if k in tampil] + \
                  ['tw', 'peluang_sb', 'prediksi', 'aktual', 'status'] + ([] if sederhana else ['fitur_terisi'])
            NAMA_KOL = {'nama': 'Nama', 'kode': 'Kode', 'jabatan': 'Jabatan', 'tw': 'Triwulan',
                        'peluang_sb': 'Peluang SB', 'prediksi': 'Prediksi', 'aktual': 'Aktual', 'status': 'Status',
                        'fitur_terisi': 'Indikator terisi'}
            tb = tampil.sort_values('peluang_sb', ascending=False)[kol].copy()
            tb['peluang_sb'] = (100 * tb.peluang_sb).round(0).astype(int).astype(str) + '%'
            st.dataframe(tb.rename(columns=NAMA_KOL), hide_index=True, height=360, **W)
            c1, c2 = st.columns([1, 2])
            with c1:
                st.download_button('Unduh hasil prediksi (CSV)', hasil[kol].to_csv(index=False).encode('utf-8'),
                                   f'prediksi_{nm_pred.replace(" ", "_").lower()}.csv', 'text/csv')
                st.caption(f'{len(tampil)} baris ditampilkan · {int((tampil.peluang_sb >= .5).sum())} diprediksi '
                           'Sangat Baik. Pilih pegawai di tab **Interpretasi (LIME)** untuk melihat alasannya.')
            with c2:
                f = px.histogram(tampil, x='peluang_sb', nbins=20, color='prediksi',
                                 color_discrete_map={'Sangat Baik': EMAS, 'Bukan SB': NAVY})
                f.add_vline(x=.5, line_dash='dash', line_color=MERAH)
                f.update_layout(height=260, template='plotly_white', xaxis_title='Peluang Sangat Baik',
                                yaxis_title='Jumlah baris', legend=dict(orientation='h', y=1.12, x=0),
                                margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(f, **W)


# ============================================================ 6. INTERPRETASI (LIME)
def tindak_lanjut(rinci, p_e, predikat_akt, sesuai):
    """Catatan tindak lanjut berbasis aturan, selaras dengan temuan Bab IV."""
    out = []
    f = {r['kolom']: r for r in rinci}
    if 'pelatihan_kumulatif' in f:
        if f['pelatihan_kumulatif']['bobot'] < 0:
            out.append('📚 **Akumulasi pelatihan menekan peluang.** Pegawai ini layak dipertimbangkan dalam alokasi '
                       'pelatihan (kriteria K3), karena pelatihan adalah faktor terkuat yang dapat diintervensi.')
        else:
            out.append('📚 Riwayat pelatihan **mendukung** peluang. Pertahankan keikutsertaan pengembangan kompetensi.')
    if 'absensi_tanpa_checkout' in f and f['absensi_tanpa_checkout']['bobot'] < 0:
        out.append('🕘 **Sering tidak absen pulang menekan peluang.** Periksa dulu apakah ini kelalaian mencatat '
                   'presensi sebelum ditafsirkan sebagai perilaku kerja.')
    tetap = sum(abs(r['bobot']) for r in rinci if r['jenis'] == 'Atribut tetap')
    total = sum(abs(r['bobot']) for r in rinci) or 1
    if tetap / total > .4:
        out.append(f'🏷️ **{tetap / total:.0%} pengaruh berasal dari atribut yang tidak dapat diubah** (usia, '
                   'pendidikan, kelompok, masa kerja). Untuk menilai perilaku kerja, gunakan skenario D.')
    if any(r['jenis'] == 'Kedisiplinan dan presensi' and r['bobot'] > 0 and r['kolom'] != 'absensi_tanpa_checkout'
           for r in rinci):
        out.append('ℹ️ Pada data penelitian ini, hubungan kehadiran dengan predikat **berlawanan dugaan** '
                   '(Subbab 4.6.3). Faktor kehadiran yang menaikkan peluang jangan dibaca sebagai saran mengurangi '
                   'kedisiplinan.')
    if sesuai is False:
        out.append('⚠️ **Prediksi berbeda dengan predikat aktual.** Jangan jadikan hasil model sebagai dasar '
                   'rekomendasi otomatis untuk pegawai ini.')
    if 0.4 <= p_e <= 0.6:
        out.append('⚖️ Peluang **dekat ambang 50%**: hasil klasifikasi mudah berubah dan perlu ditinjau manual.')
    return out


with T[5]:
    st.subheader('Interpretasi per pegawai (LIME)')
    L = st.session_state.get('latih')
    if not pm.ADA_LIME:
        st.error('Pustaka **lime** belum terpasang. Jalankan `pip install -r requirements.txt`.')
    elif L is None:
        st.info('Belum ada model terlatih. Latih model pada tab **Latih & Uji Model**, atau gunakan pelatihan cepat '
                'dengan pengaturan penelitian (skenario D, Random Forest).')
        if 'skp_predikat' in d.columns and not pm.kolom_kurang(d, 'D'):
            if st.button('Latih cepat: skenario D · Random Forest', type='primary'):
                with st.spinner('Melatih model…'):
                    dl = pm.siapkan_latih(d, 'D')
                    st.session_state['latih'] = dict(cfg=None, sk='D', ringkas=None, oof={}, dl=dl, ulang=0,
                                                     smote=True, sumber=nama_latih, lime={},
                                                     paket={'Random Forest': pm.latih_akhir(dl, 'D', 'Random Forest')})
                st.rerun()
    else:
        nm_list = list(L['paket'])
        c = st.columns([1.2, 1.2, 1])
        nm_e = c[0].selectbox('Model yang dijelaskan', nm_list,
                              index=nm_list.index('Random Forest') if 'Random Forest' in nm_list else 0, key='lime_model')
        if do is not None:
            asal = c[1].radio('Cari pegawai di', ['Data uji', 'Data latih'], horizontal=True, key='lime_asal')
        else:
            asal = 'Data latih'
            c[1].markdown(' ')
        sumber = do if asal == 'Data uji' else d
        n_f = c[2].slider('Jumlah faktor', 4, 12, 6, key='lime_nf')
        paket = L['paket'][nm_e]
        kurang = [k for k in paket['num'] + paket['kat'] if k not in sumber.columns]
        if kurang:
            st.error('Data tidak memuat kolom: ' + ', '.join(f'`{k}`' for k in kurang))
            sumber = sumber.iloc[0:0]
        pg = sumber.drop_duplicates('kode')
        punya_nama = 'nama' in pg and pg.nama.astype(str).str.len().gt(0).any()
        label_peg = {}
        for _, r in pg.iterrows():
            jab = f" · {r['jabatan']}" if 'jabatan' in pg and pd.notna(r.get('jabatan')) else ''
            label_peg[r['kode']] = (f"{r['nama']} ({r['kode']}){jab}" if punya_nama and str(r.get('nama', '')).strip()
                                    else f"{r['kode']}{jab}")
        urut_peg = sorted(label_peg, key=lambda k: label_peg[k].lower())
        c = st.columns([1.2, 2.2])
        cari = c[0].text_input('Cari nama, kode, atau jabatan', key='lime_cari',
                               placeholder='misalnya: Budi, PEG021, Pranata' if punya_nama else 'misalnya: PEG021, Pranata')
        if cari.strip():
            kata = cari.strip().lower()
            urut_peg = [k for k in urut_peg if kata in label_peg[k].lower()]
        kode = c[1].selectbox(f'Pegawai ({len(urut_peg)} cocok)', urut_peg,
                              index=0 if len(urut_peg) == 1 else None, format_func=label_peg.get,
                              placeholder='Pilih pegawai dari daftar…', key=f'lime_peg_{cari.strip().lower()}')
        if cari.strip() and not urut_peg:
            st.warning(f'Tidak ada pegawai yang cocok dengan “{cari}” pada {asal.lower()}.')
        st.caption(f"Model: {nm_e} · {pm.SKENARIO[L['sk']]['nama']} · data: "
                   f"{nama_olah if asal == 'Data uji' else nama_latih}" +
                   ('' if punya_nama else ' · pencarian nama aktif bila data atau peta identitas memuat kolom nama'))

        if kode is not None:
            rows = sumber[sumber.kode == kode].sort_values('tw').copy()
            r0 = rows.iloc[-1]
            # peluang: luar-lipatan untuk baris latih yang ikut divalidasi, model akhir untuk lainnya
            rows['peluang_sb'] = pm.peluang(paket, rows)
            rows['sumber_peluang'] = 'akhir'
            if asal == 'Data latih' and nm_e in L.get('oof', {}) and L.get('cfg') and L['cfg'][0] == sidik(d):
                oof = pd.Series(L['oof'][nm_e], index=L['dl']['_baris'].values)
                cocok = rows.index.intersection(oof.dropna().index)
                rows.loc[cocok, 'peluang_sb'] = oof.loc[cocok].values
                rows.loc[cocok, 'sumber_peluang'] = 'validasi'
            y_r = pm.label_biner(rows)
            rows['aktual'] = np.where(np.isnan(y_r), '–', np.where(y_r == 1, 'Sangat Baik', 'Bukan SB'))
            rows['predikat'] = rows['skp_predikat'].fillna('–').astype(str) if 'skp_predikat' in rows else rows.aktual
            rows.loc[pm.apakah_replikasi(rows) & (rows.predikat != '–'), 'predikat'] += ' (tahunan)'
            rows['fitur_terisi'] = kelengkapan(rows, paket['num'])
            rows['n_terisi'] = rows[[k for k in paket['num'] if k in rows]].notna().sum(axis=1)

            judul = label_peg[kode].split(' · ')[0]
            st.markdown(f'### {judul}')
            info = [(lab, r0[k]) for k, lab in [('jabatan', 'Jabatan'), ('kelompok_pegawai', 'Kelompok'),
                                                 ('pendidikan', 'Pendidikan'), ('masa_kerja_tahun', 'Masa kerja')]
                    if k in rows and pd.notna(r0[k])]
            st.markdown(' · '.join(f"**{lab}** {pm.format_nilai('masa_kerja_tahun', v) if lab == 'Masa kerja' else v}"
                                   for lab, v in info))

            pilihan_tw = list(rows.tw)
            akt = dict(zip(rows.tw, rows.aktual))
            bawaan_tw = next((t for t in reversed(pilihan_tw) if akt.get(t) != '–'), pilihan_tw[-1])
            tw_e = st.selectbox('Triwulan yang dijelaskan', pilihan_tw, index=pilihan_tw.index(bawaan_tw),
                                key=f'lime_tw_{kode}')
            baris_df = rows[rows.tw == tw_e].iloc[[0]]
            baris = baris_df.iloc[0]
            p_e = float(pm.peluang(paket, baris_df)[0])
            sesuai = None if baris['aktual'] == '–' else ((p_e >= .5) == (baris['aktual'] == 'Sangat Baik'))

            # ---- kartu ringkasan
            m = st.columns(4)
            m[0].metric('Peluang Sangat Baik', f'{100 * p_e:.0f}%')
            m[1].metric('Tebakan model', 'Sangat Baik' if p_e >= .5 else 'Bukan Sangat Baik')
            m[2].metric('Predikat aktual', baris['predikat'])
            m[3].metric('Kesesuaian', '–' if sesuai is None else ('✅ Tepat' if sesuai else '⚠️ Keliru'))

            if 'lime_pk' not in L or L.get('lime_pk_untuk') != (nm_e, id(paket['model'])):
                with st.spinner('Menyiapkan penjelas LIME…'):
                    L['lime_pk'] = pm.siapkan_lime(paket, L['dl'])
                    L['lime_pk_untuk'] = (nm_e, id(paket['model']))
            kunci = (nm_e, id(paket['model']), str(kode), tw_e, n_f, asal)
            if kunci not in L['lime']:
                with st.spinner('Menghitung penjelasan…'):
                    L['lime'][kunci] = pm.jelaskan_rinci(L['lime_pk'], baris, L['dl'], n_f)
            rinci, diisi = L['lime'][kunci]

            # ---- narasi
            naik = [r for r in rinci if r['bobot'] > 0]
            turun = [r for r in rinci if r['bobot'] < 0]
            fr = lambda r: f"*{r['faktor'].lower()} {r['nilai']}*"
            narasi = [f"Pada **{tw_e}**, model memperkirakan peluang pegawai ini mencapai predikat Sangat Baik sebesar "
                      f"**{100 * p_e:.0f}%** ({'di atas' if p_e >= .5 else 'di bawah'} ambang 50%)" +
                      ('.' if sesuai is None else f", dan predikat aktualnya **{baris['predikat']}** — tebakan model "
                       f"**{'tepat' if sesuai else 'keliru'}**.")]
            if turun:
                narasi.append('Peluang terutama **ditekan** oleh ' + ', '.join(fr(r) for r in turun[:2]) + '.')
            if naik:
                narasi.append(('Sebaliknya, ' if turun else 'Peluang terutama ') + '**dinaikkan** oleh ' +
                              ', '.join(fr(r) for r in naik[:2]) + '.')
            i_tw = pilihan_tw.index(tw_e)
            if i_tw > 0:
                sebelum = rows.iloc[i_tw - 1]
                beda = 100 * (rows.iloc[i_tw].peluang_sb - sebelum.peluang_sb)
                if abs(beda) >= 1:
                    narasi.append(f"Dibanding {sebelum.tw}, peluangnya {'naik' if beda > 0 else 'turun'} "
                                  f"{abs(beda):.0f} poin persentase.")
            st.info(' '.join(narasi))

            c1, c2 = st.columns([1.15, 1])
            with c1:
                st.markdown('**Faktor yang membentuk peluang**')
                fa = rinci[::-1]
                f = go.Figure(go.Bar(
                    x=[r['bobot'] for r in fa], y=[f"{r['faktor']}: {r['nilai']}" for r in fa], orientation='h',
                    marker_color=[HIJAU if r['bobot'] > 0 else MERAH for r in fa],
                    text=[f"{r['bobot']:+.3f}".replace('.', ',') for r in fa], textposition='outside',
                    customdata=[[r['kondisi'], r['jenis']] for r in fa],
                    hovertemplate='%{y}<br>kondisi: %{customdata[0]}<br>%{customdata[1]}<extra></extra>'))
                f.add_vline(x=0, line_color='#333')
                batas = max(abs(r['bobot']) for r in fa) * 1.5 if fa else 1
                f.update_layout(height=80 + 38 * len(fa), template='plotly_white',
                                xaxis=dict(title='◀ menurunkan peluang · menaikkan peluang ▶', range=[-batas, batas]),
                                margin=dict(l=10, r=20, t=10, b=10))
                st.plotly_chart(f, **W)
            with c2:
                st.markdown('**Catatan dan tindak lanjut**')
                for t in tindak_lanjut(rinci, p_e, baris['predikat'], sesuai):
                    st.markdown(t)
                if diisi:
                    st.warning('Indikator tidak tercatat pada triwulan ini dan diisi median data latih: ' +
                               ', '.join(diisi) + '. Penjelasan untuk indikator tersebut tidak mencerminkan pegawai.')
                if st.button('Periksa kestabilan penjelasan', key=f'stab_{kunci}',
                             help='Mengulang LIME dengan 5 bibit acak dan membandingkan lima faktor teratas '
                                  '(indeks Jaccard).'):
                    with st.spinner('Mengulang LIME lima kali…'):
                        L['lime'][('stab',) + kunci] = pm.kestabilan(paket, L['dl'], baris)
                if ('stab',) + kunci in L['lime']:
                    j = L['lime'][('stab',) + kunci]
                    vonis = 'stabil' if j >= .8 else ('cukup stabil' if j >= .6 else 'kurang stabil')
                    if j >= .6:
                        st.success(f'Indeks Jaccard {rp(j)} — penjelasan **{vonis}**.')
                    else:
                        st.warning(f'Indeks Jaccard {rp(j)} — penjelasan **{vonis}**; tafsirkan dengan hati-hati.')

            # ---- tabel rincian faktor
            st.markdown('**Rincian faktor**')
            def posisi(r):
                if r['persentil'] is None or r['diisi']:
                    return '–'
                kata = 'lebih siang' if r['kolom'] == 'absensi_rata_jam_masuk' else 'lebih tinggi'
                return f"{kata} dari {r['persentil']:.0f}% data latih"
            tb = pd.DataFrame([{'Arah': '▲ menaikkan' if r['bobot'] > 0 else '▼ menurunkan',
                                'Faktor': r['faktor'], 'Nilai pegawai': r['nilai'],
                                'Kelompok nilai menurut LIME': r['kondisi'], 'Posisi': posisi(r),
                                'Pengaruh': f"{r['bobot']:+.3f}".replace('.', ','), 'Jenis faktor': r['jenis']}
                               for r in rinci])
            st.dataframe(tb, hide_index=True, **W)

            # ---- lintasan
            st.markdown('**Lintasan peluang per triwulan**')
            f = go.Figure()
            f.add_trace(go.Scatter(x=rows.tw, y=100 * rows.peluang_sb, mode='lines', line=dict(color=NAVY, width=2),
                                   showlegend=False, hoverinfo='skip'))
            for lab, warna, simbol in [('Sangat Baik', EMAS, 'star'), ('Bukan SB', ABU, 'circle'),
                                       ('–', NAVY, 'circle-open')]:
                s = rows[rows.aktual == lab]
                if len(s):
                    f.add_trace(go.Scatter(
                        x=s.tw, y=100 * s.peluang_sb, mode='markers',
                        name={'–': 'Belum berlabel'}.get(lab, f'Aktual: {lab}'),
                        marker=dict(size=13 if lab == 'Sangat Baik' else 10, color=warna, symbol=simbol,
                                    line=dict(color=NAVY, width=1.5)),
                        customdata=np.stack([s.predikat, s.fitur_terisi], axis=1),
                        hovertemplate='%{x}<br>peluang %{y:.0f}%<br>predikat %{customdata[0]}'
                                      '<br>indikator terisi %{customdata[1]}<extra></extra>'))
            f.add_vline(x=tw_e, line_color=EMAS, line_width=2, opacity=.5)
            f.add_hline(y=50, line_dash='dash', line_color=MERAH, annotation_text='ambang 50%',
                        annotation_position='top left')
            f.update_layout(height=300, template='plotly_white', yaxis_range=[-3, 103], yaxis_title='Peluang (%)',
                            legend=dict(orientation='h', y=1.15, x=0), margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(f, **W)
            if (rows.n_terisi < len(paket['num']) / 2).any():
                st.caption('Triwulan dengan kurang dari separuh indikator terisi (misalnya sebelum absensi tersedia) '
                           'sebagian besar diisi median, sehingga peluangnya kurang bermakna.')

            with st.expander('Tabel per triwulan dan perbandingan dengan kelompok predikat'):
                tb = rows[['tw', 'peluang_sb', 'predikat', 'fitur_terisi', 'sumber_peluang']].copy()
                tb['peluang_sb'] = (100 * tb.peluang_sb).round(0).astype(int).astype(str) + '%'
                st.dataframe(tb.rename(columns={'tw': 'Triwulan', 'peluang_sb': 'Peluang', 'predikat': 'Predikat',
                                                'fitur_terisi': 'Indikator', 'sumber_peluang': 'Sumber'}),
                             hide_index=True, **W)
                st.caption('Sumber “validasi”: peluang luar-lipatan (baris itu tidak ikut melatih model). '
                           'Sumber “akhir”: model yang dilatih pada seluruh data latih.')
                dl = L['dl']; bnd = []
                for k_ in paket['num']:
                    bnd.append({'Indikator': pm.LABEL.get(k_, k_), 'Pegawai ini': pm.format_nilai(k_, baris[k_]),
                                'Median Sangat Baik': pm.format_nilai(k_, dl.loc[dl._y == 1, k_].median()),
                                'Median bukan SB': pm.format_nilai(k_, dl.loc[dl._y == 0, k_].median())})
                for k_ in paket['kat']:
                    bnd.append({'Indikator': pm.LABEL.get(k_, k_), 'Pegawai ini': str(baris[k_]),
                                'Median Sangat Baik': dl.loc[dl._y == 1, k_].mode().iat[0],
                                'Median bukan SB': dl.loc[dl._y == 0, k_].mode().iat[0]})
                st.dataframe(pd.DataFrame(bnd), hide_index=True, **W)
                st.caption('Untuk indikator kategorik ditampilkan kategori terbanyak.')

            with st.expander('Cara membaca penjelasan ini'):
                st.markdown(
                    '- **Peluang Sangat Baik** adalah keyakinan model; di atas 50% ditebak Sangat Baik.\n'
                    '- **Batang hijau** menaikkan peluang, **batang merah** menurunkannya. Makin panjang, makin besar '
                    'pengaruhnya pada triwulan ini.\n'
                    '- **Kelompok nilai menurut LIME** adalah rentang tempat nilai pegawai berada; LIME menilai '
                    'pengaruh rentang itu, bukan angka persisnya.\n'
                    '- **Jenis faktor**: *pengembangan kompetensi* dapat diintervensi lewat pelatihan; *kedisiplinan '
                    'dan presensi* perlu dibaca bersama mutu pencatatan; *atribut tetap* tidak dapat diubah.\n'
                    '- Penjelasan bersifat lokal: berlaku untuk pegawai dan triwulan ini saja, dan bersifat asosiatif, '
                    'bukan sebab-akibat.')
            unduh = pd.DataFrame([{'kode': kode, 'triwulan': tw_e, 'model': nm_e, 'peluang_sb': round(p_e, 4),
                                   'predikat_aktual': baris['predikat'], 'faktor': r['faktor'],
                                   'nilai_pegawai': r['nilai'], 'kelompok_nilai': r['kondisi'],
                                   'pengaruh': round(r['bobot'], 4), 'jenis_faktor': r['jenis']} for r in rinci])
            st.download_button('Unduh penjelasan (CSV)', unduh.to_csv(index=False).encode('utf-8'),
                               f'penjelasan_lime_{kode}_{tw_e}.csv', 'text/csv')

    # ------------------------------------------------------------ hasil tersimpan
    st.divider()
    with st.expander('Hasil interpretasi tersimpan dari penelitian (Bab IV)', expanded=False):
        if HI is None or HD is None:
            st.info('Berkas hasil interpretasi tidak ditemukan.')
        else:
            STRUKTURAL = ('pendidikan', 'kelompok', 'status', 'Usia')
            c1, c2 = st.columns(2)
            for kol, dat, jud in [(c1, HI['penting'][:10], 'Skenario A — seluruh variabel'),
                                  (c2, HD['penting'][:10], 'Skenario D — perilaku saja')]:
                with kol:
                    k = [a for a, _ in dat][::-1]; v = [100 * b for _, b in dat][::-1]
                    warna = [MERAH if any(s in a for s in STRUKTURAL) else NAVY for a in k]
                    f = go.Figure(go.Bar(x=v, y=k, orientation='h', marker_color=warna,
                                         text=[f'{x:.1f}' for x in v], textposition='outside'))
                    f.update_layout(height=340, template='plotly_white', title=jud, xaxis_title='Kontribusi (%)',
                                    margin=dict(l=10, r=40, t=40, b=10))
                    st.plotly_chart(f, **W)
            st.caption('Batang merah menandai atribut struktural. Ketiganya menyumbang 33,0 persen kepentingan fitur '
                       'pada skenario A, sementara persentase kehadiran apel hanya 7,1 persen.')
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('**Odds ratio — Regresi Logistik (skenario D)**')
                od = pd.DataFrame(HD['odds'], columns=['Fitur', 'Odds ratio'])
                f = go.Figure(go.Bar(x=od['Odds ratio'], y=od.Fitur, orientation='h',
                                     marker_color=[HIJAU if x > 1 else MERAH for x in od['Odds ratio']],
                                     text=[f'{x:.3f}' for x in od['Odds ratio']], textposition='outside'))
                f.add_vline(x=1, line_dash='dash', line_color='#333')
                f.update_layout(height=360, template='plotly_white', margin=dict(l=10, r=40, t=10, b=10),
                                yaxis=dict(autorange='reversed'))
                st.plotly_chart(f, **W)
            with c2:
                st.markdown('**Penjelasan LIME empat kelompok kasus (skenario D)**')
                kasus = st.selectbox('Kelompok kasus', list(HD['lime']))
                info = HD['lime'][kasus]
                m = st.columns(3)
                m[0].metric('Peluang', rp(info['prob']))
                m[1].metric('Aktual', 'Sangat Baik' if info['aktual'] else 'Bukan SB')
                m[2].metric('Prediksi', 'Sangat Baik' if info['prediksi'] else 'Bukan SB')
                fa = info['faktor'][::-1]
                f = go.Figure(go.Bar(x=[w for _, w in fa], y=[t for t, _ in fa], orientation='h',
                                     marker_color=[HIJAU if w > 0 else MERAH for _, w in fa],
                                     text=[f'{w:+.3f}' for _, w in fa], textposition='outside'))
                f.add_vline(x=0, line_color='#333')
                f.update_layout(height=270, template='plotly_white', xaxis_title='Bobot LIME terhadap kelas Sangat Baik',
                                margin=dict(l=10, r=40, t=10, b=10))
                st.plotly_chart(f, **W)
                st.caption(f"Ambang pada penelitian dalam skala terstandardisasi. Kestabilan penjelasan (indeks Jaccard): "
                           f"{rp(HD['jaccard'][0])} ± {rp(HD['jaccard'][1])}")

# ====================================================== 7. PRIORITAS AHP-TOPSIS
with T[6]:
    if HP is None:
        st.info('Berkas `hasil_prioritas.json` tidak ditemukan, sehingga bagian ini kosong.')
    else:
        KRIT = HP['ahp']['kriteria']
        st.warning('Bobot bawaan berasal dari matriks perbandingan **sementara** yang disusun dari urutan kepentingan '
                   'normatif. Gantilah dengan hasil kuesioner pakar melalui panel di bawah sebelum peringkat dipakai '
                   'sebagai dasar keputusan.')
        PASANG = [(i, j) for i in range(5) for j in range(i + 1, 5)]
        with st.expander('Isi perbandingan berpasangan AHP', expanded=False):
            st.caption('Skala Saaty: 1 sama penting · 3 sedikit lebih penting · 5 lebih penting · 7 sangat lebih '
                       'penting · 9 mutlak lebih penting. Nilai negatif berarti kriteria kedua yang lebih penting.')
            awal = np.array(HP['ahp']['matriks'], float)
            nilai = {}
            kol = st.columns(2)
            for n, (i, j) in enumerate(PASANG):
                a = awal[i, j]
                bawaan = int(round(a)) if a >= 1 else -int(round(1 / a))
                nilai[(i, j)] = kol[n % 2].slider(
                    f'{KRIT[i].split(" ")[0]} vs {KRIT[j].split(" ")[0]}  — '
                    f'{KRIT[i].split(" ", 1)[1]} / {KRIT[j].split(" ", 1)[1]}', -9, 9, bawaan, 1, key=f'p{i}{j}')
        M = np.eye(5)
        for (i, j), v in nilai.items():
            x = float(v) if v >= 1 else (1 / abs(v) if v <= -1 else 1.0)
            M[i, j] = x; M[j, i] = 1 / x
        val, vec = np.linalg.eig(M)
        k = int(np.argmax(val.real)); lmaks = float(val.real[k])
        w = np.abs(vec[:, k].real); w = w / w.sum()
        CI = (lmaks - 5) / 4; RI = 1.12; CR = CI / RI
        c = st.columns([1, 1, 1, 3])
        c[0].metric('λ maks', f'{lmaks:.4f}')
        c[1].metric('CI', f'{CI:.4f}')
        c[2].metric('CR', f'{CR:.4f}', 'konsisten' if CR <= .1 else 'TIDAK konsisten',
                    delta_color='normal' if CR <= .1 else 'inverse')
        with c[3]:
            f = go.Figure(go.Bar(x=[kk.split(' ', 1)[1] for kk in KRIT], y=100 * w,
                                 marker_color=[NAVY if x >= .2 else ABU for x in w],
                                 text=[f'{100 * x:.1f}%' for x in w], textposition='outside'))
            f.update_layout(height=200, template='plotly_white', yaxis_title='Bobot (%)',
                            margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(f, **W)
        if CR > .1:
            st.error('CR melebihi 0,1. Penilaian belum konsisten — perbaiki perbandingan berpasangan sebelum hasil '
                     'peringkat ditafsirkan.')
        if os.path.exists('peringkat_prioritas_topsis.csv'):
            alt = pd.read_csv('peringkat_prioritas_topsis.csv')
            X = alt[['K1', 'K2', 'K3', 'K4', 'K5']].astype(float).values
            R = X / np.sqrt((X ** 2).sum(axis=0)); V = R * w
            Dp = np.sqrt(((V - V.max(0)) ** 2).sum(1)); Dn = np.sqrt(((V - V.min(0)) ** 2).sum(1))
            alt = alt.assign(C=(Dn / (Dp + Dn)).round(4)).sort_values('C', ascending=False).reset_index(drop=True)
            alt['peringkat'] = alt.index + 1
            st.subheader(f'Peringkat prioritas pengembangan ({len(alt)} pegawai)')
            c1, c2 = st.columns([2, 3])
            with c1:
                n = st.slider('Tampilkan berapa besar', 5, min(50, len(alt)), 10)
                st.dataframe(alt.head(n)[['peringkat', 'kode_pegawai', 'jabatan', 'K1', 'K2', 'K3', 'K4', 'K5', 'C']],
                             hide_index=True, height=min(420, 40 + 28 * n), **W)
                st.download_button('Unduh peringkat lengkap (CSV)', alt.to_csv(index=False).encode('utf-8'),
                                   'peringkat_prioritas.csv', 'text/csv')
            with c2:
                f = go.Figure(go.Bar(x=alt.peringkat, y=alt.C,
                                     marker_color=[EMAS if i < n else NAVY for i in range(len(alt))],
                                     hovertext=alt.kode_pegawai))
                f.add_vline(x=n + .5, line_dash='dash', line_color=MERAH)
                f.update_layout(height=300, template='plotly_white', xaxis_title='Peringkat prioritas',
                                yaxis_title='Nilai kedekatan C', margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(f, **W)
                st.markdown('**Analisis sensitivitas** — berapa dari sepuluh besar yang bertahan ketika bobot satu '
                            'kriteria diubah')
                baris = []
                atas = set(alt.kode_pegawai.head(10))
                for j, kk in enumerate(KRIT):
                    r = {'Kriteria': kk}
                    for fk in [0.5, 0.75, 1.25, 1.5]:
                        w2 = w.copy(); w2[j] *= fk; w2 = w2 / w2.sum()
                        V2 = R * w2
                        c2v = (np.sqrt(((V2 - V2.min(0)) ** 2).sum(1)) /
                               (np.sqrt(((V2 - V2.max(0)) ** 2).sum(1)) + np.sqrt(((V2 - V2.min(0)) ** 2).sum(1))))
                        r[f'×{fk}'] = len(atas & set(alt.kode_pegawai.values[np.argsort(-c2v)][:10]))
                    baris.append(r)
                sv = pd.DataFrame(baris)
                st.dataframe(sv, hide_index=True, **W)
                rata = sv[[c for c in sv.columns if c != 'Kriteria']].values.mean()
                st.caption(f'Rerata bertahan: **{rata:.1f} dari 10**. Semakin mendekati 10, semakin tegar peringkat '
                           'terhadap ketidakpastian bobot.')
        else:
            st.info('Berkas `peringkat_prioritas_topsis.csv` tidak ditemukan, sehingga pemeringkatan tidak dapat '
                    'dihitung ulang.')

st.divider()
st.caption('Keluaran dashboard ini bersifat **rekomendatif** dan tidak menggantikan kewenangan pejabat penilai maupun '
           'pejabat pembina kepegawaian. Berkas yang diunggah hanya diproses di memori selama sesi berjalan dan tidak '
           'disimpan oleh aplikasi.')
