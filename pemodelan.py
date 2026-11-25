"""
Modul pemodelan untuk dashboard penelitian.

Mereplikasi alur Bab III laporan:
  - klasifikasi biner Sangat Baik terhadap selain Sangat Baik
  - imputasi median (numerik) dan kategori tersendiri (kategorik), one-hot dengan satu kategori acuan
  - standardisasi dan SMOTE hanya pada lipatan latih
  - stratified group k-fold (k = 5) per pegawai, diulang 1-3 kali
  - uji t berpasangan terkoreksi Nadeau-Bengio
  - LIME pada tingkat individu (5.000 sampel tiruan, bibit acak tetap)
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from imblearn.over_sampling import SMOTE
    ADA_SMOTE = True
except ImportError:            # pragma: no cover
    ADA_SMOTE = False
try:
    from xgboost import XGBClassifier
    ADA_XGB = True
except ImportError:            # pragma: no cover
    ADA_XGB = False
try:
    from lime.lime_tabular import LimeTabularExplainer
    ADA_LIME = True
except ImportError:            # pragma: no cover
    ADA_LIME = False

# ------------------------------------------------------------------ himpunan fitur
FITUR_PERILAKU = ['pelatihan_kumulatif', 'masa_kerja_tahun', 'apel_alpa', 'absensi_rata_jam_masuk',
                  'apel_persen_hadir', 'absensi_tanpa_checkout', 'absensi_rata_durasi', 'apel_dinas_luar']
FITUR_STRUKTURAL_NUM = ['usia']
FITUR_STRUKTURAL_KAT = ['pendidikan', 'kelompok_pegawai', 'status_pegawai']
FITUR_ASESMEN = ['asesmen_integritas', 'asesmen_kerjasama', 'asesmen_komunikasi', 'asesmen_orientasi_pada_hasil',
                 'asesmen_pelayanan_publik', 'asesmen_pengembangan_diri_dan_orang_lain',
                 'asesmen_mengelola_perubahan', 'asesmen_pengambilan_keputusan', 'asesmen_perekat_bangsa']

LABEL = {
    'pelatihan_kumulatif': 'Akumulasi pelatihan', 'masa_kerja_tahun': 'Masa kerja (tahun)',
    'apel_alpa': 'Apel tanpa keterangan', 'absensi_rata_jam_masuk': 'Rerata jam masuk',
    'apel_persen_hadir': 'Persentase kehadiran apel', 'absensi_tanpa_checkout': 'Tidak absen pulang',
    'absensi_rata_durasi': 'Rerata durasi kerja', 'apel_dinas_luar': 'Apel saat dinas luar',
    'usia': 'Usia', 'pendidikan': 'Pendidikan', 'kelompok_pegawai': 'Kelompok pegawai',
    'status_pegawai': 'Status pegawai',
    'asesmen_integritas': 'Asesmen integritas', 'asesmen_kerjasama': 'Asesmen kerja sama',
    'asesmen_komunikasi': 'Asesmen komunikasi', 'asesmen_orientasi_pada_hasil': 'Asesmen orientasi hasil',
    'asesmen_pelayanan_publik': 'Asesmen pelayanan publik',
    'asesmen_pengembangan_diri_dan_orang_lain': 'Asesmen pengembangan diri',
    'asesmen_mengelola_perubahan': 'Asesmen mengelola perubahan',
    'asesmen_pengambilan_keputusan': 'Asesmen pengambilan keputusan',
    'asesmen_perekat_bangsa': 'Asesmen perekat bangsa',
}

SKENARIO = {
    'A': dict(nama='A — perilaku + atribut struktural', num=FITUR_PERILAKU + FITUR_STRUKTURAL_NUM,
              kat=FITUR_STRUKTURAL_KAT, asesmen=False),
    'C': dict(nama='C — A + sembilan indikator asesmen', num=FITUR_PERILAKU + FITUR_STRUKTURAL_NUM + FITUR_ASESMEN,
              kat=FITUR_STRUKTURAL_KAT, asesmen=True),
    'D': dict(nama='D — perilaku saja (tanpa atribut struktural)', num=FITUR_PERILAKU, kat=[], asesmen=False),
}

MODEL_TERSEDIA = ['Regresi Logistik', 'Random Forest'] + (['XGBoost'] if ADA_XGB else [])
KOLOM_WAJIB = ['tw', 'skp_predikat']


def buat_model(nama, seed=0):
    if nama == 'Regresi Logistik':
        return LogisticRegression(C=1.0, max_iter=3000)  # penalti L2 (bawaan)
    if nama == 'Random Forest':
        return RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=3,
                                      class_weight='balanced', random_state=seed, n_jobs=-1)
    if nama == 'XGBoost':
        return XGBClassifier(n_estimators=300, learning_rate=0.08, max_depth=4, subsample=0.9,
                             reg_lambda=2.0, eval_metric='logloss', random_state=seed, n_jobs=-1, verbosity=0)
    raise ValueError(nama)


def buat_praproses(num, kat):
    bagian = [('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('std', StandardScaler())]), num)]
    if kat:
        bagian.append(('kat', Pipeline([
            ('imp', SimpleImputer(strategy='constant', fill_value='Tidak diketahui')),
            ('oh', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))]), kat))
    return ColumnTransformer(bagian, remainder='drop', verbose_feature_names_out=False)


# ------------------------------------------------------------------ penyiapan data
def normalkan(df):
    """Seragamkan kolom berkas yang diunggah: kunci pegawai, kolom tw, kategori kosong."""
    d = df.copy()
    d.columns = [c.strip() for c in d.columns]
    if 'kode_pegawai' in d.columns:
        d['kode'] = d['kode_pegawai'].astype(str)
    elif 'nip' in d.columns:
        d['kode'] = d['nip'].astype(str)
    else:
        d['kode'] = [f'BARIS{i + 1:04d}' for i in range(len(d))]
    if 'tw' not in d.columns and {'tahun', 'triwulan'} <= set(d.columns):
        d['tw'] = d['tahun'].astype(int).astype(str) + '-TW' + d['triwulan'].astype(int).astype(str)
    if 'tw' in d.columns:
        d['tw'] = d['tw'].astype(str)
    for c in FITUR_STRUKTURAL_KAT + ['jabatan']:
        if c in d.columns:
            d[c] = d[c].fillna('Tidak diketahui').astype(str)
    for c in FITUR_PERILAKU + FITUR_STRUKTURAL_NUM + FITUR_ASESMEN + ['asesmen_tersedia', 'skp_bobot', 'tahun']:
        if c in d.columns and not pd.api.types.is_numeric_dtype(d[c]):
            # CSV ekspor Excel berbahasa Indonesia memakai koma sebagai pemisah desimal
            d[c] = pd.to_numeric(d[c].astype(str).str.strip().str.replace(',', '.', regex=False), errors='coerce')
    if 'nama' in d.columns:
        d['nama'] = d['nama'].fillna('').astype(str).str.strip()
    return d


def kolom_kurang(df, skenario):
    sk = SKENARIO[skenario]
    perlu = sk['num'] + sk['kat']
    return [c for c in perlu if c not in df.columns]


def label_biner(df):
    if 'skp_predikat' in df.columns:
        p = df['skp_predikat']
        return np.where(p.isna(), np.nan, (p.astype(str).str.strip() == 'Sangat Baik').astype(float))
    if 'skp_sangat_baik' in df.columns:
        return pd.to_numeric(df['skp_sangat_baik'], errors='coerce').values
    return np.full(len(df), np.nan)


def apakah_replikasi(df):
    if 'skp_jenis_penilaian' in df.columns:
        return df['skp_jenis_penilaian'].astype(str).str.contains('replikasi', case=False, na=False).values
    if 'skp_bobot' in df.columns:
        return (pd.to_numeric(df['skp_bobot'], errors='coerce') < 1).values
    return np.zeros(len(df), bool)


def siapkan_latih(df, skenario, tahun_replikasi=()):
    """Baris berlabel untuk pelatihan: label triwulanan asli (+ replikasi tahun terpilih, bobot 0,25)."""
    y = label_biner(df)
    rep = apakah_replikasi(df)
    ada = ~np.isnan(y)
    pakai = ada & ~rep
    if len(tahun_replikasi) and 'tahun' in df.columns:
        pakai |= ada & rep & df['tahun'].isin(list(tahun_replikasi)).values
    if SKENARIO[skenario]['asesmen']:
        if 'asesmen_tersedia' in df.columns:
            pakai &= (pd.to_numeric(df['asesmen_tersedia'], errors='coerce') == 1).values
        else:
            pakai &= df[FITUR_ASESMEN].notna().any(axis=1).values
    d = df[pakai].copy()
    d['_y'] = y[pakai].astype(int)
    d['_w'] = np.where(rep[pakai], 0.25, 1.0)
    return d.reset_index(drop=False).rename(columns={'index': '_baris'})


# ------------------------------------------------------------------ pelatihan
def _latih_satu(nama, prep, X, y, w, seed, pakai_smote):
    Xt = prep.fit_transform(X)
    wt = np.asarray(w, float)
    yt = np.asarray(y)
    if pakai_smote and ADA_SMOTE and yt.min() != yt.max() and min(np.bincount(yt)) > 5:
        n0 = len(yt)
        Xt, yt = SMOTE(random_state=seed).fit_resample(Xt, yt)
        wt = np.concatenate([wt, np.ones(len(yt) - n0)])   # sampel sintetis berbobot 1
    m = buat_model(nama, seed)
    m.fit(Xt, yt, sample_weight=wt)
    return m


def _metrik(y, p, prob):
    return dict(akurasi=accuracy_score(y, p), presisi=precision_score(y, p, zero_division=0),
                recall=recall_score(y, p, zero_division=0), f1=f1_score(y, p, zero_division=0),
                auc=roc_auc_score(y, prob) if len(set(y)) > 1 else np.nan, mcc=matthews_corrcoef(y, p))


def validasi(dl, skenario, model_list, ulang=3, k=5, pakai_smote=True, progres=None):
    """Stratified group k-fold per pegawai. Mengembalikan metrik per lipatan dan prediksi luar-lipatan."""
    sk = SKENARIO[skenario]
    X = dl[sk['num'] + sk['kat']]
    y = dl['_y'].values
    w = dl['_w'].values
    g = dl['kode'].values
    hasil = {m: [] for m in model_list}
    oof = {m: np.full(len(dl), np.nan) for m in model_list}
    total = ulang * k * len(model_list); n = 0
    for r in range(ulang):
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=r)
        for latih, uji in cv.split(X, y, g):
            for nm in model_list:
                prep = buat_praproses(sk['num'], sk['kat'])
                m = _latih_satu(nm, prep, X.iloc[latih], y[latih], w[latih], r, pakai_smote)
                prob = m.predict_proba(prep.transform(X.iloc[uji]))[:, 1]
                pred = (prob >= 0.5).astype(int)
                hasil[nm].append(_metrik(y[uji], pred, prob))
                if r == 0:
                    oof[nm][uji] = prob
                n += 1
                if progres:
                    progres(n / total)
    ringkas = {}
    for nm in model_list:
        t = pd.DataFrame(hasil[nm])
        ringkas[nm] = dict(rerata=t.mean().to_dict(), sb=t.std(ddof=1).to_dict(), lipatan=t)
    return ringkas, oof


def uji_nadeau_bengio(a, b, k=5):
    """Uji t berpasangan terkoreksi (Nadeau & Bengio, 2003) atas penaksiran per lipatan."""
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    if n < 2 or np.var(d, ddof=1) == 0:
        return np.nan, np.nan
    s2 = np.var(d, ddof=1)
    t = d.mean() / np.sqrt((1 / n + 1 / (k - 1)) * s2)
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    return float(t), float(p)


def latih_akhir(dl, skenario, nama, pakai_smote=True, seed=0):
    """Model akhir dilatih pada seluruh data latih; dipakai untuk data olah dan LIME."""
    sk = SKENARIO[skenario]
    prep = buat_praproses(sk['num'], sk['kat'])
    X = dl[sk['num'] + sk['kat']]
    m = _latih_satu(nama, prep, X, dl['_y'].values, dl['_w'].values, seed, pakai_smote)
    return dict(prep=prep, model=m, num=sk['num'], kat=sk['kat'], nama=nama, skenario=skenario)


def peluang(paket, df):
    X = df[paket['num'] + paket['kat']]
    return paket['model'].predict_proba(paket['prep'].transform(X))[:, 1]


def kepentingan(paket):
    """Kepentingan fitur (RF/XGB) atau odds ratio per simpangan baku (Regresi Logistik)."""
    nama_f = list(paket['prep'].get_feature_names_out())
    pretty = []
    for f in nama_f:
        dasar = next((c for c in paket['kat'] if f.startswith(c + '_')), None)
        pretty.append(f"{LABEL.get(dasar, dasar)}: {f[len(dasar) + 1:]}" if dasar else LABEL.get(f, f))
    m = paket['model']
    if hasattr(m, 'feature_importances_'):
        v = m.feature_importances_
        return pd.DataFrame({'Fitur': pretty, 'Nilai': v}).sort_values('Nilai', ascending=False), 'kepentingan'
    v = np.exp(m.coef_[0])
    return pd.DataFrame({'Fitur': pretty, 'Nilai': v}).sort_values('Nilai', ascending=False), 'odds'


# ------------------------------------------------------------------ LIME
def siapkan_lime(paket, dl, seed=0):
    """Penjelas LIME pada ruang fitur asli (kategori dikodekan bilangan bulat agar terbaca)."""
    num, kat = paket['num'], paket['kat']
    latar = dl[num + kat].copy()
    median = latar[num].median()
    latar[num] = latar[num].fillna(median)
    kategori = {}
    for c in kat:
        latar[c] = latar[c].fillna('Tidak diketahui').astype(str)
        kategori[c] = sorted(latar[c].unique())
        latar[c] = latar[c].map({v: i for i, v in enumerate(kategori[c])})
    kolom = num + kat
    eksp = LimeTabularExplainer(
        latar.values.astype(float), feature_names=[LABEL.get(c, c) for c in kolom],
        class_names=['Bukan Sangat Baik', 'Sangat Baik'], mode='classification',
        categorical_features=[kolom.index(c) for c in kat],
        categorical_names={kolom.index(c): kategori[c] for c in kat},
        discretize_continuous=True, random_state=seed)

    def fungsi(arr):
        X = pd.DataFrame(arr, columns=kolom)
        for c in kat:
            X[c] = [kategori[c][int(round(v))] if 0 <= int(round(v)) < len(kategori[c]) else 'Tidak diketahui'
                    for v in X[c]]
        return paket['model'].predict_proba(paket['prep'].transform(X))

    return dict(eksp=eksp, fungsi=fungsi, median=median, kategori=kategori, kolom=kolom)


def jelaskan(lime_pk, baris, n_faktor=8, n_sampel=5000):
    num = [c for c in lime_pk['kolom'] if c not in lime_pk['kategori']]
    x = baris[lime_pk['kolom']].copy()
    diisi = [LABEL.get(c, c) for c in num if pd.isna(x[c])]
    for c in num:
        if pd.isna(x[c]):
            x[c] = lime_pk['median'][c]
    for c, daftar in lime_pk['kategori'].items():
        v = str(x[c]) if pd.notna(x[c]) else 'Tidak diketahui'
        x[c] = daftar.index(v) if v in daftar else 0
    e = lime_pk['eksp'].explain_instance(x.values.astype(float), lime_pk['fungsi'], num_features=n_faktor,
                                         num_samples=n_sampel, labels=(1,))
    return e.as_list(label=1), diisi


# ------------------------------------------------------------------ pembagian 80:20
def bagi_latih_uji(df, porsi_uji=0.2, seed=42):
    """Membagi dataset per pegawai (semua baris seorang pegawai berada di satu sisi), berstrata menurut
    kelompok pegawai dan banyaknya predikat Sangat Baik pada label triwulanan asli.
    Mengembalikan (data_latih, data_uji, ringkasan) dengan kolom asli berkas."""
    from sklearn.model_selection import train_test_split
    d = normalkan(df)
    y = label_biner(d)
    asli = ~np.isnan(y) & ~apakah_replikasi(d)
    per = pd.DataFrame({'kode': d['kode'], 'sb': np.where(asli, y, 0)}).groupby('kode').sb.sum()
    kel = d.groupby('kode')['kelompok_pegawai'].first() if 'kelompok_pegawai' in d else pd.Series('-', per.index)
    strata = kel.reindex(per.index).astype(str) + '|' + pd.cut(per, [-1, 0, 1, 99], labels=['0', '1', '2+']).astype(str)
    kecil = strata.map(strata.value_counts()) < 2
    strata[kecil] = 'lain'
    kode_latih, kode_uji = train_test_split(np.asarray(per.index, dtype=object), test_size=porsi_uji, random_state=seed,
                                            stratify=np.asarray(strata, dtype=object) if strata.value_counts().min() >= 2 else None)
    m_uji = d['kode'].isin(set(kode_uji)).values
    latih, uji = df[~m_uji].copy(), df[m_uji].copy()

    def info(mask):
        yy = y[mask & asli]
        return dict(baris=int(mask.sum()), pegawai=int(d.loc[mask, 'kode'].nunique()),
                    berlabel=int(len(yy)), sangat_baik=int(np.nansum(yy)),
                    persen_sb=float(100 * np.nanmean(yy)) if len(yy) else float('nan'))
    return latih, uji, dict(latih=info(~m_uji), uji=info(m_uji), seed=seed, porsi_uji=porsi_uji)


# ------------------------------------------------------------------ bahasa sederhana untuk LIME
JENIS_FAKTOR = {
    'pelatihan_kumulatif': 'Pengembangan kompetensi',
    'masa_kerja_tahun': 'Atribut tetap', 'usia': 'Atribut tetap', 'pendidikan': 'Atribut tetap',
    'kelompok_pegawai': 'Atribut tetap', 'status_pegawai': 'Atribut tetap',
}
for _c in FITUR_ASESMEN:
    JENIS_FAKTOR[_c] = 'Pengembangan kompetensi'
for _c in ['apel_alpa', 'absensi_rata_jam_masuk', 'apel_persen_hadir', 'absensi_tanpa_checkout',
           'absensi_rata_durasi', 'apel_dinas_luar']:
    JENIS_FAKTOR[_c] = 'Kedisiplinan dan presensi'


def format_nilai(kolom, v):
    """Menampilkan nilai indikator dalam satuan yang mudah dibaca."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'tidak tercatat'
    if kolom not in FITUR_PERILAKU + FITUR_STRUKTURAL_NUM + FITUR_ASESMEN:
        return str(v)
    v = float(v)
    if kolom == 'absensi_rata_jam_masuk':
        j = int(v); m = int(round((v - j) * 60))
        if m == 60:
            j, m = j + 1, 0
        return f'pukul {j:02d}.{m:02d}'
    if kolom == 'absensi_rata_durasi':
        j = int(v); m = int(round((v - j) * 60))
        return f'{j} jam {m} menit'
    if kolom == 'apel_persen_hadir':
        return f'{v:.1f}%'.replace('.', ',')
    if kolom in ('masa_kerja_tahun', 'usia'):
        return f'{v:.0f} tahun'
    if kolom == 'pelatihan_kumulatif':
        return f'{v:.0f} kegiatan'
    if kolom in ('apel_alpa', 'apel_dinas_luar'):
        return f'{v:.0f} kali'
    if kolom == 'absensi_tanpa_checkout':
        return f'{v:.0f} hari'
    return f'{v:.1f}'.replace('.', ',').replace(',0', '')


def kalimat_aturan(kolom, aturan):
    """Mengubah aturan LIME (mis. '1.00 < Akumulasi pelatihan <= 4.00') menjadi frasa sehari-hari."""
    import re
    if kolom not in FITUR_PERILAKU + FITUR_STRUKTURAL_NUM + FITUR_ASESMEN:
        return 'kategori ' + aturan.split('=', 1)[-1] if '=' in aturan else aturan
    SATUAN = {'pelatihan_kumulatif': 'kegiatan', 'apel_alpa': 'kali', 'apel_dinas_luar': 'kali',
              'absensi_tanpa_checkout': 'hari', 'masa_kerja_tahun': 'tahun', 'usia': 'tahun'}
    f = lambda v: format_nilai(kolom, float(v))
    jam = lambda v: f(v).replace('pukul ', '')
    m = re.match(r'^\s*(-?[\d.]+)\s*<\s*.+?\s*<=\s*(-?[\d.]+)\s*$', aturan)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if kolom in SATUAN:                       # bilangan bulat: (a, b] = a+1 .. b
            lo, hi = int(np.floor(a)) + 1, int(np.floor(b))
            return f'{hi} {SATUAN[kolom]}' if lo >= hi else f'{lo}–{hi} {SATUAN[kolom]}'
        if kolom == 'absensi_rata_jam_masuk':
            return f'pukul {jam(a)}–{jam(b)}'
        if kolom == 'apel_persen_hadir':
            return f'{f(a)[:-1]}–{f(b)}'
        return f'{f(a)} sampai {f(b)}'
    m = re.match(r'^.+?\s*<=\s*(-?[\d.]+)\s*$', aturan)
    if m:
        b = float(m.group(1))
        if kolom in SATUAN and b < 1:
            return f'0 {SATUAN[kolom]} (tidak ada)'
        if kolom == 'absensi_rata_jam_masuk':
            return f'pukul {jam(b)} atau lebih pagi'
        return f'{f(b)} atau kurang'
    m = re.match(r'^.+?\s*>\s*(-?[\d.]+)\s*$', aturan)
    if m:
        a = float(m.group(1))
        if kolom in SATUAN:
            return f'{int(np.floor(a)) + 1} {SATUAN[kolom]} atau lebih'
        if kolom == 'absensi_rata_jam_masuk':
            return f'lebih siang dari pukul {jam(a)}'
        return f'lebih dari {f(a)}'
    return aturan


def jelaskan_rinci(lime_pk, baris, dl, n_faktor=8, n_sampel=5000):
    """Penjelasan LIME dalam tabel yang terbaca: nilai pegawai, kondisi, arah, besar pengaruh, jenis faktor."""
    num = [c for c in lime_pk['kolom'] if c not in lime_pk['kategori']]
    x = baris[lime_pk['kolom']].copy()
    diisi = [c for c in num if pd.isna(x[c])]
    asli = {c: baris[c] for c in lime_pk['kolom']}
    for c in num:
        if pd.isna(x[c]):
            x[c] = lime_pk['median'][c]
    for c, daftar in lime_pk['kategori'].items():
        v = str(x[c]) if pd.notna(x[c]) else 'Tidak diketahui'
        x[c] = daftar.index(v) if v in daftar else 0
    e = lime_pk['eksp'].explain_instance(x.values.astype(float), lime_pk['fungsi'], num_features=n_faktor,
                                         num_samples=n_sampel, labels=(1,))
    teks = dict(e.as_list(label=1))
    baris_out = []
    for idx, w in e.as_map()[1]:
        kol = lime_pk['kolom'][idx]
        aturan = next((t for t in teks if LABEL.get(kol, kol) in t), '')
        nilai = asli[kol]
        pct = None
        if kol in num and pd.notna(nilai) and kol in dl:
            ref = dl[kol].dropna()
            pct = float(100 * (ref < float(nilai)).mean()) if len(ref) else None
        baris_out.append(dict(
            kolom=kol, faktor=LABEL.get(kol, kol), bobot=float(w), aturan=aturan,
            nilai=format_nilai(kol, nilai) + (' (diisi median)' if kol in diisi else ''),
            kondisi=kalimat_aturan(kol, aturan), persentil=pct, diisi=kol in diisi,
            jenis=JENIS_FAKTOR.get(kol, 'Lainnya')))
    return baris_out, [LABEL.get(c, c) for c in diisi]


def kestabilan(paket, dl, baris, n_ulang=5, n_faktor=5):
    """Indeks Jaccard rerata antar-pasangan lima faktor teratas pada beberapa bibit acak (Subbab 3.6)."""
    himp = []
    for s in range(n_ulang):
        lp = siapkan_lime(paket, dl, seed=100 + s)
        f, _ = jelaskan(lp, baris, n_faktor)
        himp.append({t.split(' ')[0] if '=' in t else t for t, _ in f})
        himp[-1] = {next((LABEL.get(c, c) for c in lp['kolom'] if LABEL.get(c, c) in t), t) for t, _ in f}
    nilai = [len(a & b) / len(a | b) for i, a in enumerate(himp) for b in himp[i + 1:] if a | b]
    return float(np.mean(nilai)) if nilai else float('nan')
