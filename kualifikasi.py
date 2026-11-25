"""
Deteksi kesesuaian kualifikasi pendidikan pegawai.

1. Jenjang  : jenjang pendidikan pegawai dibandingkan dengan jenjang minimal jabatan
              (kolom pendidikan_formal pada tabel tugas_pegawai / analisis jabatan institusi).
2. Bidang   : bidang ilmu pegawai (diperkirakan dari gelar akademik pada nama, atau kolom bidang_pendidikan)
              dibandingkan dengan rumpun bidang yang relevan bagi unit kerjanya.

Acuan gelar dan rumpun unit disimpan sebagai CSV agar dapat diperiksa dan diubah oleh institusi.
"""
import os
import re

import numpy as np
import pandas as pd

JENJANG = {'SD': 1, 'SMP': 2, 'SLTP': 2, 'SLTA': 3, 'SMA': 3, 'SMK': 3, 'SMA/SMK': 3, 'D-1': 4, 'D-2': 4, 'D-3': 4,
           'D1': 4, 'D2': 4, 'D3': 4, 'D-4': 5, 'D4': 5, 'S-1': 5, 'S1': 5, 'S-2': 6, 'S2': 6, 'S-3': 7, 'S3': 7}
NAMA_JENJANG = {1: 'SD', 2: 'SMP', 3: 'SLTA', 4: 'Diploma (D-1 s.d. D-3)', 5: 'S-1/D-4', 6: 'S-2', 7: 'S-3'}
STATUS_JENJANG = {1: 'Di bawah syarat', 2: 'Sesuai syarat', 3: 'Di atas syarat'}

ACUAN_GELAR = [  # gelar (tanpa titik dan spasi, huruf besar) -> bidang
    ('SP', 'Pertanian'), ('MP', 'Pertanian'), ('IR', 'Pertanian'), ('SHUT', 'Kehutanan'), ('MHUT', 'Kehutanan'),
    ('SPI', 'Perikanan'), ('MPI', 'Perikanan'), ('SSTPI', 'Perikanan'), ('API', 'Perikanan'), ('SKEL', 'Perikanan'),
    ('SPT', 'Peternakan'), ('MPT', 'Peternakan'), ('DRH', 'Peternakan'), ('SKH', 'Peternakan'),
    ('STP', 'Teknologi Pertanian'), ('MTP', 'Teknologi Pertanian'), ('AMDTP', 'Teknologi Pertanian'),
    ('ST', 'Teknik'), ('MT', 'Teknik'), ('MENG', 'Teknik'), ('STRT', 'Teknik'), ('SST', 'Teknik'),
    ('SSI', 'Sains'), ('MIL', 'Sains'), ('SKOM', 'Komputer'), ('MKOM', 'Komputer'), ('AMDKOM', 'Komputer'),
    ('MSTAT', 'Statistik'), ('SSTAT', 'Statistik'),
    ('SE', 'Ekonomi dan Manajemen'), ('ME', 'Ekonomi dan Manajemen'), ('MM', 'Ekonomi dan Manajemen'),
    ('SM', 'Ekonomi dan Manajemen'), ('SAK', 'Ekonomi dan Manajemen'), ('MAK', 'Ekonomi dan Manajemen'),
    ('AMDAB', 'Ekonomi dan Manajemen'),
    ('SH', 'Hukum'), ('MH', 'Hukum'),
    ('SSOS', 'Sosial dan Administrasi'), ('SAP', 'Sosial dan Administrasi'), ('MAP', 'Sosial dan Administrasi'),
    ('SIP', 'Sosial dan Administrasi'), ('SIKOM', 'Sosial dan Administrasi'), ('SPSI', 'Sosial dan Administrasi'),
    ('SPD', 'Pendidikan'), ('MPD', 'Pendidikan'), ('SS', 'Bahasa'), ('SHUM', 'Bahasa'),
    ('MSI', 'Tidak spesifik'), ('AMD', 'Tidak spesifik'), ('MSC', 'Tidak spesifik'), ('DR', 'Tidak spesifik'),
    ('SA', 'Tidak spesifik'),
]

TEKNIS = ['Pertanian', 'Kehutanan', 'Perikanan', 'Peternakan', 'Teknologi Pertanian', 'Teknik', 'Sains']
ACUAN_UNIT = [  # rumpun unit, kata kunci pada nama unit, bidang yang relevan
    ('Agribisnis perikanan', 'perikanan', ['Perikanan', 'Sains', 'Teknologi Pertanian', 'Pendidikan']),
    ('Agribisnis peternakan dan kesehatan hewan', 'peternakan|kesehatan hewan', ['Peternakan', 'Sains', 'Pendidikan']),
    ('Agribisnis tanaman dan kehutanan', 'tanaman|kehutanan', ['Pertanian', 'Kehutanan', 'Sains', 'Pendidikan']),
    ('Agroindustri dan teknik kimia', 'agroindustri|teknik kimia', ['Teknologi Pertanian', 'Teknik', 'Sains',
                                                                    'Pendidikan']),
    ('Sains terapan dan otomatisasi pertanian', 'sain|otomatisasi|pengajaran umum',
     ['Sains', 'Teknik', 'Komputer', 'Statistik', 'Pendidikan', 'Bahasa', 'Pertanian', 'Teknologi Pertanian']),
    ('Penjaminan mutu dan fasilitasi pendidikan vokasi', 'mutu|fasilitasi|kemitraan|kerjasama|penyelarasan|kompetensi',
     TEKNIS + ['Pendidikan', 'Ekonomi dan Manajemen', 'Sosial dan Administrasi', 'Statistik', 'Komputer']),
    ('Data, informasi, dan publikasi', 'data|informasi|publikasi|hubungan masyarakat|humas',
     ['Komputer', 'Statistik', 'Sosial dan Administrasi', 'Bahasa', 'Teknik']),
    ('Perencanaan dan penganggaran', 'perencanaan|anggaran',
     ['Ekonomi dan Manajemen', 'Statistik', 'Sosial dan Administrasi', 'Komputer']),
    ('Kepegawaian dan tata laksana', 'kepegawaian|tata laksana',
     ['Hukum', 'Sosial dan Administrasi', 'Ekonomi dan Manajemen', 'Komputer']),
    ('Tata usaha dan rumah tangga', 'tata usaha|rumah tangga|balai',
     ['Ekonomi dan Manajemen', 'Sosial dan Administrasi', 'Hukum', 'Komputer', 'Teknik']),
]


def acuan_gelar(folder='.'):
    p = os.path.join(folder, 'acuan_gelar.csv')
    t = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(ACUAN_GELAR, columns=['gelar', 'bidang'])
    return dict(zip(t.gelar.astype(str).str.upper(), t.bidang))


def acuan_unit(folder='.'):
    p = os.path.join(folder, 'acuan_rumpun_unit.csv')
    if os.path.exists(p):
        t = pd.read_csv(p)
        return [(r.rumpun_unit, r.kata_kunci, [b.strip() for b in str(r.bidang_relevan).split(';')]) for r in t.itertuples()]
    return ACUAN_UNIT


def tabel_acuan_unit():
    return pd.DataFrame([(a, b, '; '.join(c)) for a, b, c in ACUAN_UNIT],
                        columns=['rumpun_unit', 'kata_kunci', 'bidang_relevan'])


# ------------------------------------------------------------------ syarat jabatan
def jenjang_dari_teks(teks):
    """Jenjang terendah yang disebut pada teks syarat pendidikan (mis. '- SLTA - D-1 - D-3' -> 3)."""
    if not isinstance(teks, str):
        return np.nan
    t = teks.upper().replace('–', '-')
    v = [n for k, n in JENJANG.items() if re.search(r'(?<![A-Z0-9])' + re.escape(k) + r'(?![0-9])', t)]
    return min(v) if v else np.nan


def syarat_dari_tugas(tugas):
    """Dari tabel tugas_pegawai: nama_jabatan -> jenjang minimal dan teks syarat."""
    t = tugas.copy()
    t['nama_jabatan'] = t['nama_jabatan'].astype(str).str.strip()
    t['teks'] = t['pendidikan_formal'].fillna('').astype(str).str.replace(r'\s*\n\s*', ' ', regex=True).str.strip()
    t['jenjang_minimal'] = t['pendidikan_formal'].map(jenjang_dari_teks)
    g = t.groupby('nama_jabatan').agg(jenjang_minimal=('jenjang_minimal', 'min'),
                                      syarat_pendidikan=('teks', lambda s: ' | '.join(sorted(set(str(x) for x in s if isinstance(x, str) and x and x != 'nan')))))
    g = g[g.jenjang_minimal.notna()]
    g['jenjang_minimal'] = g['jenjang_minimal'].astype(int)
    return g.reset_index().rename(columns={'nama_jabatan': 'jabatan'})


# ------------------------------------------------------------------ bidang dari gelar
def bidang_dari_nama(nama, peta=None):
    """Daftar bidang ilmu yang tersirat dari gelar akademik pada nama."""
    peta = peta or dict(ACUAN_GELAR)
    if not isinstance(nama, str) or not nama.strip():
        return []
    bagian = nama.split(',')
    gelar = [g for b in bagian[1:] for g in re.split(r'\s{2,}|(?<=\.)\s+(?=[A-Z])', b.strip())]
    gelar += re.findall(r'\b(Dr|Ir|drh|Prof)\.', bagian[0])
    hasil = []
    for g in gelar:
        k = re.sub(r'[\s.,]', '', g).upper()
        if k in peta and peta[k] not in hasil:
            hasil.append(peta[k])
    return hasil


def rumpun_unit(unit, acuan=None):
    acuan = acuan or ACUAN_UNIT
    u = str(unit).lower()
    for nama, kunci, relevan in acuan:
        if re.search(kunci, u):
            return nama, relevan
    return 'Tidak terpetakan', []


# ------------------------------------------------------------------ penilaian per pegawai
def nilai_kesesuaian(d, syarat, folder='.'):
    """d: satu baris per pegawai (kolom jabatan, pendidikan, unit_kerja, opsional nama atau bidang_pendidikan)."""
    peta_g = acuan_gelar(folder)
    acu_u = acuan_unit(folder)
    s = syarat.set_index('jabatan')
    out = d.copy()
    out['jenjang_pegawai'] = out['pendidikan'].map(lambda x: JENJANG.get(str(x).strip().upper(), np.nan))
    out['jenjang_minimal_jabatan'] = out['jabatan'].map(s['jenjang_minimal'])
    out['syarat_pendidikan_jabatan'] = out['jabatan'].map(s['syarat_pendidikan']).fillna('tidak ada di tabel tugas')
    a, b = out['jenjang_pegawai'], out['jenjang_minimal_jabatan']
    kode = np.where(a.isna() | b.isna(), np.nan, np.where(a < b, 1, np.where(a == b, 2, 3)))
    out['status_jenjang'] = pd.Series(kode, index=out.index).map(STATUS_JENJANG).fillna('Tidak dapat dinilai')
    if 'nama' in out and out['nama'].astype(str).str.len().gt(0).any():
        out['bidang_pendidikan'] = out['nama'].map(lambda n: '; '.join(bidang_dari_nama(n, peta_g)))
    elif 'bidang_pendidikan' not in out:
        out['bidang_pendidikan'] = ''
    out['bidang_pendidikan'] = out['bidang_pendidikan'].fillna('').astype(str)
    ru = out['unit_kerja'].map(lambda u: rumpun_unit(u, acu_u)) if 'unit_kerja' in out else pd.Series([('–', [])] * len(out))
    out['rumpun_unit'] = [r[0] for r in ru]
    status = []
    for bid, (_, relevan), jp, jm in zip(out['bidang_pendidikan'], ru, out['jenjang_pegawai'], out['jenjang_minimal_jabatan']):
        bidang = [x.strip() for x in bid.split(';') if x.strip()]
        spesifik = [x for x in bidang if x != 'Tidak spesifik']
        if pd.notna(jp) and jp <= 3:
            status.append('Tidak dinilai (pendidikan SLTA)')
        elif not bidang:
            status.append('Bidang tidak diketahui')
        elif not spesifik:
            status.append('Bidang tidak spesifik')
        elif not relevan:
            status.append('Unit kerja kosong/tidak terpetakan')
        else:
            status.append('Relevan dengan unit' if any(x in relevan for x in spesifik) else 'Kurang relevan dengan unit')
    out['status_bidang_unit'] = status
    out['bidang_relevan_unit'] = [', '.join(r[1]) for r in ru]
    return out


# ------------------------------------------------------------------ matriks kompetensi dengan acuan resmi
def matriks_acuan(m, syarat, resmi=True):
    """Menghitung ulang aspek pendidikan formal (K-formal), total, kesesuaian (> 8) dan kesenjangan (12 - total)
    pada matriks kompetensi. resmi=True memakai jenjang minimal dari tabel tugas jabatan; jabatan yang tidak ada
    di tabel tetap memakai aturan laporan. resmi=False mengembalikan nilai laporan apa adanya."""
    m = m.copy()
    if not resmi:
        m['acuan_jenjang'] = 'aturan laporan'
        return m
    s = syarat.set_index('jabatan')['jenjang_minimal']
    jp = m['pendidikan'].map(lambda x: JENJANG.get(str(x).strip().upper(), np.nan))
    jm = m['jabatan'].map(s)
    ada = jm.notna() & jp.notna()
    baru = np.where(jp < jm, 1, np.where(jp == jm, 2, 3)).astype(float)
    m['k_formal_laporan'] = m['k_formal']
    m.loc[ada & m['k_formal'].notna(), 'k_formal'] = baru[(ada & m['k_formal'].notna()).values]
    m.loc[ada, 'jenj_minta'] = jm[ada]
    m['acuan_jenjang'] = np.where(ada, 'tabel tugas jabatan', 'aturan laporan (jabatan tidak ada di tabel)')
    lengkap = m['kompetensi_total'].notna()
    m.loc[lengkap, 'kompetensi_total'] = m.loc[lengkap, ['k_formal', 'k_nonformal', 'k_terampil', 'k_alaman']].sum(axis=1)
    m.loc[lengkap, 'kompetensi_sesuai'] = (m.loc[lengkap, 'kompetensi_total'] > 8).astype(float)
    m.loc[lengkap, 'kesenjangan'] = 12 - m.loc[lengkap, 'kompetensi_total']
    return m
