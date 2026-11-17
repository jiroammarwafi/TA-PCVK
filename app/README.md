# app/ — Aplikasi Web (Divisi A4)

Folder ini **sengaja masih kosong**. Disediakan sebagai tempat kerja divisi
**A4 (Web & Integrasi)** agar terpisah dari kode penelitian di `research/`.

## Rencana isi

```
app/
├── main.py             # entry point (FastAPI atau server stdlib)
├── templates/          # HTML
├── static/
│   ├── css/            # CSS kustom (tanpa CDN)
│   └── js/             # JS vanilla
└── inference.py        # memuat model dari reports/ + prediksi
```

## Bahan yang dibutuhkan dari A1–A3

| Kebutuhan | Lokasi | Status |
|---|---|---|
| Matriks fitur | `data/a2-features/*.npz` | ✅ tersedia |
| Parameter model (LDA + LS-SVM) | belum diekspor | ❌ perlu dibuat A3 |
| Daftar kelas & metrik | `reports/a3-models/*.json` | ✅ tersedia |
| Hasil uji noise & split | `reports/a1-subject-independent/*` | ✅ tersedia |

## Catatan teknis

- Rencana UI: **HTML + CSS + JavaScript vanilla**, tanpa framework dan tanpa CDN
  agar demo tetap berjalan offline.
- Belum ada Docker pada tahap ini.
- Agar web dan penelitian tidak tercampur, A4 **tidak** menaruh kode di
  `research/`; cukup mengimpor modul dari sana bila perlu.
