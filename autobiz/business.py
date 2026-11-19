"""Name-based business data summaries with explicit heuristic warnings."""

from __future__ import annotations

import pandas as pd


_BUSINESS_FIELDS = (
    (
        "Penjualan / pendapatan",
        ("sales", "revenue", "penjualan", "omzet", "pendapatan"),
        "numeric",
    ),
    (
        "Laba / biaya",
        ("profit", "laba", "cost", "biaya", "expense", "pengeluaran"),
        "numeric",
    ),
    (
        "Persediaan",
        ("stock", "stok", "inventory", "persediaan"),
        "numeric",
    ),
    (
        "Pelanggan / pesanan",
        ("customer", "pelanggan", "client", "buyer", "order", "pesanan"),
        "any",
    ),
    (
        "Pemasaran",
        ("marketing", "campaign", "iklan", "ads", "click", "lead", "konversi"),
        "numeric",
    ),
    (
        "Operasional",
        (
            "operation", "operasional", "delivery", "pengiriman",
            "return", "retur", "lead_time",
        ),
        "numeric",
    ),
)


def business_summary(frame: pd.DataFrame) -> str:
    lines = [
        "RINGKASAN KPI BISNIS (DETEKSI OTOMATIS)",
        "=" * 48,
        "Kolom dipetakan berdasarkan nama; pastikan arti dan satuannya benar.",
        "",
    ]

    for label, keywords, value_kind in _BUSINESS_FIELDS:
        matches = [
            column for column in frame.columns
            if any(word in str(column).casefold() for word in keywords)
            and (
                value_kind == "any"
                or pd.api.types.is_numeric_dtype(frame[column])
            )
        ]
        lines.append(label.upper())
        if not matches:
            lines.append("- Kolom yang sesuai belum terdeteksi.")
            lines.append("")
            continue

        for column in matches[:5]:
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series):
                values = pd.to_numeric(series, errors="coerce").dropna()
                if values.empty:
                    lines.append(f"- {column}: tidak ada nilai numerik valid.")
                    continue
                if label == "Persediaan":
                    lines.append(
                        f"- {column}: rata-rata {values.mean():,.2f}; "
                        f"nilai nol {(values == 0).sum():,}; "
                        f"nilai negatif {(values < 0).sum():,}."
                    )
                else:
                    lines.append(
                        f"- {column}: total {values.sum():,.2f}; "
                        f"rata-rata {values.mean():,.2f} "
                        f"dari {len(values):,} nilai."
                    )
            else:
                lines.append(
                    f"- {column}: {series.nunique(dropna=True):,} nilai unik; "
                    f"{series.isna().sum():,} kosong."
                )
        lines.append("")

    lines.extend([
        "Catatan: angka adalah ringkasan kolom mentah, bukan KPI tervalidasi. "
        "Periksa satuan, duplikasi transaksi, dan definisi bisnis sebelum "
        "mengambil keputusan.",
    ])
    return "\n".join(lines)
