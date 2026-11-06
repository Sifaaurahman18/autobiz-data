import sys
from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QStackedWidget,
    QFrame,
    QHeaderView,
    QTextEdit,
    QLineEdit,
    QSizePolicy,
)


APP_STYLE = """
/* Global Window & Background */
QMainWindow { 
    background-color: #F0F4F9; 
}
QWidget { 
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif; 
    font-size: 13px; 
    color: #1E293B; 
}
QWidget#AppRoot, QWidget#MainContent { 
    background-color: #F0F4F9; 
}
QStackedWidget { 
    background-color: transparent; 
    border: none; 
}
QStackedWidget > QWidget { 
    background-color: transparent; 
}

/* Sidebar Styles */
QFrame#Sidebar { 
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0B192C, stop:1 #1E3E62);
    border: none; 
    border-top-right-radius: 24px;
    border-bottom-right-radius: 24px;
}
QFrame#Sidebar QLabel#Brand { 
    color: #FFFFFF; 
    background-color: transparent; 
    font-size: 22px; 
    font-weight: 800; 
    letter-spacing: 0.5px;
}
QFrame#Sidebar QLabel#BrandSub { 
    color: #38BDF8; 
    background-color: transparent; 
    font-size: 10px; 
    font-weight: 700;
    letter-spacing: 1px;
}
QFrame#Sidebar QLabel { 
    color: #94A3B8; 
    background-color: transparent; 
}

/* Sidebar Navigation Buttons */
QPushButton#NavButton { 
    background-color: transparent; 
    color: #94A3B8; 
    text-align: left; 
    border: none; 
    border-radius: 12px; 
    padding: 12px 16px; 
    font-size: 13px; 
    font-weight: 600;
}
QPushButton#NavButton:hover { 
    background-color: rgba(255, 255, 255, 0.08); 
    color: #F1F5F9; 
}
QPushButton#NavButton[active="true"] { 
    background-color: #2563EB; 
    color: #FFFFFF; 
    font-weight: 700; 
}

/* Page Header */
QLabel#PageTitle { 
    color: #0F172A; 
    font-size: 26px; 
    font-weight: 800; 
}
QLabel#Muted { 
    color: #64748B; 
    background-color: transparent; 
}

/* Panels & Metric Cards */
QFrame#Panel { 
    background-color: #FFFFFF; 
    border: 1px solid #E2E8F0; 
    border-radius: 16px; 
}
QFrame#MetricCard { 
    background-color: #FFFFFF; 
    border: 1px solid #E2E8F0; 
    border-radius: 16px; 
}
QFrame#MetricCard:hover { 
    border: 1px solid #3B82F6; 
}
QLabel#MetricValue { 
    color: #0F172A; 
    font-size: 26px; 
    font-weight: 800; 
}
QLabel#MetricTitle { 
    color: #64748B; 
    font-size: 12px; 
    font-weight: 600;
}

/* Buttons */
QPushButton#PrimaryButton { 
    background-color: #2563EB; 
    color: #FFFFFF; 
    border: none; 
    border-radius: 10px; 
    padding: 11px 20px; 
    font-weight: 700; 
    font-size: 13px;
}
QPushButton#PrimaryButton:hover { 
    background-color: #1D4ED8; 
}
QPushButton#SecondaryButton { 
    background-color: #EFF6FF; 
    color: #2563EB; 
    border: 1px solid #BFDBFE; 
    border-radius: 10px; 
    padding: 10px 18px; 
    font-weight: 600; 
    font-size: 13px;
}
QPushButton#SecondaryButton:hover { 
    background-color: #DBEAFE; 
}
QPushButton:disabled { 
    background-color: #E2E8F0; 
    color: #94A3B8; 
    border: none;
}

/* Tables */
QTableWidget { 
    background-color: #FFFFFF; 
    alternate-background-color: #F8FAFC; 
    color: #1E293B; 
    border: 1px solid #E2E8F0; 
    border-radius: 12px; 
    gridline-color: #F1F5F9; 
    selection-background-color: #EFF6FF; 
    selection-color: #1E40AF; 
}
QHeaderView::section { 
    background-color: #F1F5F9; 
    color: #475569; 
    padding: 12px; 
    border: none; 
    border-bottom: 2px solid #E2E8F0; 
    font-weight: 700; 
}

/* Inputs & Text Area */
QTextEdit, QLineEdit { 
    background-color: #FFFFFF; 
    color: #1E293B; 
    border: 1px solid #CBD5E1; 
    border-radius: 10px; 
    padding: 12px; 
    selection-background-color: #DBEAFE; 
}
QTextEdit:focus, QLineEdit:focus {
    border: 2px solid #3B82F6;
}
"""


def make_panel(title=None):
    panel = QFrame()
    panel.setObjectName("Panel")
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(22, 20, 22, 22)
    layout.setSpacing(12)
    if title:
        heading = QLabel(title)
        heading.setStyleSheet("font-size: 16px; font-weight: 700; color: #0F172A; background: transparent;")
        layout.addWidget(heading)
    return panel, layout


def make_metric(title, value="—", note="Menunggu data"):
    card = QFrame()
    card.setObjectName("MetricCard")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(20, 18, 20, 18)
    layout.setSpacing(8)
    title_label = QLabel(title)
    title_label.setObjectName("MetricTitle")
    value_label = QLabel(value)
    value_label.setObjectName("MetricValue")
    note_label = QLabel(note)
    note_label.setObjectName("Muted")
    note_label.setWordWrap(True)
    layout.addWidget(title_label)
    layout.addWidget(value_label)
    layout.addWidget(note_label)
    return card, value_label, note_label


class AutoBizData(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoBiz Data | Business Growth Assistant")
        self.resize(1280, 800)
        self.setMinimumSize(1000, 650)

        self.df = None
        self.file_name = "Belum ada file"
        self.nav_buttons = []
        self.metric_values = {}

        central = QWidget()
        central.setObjectName("AppRoot")
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Sidebar
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(240)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(20, 28, 20, 22)
        side_layout.setSpacing(10)

        brand = QLabel("AutoBiz Data")
        brand.setObjectName("Brand")
        brand_sub = QLabel("BUSINESS GROWTH ASSISTANT")
        brand_sub.setObjectName("BrandSub")
        side_layout.addWidget(brand)
        side_layout.addWidget(brand_sub)
        side_layout.addSpacing(28)

        nav_items = [
            ("Dashboard", 0),
            ("Data Center", 1),
            ("Business Analysis", 2),
            ("AI Business Advisor", 3),
            ("Reports", 4),
            ("Settings", 5),
        ]
        for label, index in nav_items:
            button = QPushButton(label)
            button.setObjectName("NavButton")
            button.setProperty("active", index == 0)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda checked=False, i=index: self.navigate(i))
            self.nav_buttons.append(button)
            side_layout.addWidget(button)

        side_layout.addStretch()
        version = QLabel("Prototype • Local desktop app")
        version.setStyleSheet("color: #64748B; font-size: 11px;")
        side_layout.addWidget(version)

        # Main content area
        content = QWidget()
        content.setObjectName("MainContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(32, 28, 32, 28)
        content_layout.setSpacing(20)

        topbar = QHBoxLayout()
        self.page_title = QLabel("Dashboard")
        self.page_title.setObjectName("PageTitle")
        topbar.addWidget(self.page_title)
        topbar.addStretch()
        self.file_badge = QLabel("Belum ada data")
        self.file_badge.setObjectName("Muted")
        self.file_badge.setStyleSheet(
            "background-color: #E2E8F0; padding: 6px 14px; border-radius: 20px; font-weight: 600; color: #475569;"
        )
        topbar.addWidget(self.file_badge)
        content_layout.addLayout(topbar)

        self.pages = QStackedWidget()
        self.pages.addWidget(self.build_dashboard_page())
        self.pages.addWidget(self.build_data_page())
        self.pages.addWidget(self.build_analysis_page())
        self.pages.addWidget(self.build_advisor_page())
        self.pages.addWidget(self.build_reports_page())
        self.pages.addWidget(self.build_settings_page())
        content_layout.addWidget(self.pages, 1)

        root.addWidget(sidebar)
        root.addWidget(content, 1)
        self.setStyleSheet(APP_STYLE)
        self.navigate(0)

    def navigate(self, index):
        titles = [
            "Dashboard", "Data Center", "Business Analysis",
            "AI Business Advisor", "Reports", "Settings"
        ]
        self.pages.setCurrentIndex(index)
        self.page_title.setText(titles[index])
        for i, button in enumerate(self.nav_buttons):
            button.setProperty("active", i == index)
            button.style().unpolish(button)
            button.style().polish(button)

    def build_dashboard_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)

        intro = QLabel("Pantau data, temukan insight, dan rencanakan langkah bisnis berikutnya.")
        intro.setObjectName("Muted")
        intro.setStyleSheet("font-size: 14px; padding-bottom: 4px;")
        layout.addWidget(intro)

        metrics = QGridLayout()
        metrics.setSpacing(16)
        for position, (key, title, value, note) in enumerate([
            ("rows", "Jumlah baris", "—", "Data yang dimuat"),
            ("columns", "Jumlah kolom", "—", "Variabel dalam dataset"),
            ("missing", "Sel kosong", "—", "Perlu diperiksa"),
            ("duplicates", "Baris duplikat", "—", "Baris identik terdeteksi"),
        ]):
            card, value_label, note_label = make_metric(title, value, note)
            metrics.addWidget(card, 0, position)
            self.metric_values[key] = (value_label, note_label)
        layout.addLayout(metrics)

        panel, panel_layout = make_panel("Mulai dari sini")
        text = QLabel(
            "Impor file CSV atau Excel untuk melihat ringkasan data. "
            "Setelah data tersedia, AutoBiz Data dapat membantu menyiapkan "
            "analisis yang sesuai dengan kolom yang benar-benar ada."
        )
        text.setWordWrap(True)
        text.setObjectName("Muted")
        panel_layout.addWidget(text)
        buttons = QHBoxLayout()
        import_button = QPushButton("Impor Data")
        import_button.setObjectName("PrimaryButton")
        import_button.clicked.connect(self.import_data)
        data_button = QPushButton("Buka Data Center")
        data_button.setObjectName("SecondaryButton")
        data_button.clicked.connect(lambda: self.navigate(1))
        buttons.addWidget(import_button)
        buttons.addWidget(data_button)
        buttons.addStretch()
        panel_layout.addLayout(buttons)
        layout.addWidget(panel)
        layout.addStretch()
        return page

    def build_data_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        description = QLabel("Impor, periksa, bersihkan, dan ekspor data bisnis.")
        description.setObjectName("Muted")
        layout.addWidget(description)

        actions = QHBoxLayout()
        self.btn_import = QPushButton("Impor Data")
        self.btn_import.setObjectName("PrimaryButton")
        self.btn_clean = QPushButton("Bersihkan Data")
        self.btn_clean.setObjectName("SecondaryButton")
        self.btn_export = QPushButton("Ekspor ke Excel")
        self.btn_export.setObjectName("SecondaryButton")
        self.btn_clean.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_import.clicked.connect(self.import_data)
        self.btn_clean.clicked.connect(self.clean_data)
        self.btn_export.clicked.connect(self.export_data)
        actions.addWidget(self.btn_import)
        actions.addWidget(self.btn_clean)
        actions.addWidget(self.btn_export)
        actions.addStretch()
        layout.addLayout(actions)

        self.status = QLabel("Silakan impor file Excel atau CSV.")
        self.status.setStyleSheet(
            "padding: 12px 16px; background: #EFF6FF; "
            "border: 1px solid #BFDBFE; border-radius: 10px; color: #1E40AF; font-weight: 600;"
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        panel, panel_layout = make_panel("Preview data")
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        panel_layout.addWidget(self.table)
        layout.addWidget(panel, 1)
        return page

    def build_analysis_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        panel, panel_layout = make_panel("Business Analysis")
        info = QLabel(
            "Halaman analisis akan menghitung metrik berdasarkan dataset dan "
            "kolom yang tersedia. Kita tidak akan mengarang angka seperti "
            "omzet atau laba jika kolom yang dibutuhkan belum ada."
        )
        info.setWordWrap(True)
        info.setObjectName("Muted")
        panel_layout.addWidget(info)
        self.analysis_summary = QLabel("Impor data melalui Data Center untuk memulai.")
        self.analysis_summary.setWordWrap(True)
        panel_layout.addWidget(self.analysis_summary)
        open_data = QPushButton("Buka Data Center")
        open_data.setObjectName("PrimaryButton")
        open_data.clicked.connect(lambda: self.navigate(1))
        panel_layout.addWidget(open_data, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(panel)
        layout.addStretch()
        return page

    def build_advisor_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        panel, panel_layout = make_panel("Konsultasi bisnis dengan AI")
        intro = QLabel(
            "Area konsultasi ini adalah kerangka antarmuka. AI belum tersambung "
            "ke API pada versi ini. Setelah integrasi dilakukan, percakapan dapat "
            "memakai profil bisnis dan ringkasan data yang kamu izinkan."
        )
        intro.setWordWrap(True)
        intro.setObjectName("Muted")
        panel_layout.addWidget(intro)

        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setPlaceholderText(
            "Riwayat percakapan akan tampil di sini setelah integrasi AI."
        )
        self.chat_history.setMinimumHeight(260)
        panel_layout.addWidget(self.chat_history, 1)

        chat_row = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText(
            "Contoh: Bagaimana cara meningkatkan keuntungan bisnis saya?"
        )
        self.chat_input.setEnabled(False)
        send_button = QPushButton("Kirim")
        send_button.setObjectName("PrimaryButton")
        send_button.setEnabled(False)
        chat_row.addWidget(self.chat_input, 1)
        chat_row.addWidget(send_button)
        panel_layout.addLayout(chat_row)
        layout.addWidget(panel)
        layout.addStretch()
        return page

    def build_reports_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        panel, panel_layout = make_panel("Reports")
        description = QLabel(
            "Ekspor dataset yang sedang aktif ke file Excel melalui Data Center. "
            "Nantinya halaman ini akan menjadi pusat laporan analisis dan ringkasan bisnis."
        )
        description.setWordWrap(True)
        description.setObjectName("Muted")
        panel_layout.addWidget(description)
        export_button = QPushButton("Ekspor data ke Excel")
        export_button.setObjectName("PrimaryButton")
        export_button.clicked.connect(self.export_data)
        panel_layout.addWidget(export_button, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(panel)
        layout.addStretch()
        return page

    def build_settings_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        panel, panel_layout = make_panel("Pengaturan aplikasi")
        info = QLabel(
            "Pengaturan profil bisnis, preferensi analisis, penyimpanan lokal, "
            "dan konfigurasi provider AI akan ditambahkan pada tahap berikutnya. "
            "Jangan menaruh API key langsung di dalam kode sumber."
        )
        info.setWordWrap(True)
        info.setObjectName("Muted")
        panel_layout.addWidget(info)
        layout.addWidget(panel)
        layout.addStretch()
        return page

    def update_dashboard(self):
        if self.df is None:
            return
        missing = int(self.df.isna().sum().sum())
        duplicates = int(self.df.duplicated().sum())
        values = {
            "rows": (f"{len(self.df):,}", "Baris dalam dataset aktif"),
            "columns": (f"{len(self.df.columns):,}", "Kolom dalam dataset aktif"),
            "missing": (f"{missing:,}", "Sel bernilai kosong"),
            "duplicates": (f"{duplicates:,}", "Baris identik terdeteksi"),
        }
        for key, (value, note) in values.items():
            self.metric_values[key][0].setText(value)
            self.metric_values[key][1].setText(note)

        self.analysis_summary.setText(
            f"Dataset aktif: {self.file_name}\n"
            f"Jumlah baris: {len(self.df):,}\n"
            f"Jumlah kolom: {len(self.df.columns):,}\n"
            f"Sel kosong: {missing:,}\n"
            f"Baris duplikat: {duplicates:,}\n\n"
            "Metrik keuangan dan operasional akan dihitung setelah kolom "
            "yang relevan dikenali atau dipetakan."
        )

    def import_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Pilih file data",
            "",
            "File Data (*.csv *.xlsx);;Semua File (*)",
        )
        if not path:
            return

        try:
            file_path = Path(path)
            if file_path.suffix.lower() == ".csv":
                try:
                    df = pd.read_csv(file_path, sep=None, engine="python")
                except UnicodeDecodeError:
                    df = pd.read_csv(
                        file_path, sep=None, engine="python", encoding="latin-1"
                    )
            elif file_path.suffix.lower() == ".xlsx":
                df = pd.read_excel(file_path)
            else:
                QMessageBox.warning(
                    self, "Format tidak didukung", "Pilih file CSV atau XLSX."
                )
                return

            if df.empty:
                QMessageBox.warning(
                    self, "Data kosong", "File tidak memiliki baris data."
                )
                return

            self.df = df
            self.file_name = file_path.name
            self.show_data()
            self.btn_clean.setEnabled(True)
            self.btn_export.setEnabled(True)
            self.status.setText(
                f"File: {self.file_name} | {len(df):,} baris | "
                f"{len(df.columns)} kolom"
            )
            self.file_badge.setText(self.file_name)
            self.update_dashboard()
        except Exception as error:
            QMessageBox.critical(
                self, "Gagal mengimpor", f"File tidak dapat dibaca:\n{error}"
            )

    def show_data(self):
        if self.df is None:
            return

        preview = self.df.head(500)
        self.table.clear()
        self.table.setRowCount(len(preview))
        self.table.setColumnCount(len(preview.columns))
        self.table.setHorizontalHeaderLabels(
            [str(column) for column in preview.columns]
        )

        for row_index, row in enumerate(preview.itertuples(index=False, name=None)):
            for col_index, value in enumerate(row):
                text = "" if pd.isna(value) else str(value)
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row_index, col_index, item)
        self.table.resizeColumnsToContents()

    def clean_data(self):
        if self.df is None:
            return

        before_rows = len(self.df)
        self.df = self.df.dropna(how="all").drop_duplicates()
        self.df.columns = [str(column).strip() for column in self.df.columns]

        for column in self.df.select_dtypes(include=["object", "string"]).columns:
            self.df[column] = self.df[column].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )

        removed = before_rows - len(self.df)
        self.show_data()
        self.update_dashboard()
        self.status.setText(
            f"Data dibersihkan: {removed:,} baris kosong/duplikat dihapus. "
            f"Sisa {len(self.df):,} baris."
        )

    def export_data(self):
        if self.df is None:
            QMessageBox.information(
                self, "Belum ada data", "Impor data terlebih dahulu."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Simpan laporan", "laporan_autobiz.xlsx", "Excel (*.xlsx)"
        )
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"

        try:
            self.df.to_excel(path, index=False)
            QMessageBox.information(
                self, "Berhasil", f"Laporan berhasil disimpan:\n{path}"
            )
        except Exception as error:
            QMessageBox.critical(
                self, "Gagal mengekspor", f"Laporan tidak dapat disimpan:\n{error}"
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AutoBizData()
    window.show()
    sys.exit(app.exec())