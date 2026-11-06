
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFormLayout, QStackedWidget, QFileDialog,
    QMessageBox, QTableWidget, QTableWidgetItem, QComboBox, QSpinBox,
    QLineEdit, QTextEdit, QScrollArea, QSizePolicy, QHeaderView,
    QAbstractItemView
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, r2_score, silhouette_score,
    classification_report
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    IsolationForest, ExtraTreesRegressor
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

warnings.filterwarnings("ignore")


APP_STYLE = """
QMainWindow, QWidget#mainContent {
    background: #f2f6fc;
    color: #18345b;
    font-family: Segoe UI;
    font-size: 12px;
}
QWidget#sidebar {
    background: #103477;
}
QLabel#brand {
    color: white;
    font-size: 23px;
    font-weight: 700;
}
QLabel#brandSub {
    color: #d7e5ff;
    font-size: 9px;
}
QLabel#pageTitle {
    color: #153765;
    font-size: 27px;
    font-weight: 700;
}
QLabel#subtitle {
    color: #7085a8;
    font-size: 12px;
}
QFrame#card {
    background: white;
    border: 1px solid #dce6f5;
    border-radius: 12px;
}
QLabel#sectionTitle {
    font-size: 16px;
    font-weight: 700;
    color: #173765;
    background: transparent;
    border: none;
}
QLabel#metricLabel {
    color: #7185a8;
    font-size: 10px;
    background: transparent;
    border: none;
}
QLabel#metricValue {
    color: #153765;
    font-size: 24px;
    font-weight: 700;
    background: transparent;
    border: none;
}
QLabel#muted {
    color: #7085a8;
    background: transparent;
    border: none;
}
QPushButton {
    border: none;
    border-radius: 7px;
    padding: 10px 12px;
    background: #eaf1ff;
    color: #17458e;
    font-weight: 600;
}
QPushButton:hover {
    background: #dce8ff;
}
QPushButton#primary {
    background: #2861e8;
    color: white;
    min-height: 20px;
}
QPushButton#primary:hover {
    background: #174dcc;
}
QPushButton#navButton {
    background: transparent;
    color: #e5edff;
    text-align: left;
    padding: 12px 10px;
    border-radius: 8px;
    min-height: 18px;
}
QPushButton#navButton:hover, QPushButton#navButton:checked {
    background: #2868e8;
    color: white;
}
QComboBox, QSpinBox, QLineEdit {
    background: white;
    color: #18345b;
    border: 1px solid #d5e0f1;
    border-radius: 7px;
    padding: 7px 9px;
    min-height: 22px;
}
QComboBox QAbstractItemView {
    background: white;
    color: #18345b;
    selection-background-color: #dce8ff;
}
QTextEdit, QTableWidget {
    background: white;
    color: #18345b;
    border: 1px solid #dce6f5;
    border-radius: 7px;
    gridline-color: #e7edf7;
    selection-background-color: #dce8ff;
    selection-color: #173765;
}
QHeaderView::section {
    background: #f4f7fc;
    color: #254873;
    border: none;
    border-bottom: 1px solid #dce6f5;
    padding: 8px;
    font-weight: 600;
}
QScrollArea {
    border: none;
    background: transparent;
}
"""


def numeric_columns(df):
    if df is None:
        return []
    return list(df.select_dtypes(include=np.number).columns)


def make_card():
    card = QFrame()
    card.setObjectName("card")
    return card


class ChartCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(7, 3), dpi=100)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setMinimumHeight(240)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

    def clear_chart(self, message="Belum ada visualisasi"):
        self.axes.clear()
        self.axes.text(
            0.5, 0.5, message, ha="center", va="center",
            transform=self.axes.transAxes, color="#7085a8"
        )
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.figure.tight_layout()
        self.draw()

    def plot_data(self, df, column=None):
        self.axes.clear()
        if df is None or df.empty:
            self.clear_chart("Impor dataset terlebih dahulu")
            return

        cols = numeric_columns(df)
        if not cols:
            self.clear_chart("Dataset tidak memiliki kolom numerik")
            return

        col = column if column in cols else cols[0]
        series = pd.to_numeric(df[col], errors="coerce").dropna()

        if series.empty:
            self.clear_chart("Tidak ada nilai numerik valid")
            return

        # Gunakan sampel untuk visualisasi agar tetap responsif.
        if len(series) > 1500:
            series = series.sample(1500, random_state=42)

        self.axes.plot(
            np.arange(len(series)), series.values,
            color="#3478f6", linewidth=1.5
        )
        self.axes.set_title(str(col), loc="left", color="#254873")
        self.axes.set_xlabel("Observasi")
        self.axes.grid(True, alpha=0.2)
        self.figure.tight_layout()
        self.draw()


class AutoBizData(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AutoBiz Data | Business Growth Assistant")
        self.resize(1400, 850)
        self.setMinimumSize(900, 620)

        # Logo aplikasi: simpan gambar di assets/autobiz_logo.png.
        self.base_dir = Path(__file__).resolve().parent
        self.logo_path = self.base_dir / "assets" / "autobiz_logo.png"
        if not self.logo_path.exists():
            # Alternatif jika logo disimpan dalam format JPG.
            jpg_logo = self.base_dir / "assets" / "autobiz_logo.jpg"
            if jpg_logo.exists():
                self.logo_path = jpg_logo
        if self.logo_path.exists():
            self.setWindowIcon(QIcon(str(self.logo_path)))

        self.df = None
        self.original_df = None
        self.file_path = None
        self.last_analysis = ""

        self.pages = QStackedWidget()
        self.nav_buttons = {}

        self.build_ui()
        self.show_page("Dashboard")

    # ---------------------------------------------------------
    # Layout utama
    # ---------------------------------------------------------
    def build_ui(self):
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(205)

        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(16, 20, 16, 18)
        side_layout.setSpacing(8)

        logo = QLabel()
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setFixedSize(88, 88)
        logo.setStyleSheet(
            "background:white;border-radius:12px;padding:3px;"
        )
        if self.logo_path.exists():
            pixmap = QPixmap(str(self.logo_path))
            if not pixmap.isNull():
                logo.setPixmap(pixmap.scaled(
                    82, 82,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                ))
            else:
                logo.setText("✦")
        else:
            logo.setText("✦")
            logo.setStyleSheet(
                "background:white;color:#2861e8;border-radius:12px;"
                "font-size:32px;font-weight:bold;"
            )

        brand = QLabel("AutoBiz Data")
        brand.setObjectName("brand")

        brand_sub = QLabel("BUSINESS GROWTH ASSISTANT")
        brand_sub.setObjectName("brandSub")

        side_layout.addWidget(logo)
        side_layout.addSpacing(4)
        side_layout.addWidget(brand)
        side_layout.addWidget(brand_sub)
        side_layout.addSpacing(22)

        section = QLabel("WORKSPACE")
        section.setStyleSheet("color:#a9c5ff;font-size:10px;")
        side_layout.addWidget(section)

        navigation = [
            "Dashboard",
            "Data Center",
            "Business Analysis",
            "AI Business Advisor",
            "Reports",
            "Settings",
        ]

        for name in navigation:
            button = QPushButton("  " + name)
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(
                lambda checked=False, n=name: self.show_page(n)
            )
            self.nav_buttons[name] = button
            side_layout.addWidget(button)

        side_layout.addStretch()

        local_note = QLabel(
            "● LOCAL WORKSPACE\n\n"
            "Data diproses secara lokal.\n"
            "Pilih dataset untuk memulai."
        )
        local_note.setWordWrap(True)
        local_note.setStyleSheet(
            "color:#d7e5ff;font-size:10px;background:transparent;"
        )
        side_layout.addWidget(local_note)

        # Kredit pembuat aplikasi di bagian bawah sidebar.
        credit = QLabel("Developed by Sifa Aurahman\n© 2026 AutoBiz Data")
        credit.setWordWrap(True)
        credit.setAlignment(Qt.AlignmentFlag.AlignLeft)
        credit.setStyleSheet(
            "color:#d7e5ff;font-size:9px;background:transparent;"
            "border-top:1px solid #31558b;padding-top:10px;"
        )
        side_layout.addWidget(credit)

        content = QWidget()
        content.setObjectName("mainContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(22, 18, 22, 18)
        content_layout.setSpacing(12)
        content_layout.addWidget(self.pages)

        root_layout.addWidget(sidebar)
        root_layout.addWidget(content, 1)

        self.setCentralWidget(root)

        self.dashboard_page = self.create_dashboard_page()
        self.data_page = self.create_data_page()
        self.analysis_page = self.create_analysis_page()
        self.ai_page = self.create_ai_page()
        self.reports_page = self.create_reports_page()
        self.settings_page = self.create_settings_page()

        for page in (
            self.dashboard_page,
            self.data_page,
            self.analysis_page,
            self.ai_page,
            self.reports_page,
            self.settings_page,
        ):
            self.pages.addWidget(page)

    def page_header(self, title, subtitle):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setObjectName("pageTitle")

        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("subtitle")
        subtitle_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        return widget

    def show_page(self, name):
        mapping = {
            "Dashboard": self.dashboard_page,
            "Data Center": self.data_page,
            "Business Analysis": self.analysis_page,
            "AI Business Advisor": self.ai_page,
            "Reports": self.reports_page,
            "Settings": self.settings_page,
        }
        self.pages.setCurrentWidget(mapping[name])

        for nav_name, button in self.nav_buttons.items():
            button.setChecked(nav_name == name)

        if name == "Dashboard":
            self.refresh_dashboard()
        elif name == "Data Center":
            self.refresh_data_table()
        elif name == "Business Analysis":
            self.refresh_analysis_columns()

    # ---------------------------------------------------------
    # Dashboard
    # ---------------------------------------------------------
    def create_dashboard_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        header_row = QHBoxLayout()
        header_row.addWidget(
            self.page_header(
                "Dashboard",
                "Pantau data, temukan insight, dan rencanakan langkah bisnis."
            ),
            1
        )

        self.file_badge = QLabel("Belum ada dataset")
        self.file_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_badge.setStyleSheet(
            "background:white;border:1px solid #dce6f5;"
            "border-radius:18px;padding:9px 13px;color:#526d96;"
        )
        header_row.addWidget(self.file_badge, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(header_row)

        import_button = QPushButton("+  Import business data")
        import_button.setObjectName("primary")
        import_button.clicked.connect(self.import_data)
        layout.addWidget(import_button, 0, Qt.AlignmentFlag.AlignRight)

        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(12)

        self.metric_values = {}
        for key, label in [
            ("rows", "TOTAL BARIS"),
            ("columns", "TOTAL KOLOM"),
            ("missing", "SEL KOSONG"),
            ("duplicates", "DUPLIKAT"),
        ]:
            card = make_card()
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(14, 12, 14, 12)
            card_layout.setSpacing(5)

            label_widget = QLabel(label)
            label_widget.setObjectName("metricLabel")
            value_widget = QLabel("—")
            value_widget.setObjectName("metricValue")

            detail = QLabel("Menunggu dataset")
            detail.setObjectName("muted")

            card_layout.addWidget(label_widget)
            card_layout.addWidget(value_widget)
            card_layout.addWidget(detail)

            self.metric_values[key] = (value_widget, detail)
            metrics_row.addWidget(card)

        layout.addLayout(metrics_row)

        body = QHBoxLayout()
        body.setSpacing(12)

        left_column = QVBoxLayout()
        left_column.setSpacing(12)

        chart_card = make_card()
        chart_layout = QVBoxLayout(chart_card)
        chart_layout.setContentsMargins(15, 12, 15, 12)

        chart_title = QLabel("Data overview")
        chart_title.setObjectName("sectionTitle")
        chart_desc = QLabel("Visualisasi kolom numerik dari dataset aktif")
        chart_desc.setObjectName("muted")

        self.dashboard_column_combo = QComboBox()
        self.dashboard_column_combo.currentTextChanged.connect(
            self.update_dashboard_chart
        )

        self.dashboard_chart = ChartCanvas()

        chart_layout.addWidget(chart_title)
        chart_layout.addWidget(chart_desc)
        chart_layout.addWidget(self.dashboard_column_combo)
        chart_layout.addWidget(self.dashboard_chart, 1)

        preview_card = make_card()
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(15, 12, 15, 12)

        preview_title = QLabel("Data preview")
        preview_title.setObjectName("sectionTitle")
        preview_desc = QLabel("Contoh baris dari dataset aktif")
        preview_desc.setObjectName("muted")

        self.dashboard_table = QTableWidget()
        self.configure_table(self.dashboard_table)

        preview_layout.addWidget(preview_title)
        preview_layout.addWidget(preview_desc)
        preview_layout.addWidget(self.dashboard_table, 1)

        left_column.addWidget(chart_card, 3)
        left_column.addWidget(preview_card, 2)

        ai_card = make_card()
        ai_layout = QVBoxLayout(ai_card)
        ai_layout.setContentsMargins(14, 12, 14, 12)
        ai_layout.setSpacing(10)

        ai_title = QLabel("Ask AI ✦")
        ai_title.setObjectName("sectionTitle")

        ai_subtitle = QLabel("Asisten eksplorasi bisnis")
        ai_subtitle.setObjectName("muted")

        self.dashboard_ai_text = QTextEdit()
        self.dashboard_ai_text.setReadOnly(True)
        self.dashboard_ai_text.setPlainText(
            "Halo! Impor dataset untuk memulai eksplorasi bisnis.\n\n"
            "AI eksternal belum terhubung. Insight statistik dasar "
            "tersedia melalui Business Analysis."
        )

        ask_analysis = QPushButton("Buka Business Analysis")
        ask_analysis.clicked.connect(
            lambda: self.show_page("Business Analysis")
        )

        ai_layout.addWidget(ai_title)
        ai_layout.addWidget(ai_subtitle)
        ai_layout.addWidget(self.dashboard_ai_text, 1)
        ai_layout.addWidget(ask_analysis)

        body.addLayout(left_column, 7)
        body.addWidget(ai_card, 3)
        layout.addLayout(body, 1)

        return page

    def refresh_dashboard(self):
        if self.df is None:
            self.file_badge.setText("Belum ada dataset")
            for value, detail in self.metric_values.values():
                value.setText("—")
                detail.setText("Menunggu dataset")

            self.dashboard_column_combo.clear()
            self.dashboard_table.clear()
            self.dashboard_table.setRowCount(0)
            self.dashboard_table.setColumnCount(0)
            self.dashboard_chart.clear_chart("Impor dataset terlebih dahulu")
            return

        self.file_badge.setText(
            Path(self.file_path).name if self.file_path else "Dataset aktif"
        )

        df = self.df
        metrics = {
            "rows": (len(df), "Jumlah catatan"),
            "columns": (len(df.columns), "Variabel dataset"),
            "missing": (int(df.isna().sum().sum()), "Perlu ditinjau"),
            "duplicates": (int(df.duplicated().sum()), "Baris identik"),
        }

        for key, (number, description) in metrics.items():
            value_label, detail_label = self.metric_values[key]

            value_label.setText(f"{number:,}")
            detail_label.setText(description)

        old_col = self.dashboard_column_combo.currentText()
        self.dashboard_column_combo.blockSignals(True)
        self.dashboard_column_combo.clear()
        self.dashboard_column_combo.addItems(numeric_columns(df))

        if old_col in numeric_columns(df):
            self.dashboard_column_combo.setCurrentText(old_col)

        self.dashboard_column_combo.blockSignals(False)

        self.refresh_table_widget(self.dashboard_table, df.head(8))
        self.update_dashboard_chart()

    def update_dashboard_chart(self, *_):
        self.dashboard_chart.plot_data(
            self.df, self.dashboard_column_combo.currentText()
        )

    # ---------------------------------------------------------
    # Data Center
    # ---------------------------------------------------------
    def create_data_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        layout.addWidget(
            self.page_header(
                "Data Center",
                "Impor, periksa, bersihkan, dan ekspor dataset."
            )
        )

        toolbar = QHBoxLayout()

        import_btn = QPushButton("Import CSV / Excel")
        import_btn.setObjectName("primary")
        import_btn.clicked.connect(self.import_data)

        clean_btn = QPushButton("Clean data")
        clean_btn.clicked.connect(self.clean_data)

        export_btn = QPushButton("Export Excel")
        export_btn.clicked.connect(self.export_data)

        toolbar.addWidget(import_btn)
        toolbar.addWidget(clean_btn)
        toolbar.addWidget(export_btn)
        toolbar.addStretch()

        layout.addLayout(toolbar)

        self.data_summary = QLabel("Belum ada dataset.")
        self.data_summary.setObjectName("muted")
        self.data_summary.setWordWrap(True)
        layout.addWidget(self.data_summary)

        self.data_table = QTableWidget()
        self.configure_table(self.data_table)
        layout.addWidget(self.data_table, 1)

        return page

    def configure_table(self, table):
        table.setAlternatingRowColors(True)
        table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        table.setHorizontalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setDefaultSectionSize(28)

    def refresh_table_widget(self, table, df):
        table.clear()
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels([str(c) for c in df.columns])

        for i, (_, row) in enumerate(df.iterrows()):
            for j, value in enumerate(row):
                if pd.isna(value):
                    text = ""
                else:
                    text = str(value)

                item = QTableWidgetItem(text)
                item.setToolTip(text)
                table.setItem(i, j, item)

        table.resizeRowsToContents()

    def refresh_data_table(self):
        if self.df is None:
            self.data_summary.setText("Impor CSV atau Excel untuk memulai.")
            self.data_table.setRowCount(0)
            self.data_table.setColumnCount(0)
            return

        self.data_summary.setText(
            f"{len(self.df):,} baris | {len(self.df.columns)} kolom | "
            f"{self.df.isna().sum().sum():,} sel kosong"
        )
        self.refresh_table_widget(self.data_table, self.df.head(1000))

    # ---------------------------------------------------------
    # Import, cleaning, export
    # ---------------------------------------------------------
    def import_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Pilih dataset bisnis",
            "",
            "Data files (*.csv *.xlsx *.xls);;CSV (*.csv);;Excel (*.xlsx *.xls)"
        )
        if not path:
            return

        try:
            suffix = Path(path).suffix.lower()

            if suffix == ".csv":
                try:
                    df = pd.read_csv(path, low_memory=False)
                except UnicodeDecodeError:
                    df = pd.read_csv(
                        path, encoding="latin-1", low_memory=False
                    )
            elif suffix in (".xlsx", ".xls"):
                df = pd.read_excel(path)
            else:
                raise ValueError("Format file belum didukung.")

            if df.empty:
                QMessageBox.warning(
                    self, "Dataset kosong",
                    "File berhasil dibuka tetapi tidak berisi baris data."
                )
                return

            self.df = df
            self.original_df = df.copy()
            self.file_path = path

            self.refresh_dashboard()
            self.refresh_data_table()
            self.refresh_analysis_columns()

            QMessageBox.information(
                self, "Import berhasil",
                f"Dataset berhasil dimuat.\n\n"
                f"Baris: {len(df):,}\n"
                f"Kolom: {len(df.columns)}"
            )

        except Exception as exc:
            QMessageBox.critical(
                self, "Gagal mengimpor data", str(exc)
            )

    def clean_data(self):
        if self.df is None:
            QMessageBox.warning(
                self, "Tidak ada dataset", "Impor dataset terlebih dahulu."
            )
            return

        before_rows = len(self.df)
        before_duplicates = int(self.df.duplicated().sum())

        df = self.df.copy()
        df.columns = [
            str(c).strip() for c in df.columns
        ]
        df = df.dropna(how="all")
        df = df.drop_duplicates()

        self.df = df.reset_index(drop=True)

        self.refresh_dashboard()
        self.refresh_data_table()
        self.refresh_analysis_columns()

        QMessageBox.information(
            self, "Pembersihan selesai",
            f"Baris sebelum: {before_rows:,}\n"
            f"Baris sesudah: {len(self.df):,}\n"
            f"Baris duplikat yang dihapus: {before_duplicates:,}\n\n"
            "Pembersihan tidak menghapus semua nilai kosong pada "
            "baris yang masih memiliki data."
        )

    def export_data(self):
        if self.df is None:
            QMessageBox.warning(
                self, "Tidak ada dataset", "Impor dataset terlebih dahulu."
            )
            return

        path, selected = QFileDialog.getSaveFileName(
            self, "Ekspor dataset", "business_data.xlsx",
            "Excel (*.xlsx);;CSV (*.csv)"
        )
        if not path:
            return

        try:
            if path.lower().endswith(".csv") or "CSV" in selected:
                if not path.lower().endswith(".csv"):
                    path += ".csv"
                self.df.to_csv(path, index=False)
            else:
                if not path.lower().endswith(".xlsx"):
                    path += ".xlsx"
                self.df.to_excel(path, index=False)

            QMessageBox.information(
                self, "Ekspor berhasil", f"File disimpan:\n{path}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Gagal ekspor", str(exc))

    # ---------------------------------------------------------
    # Business Analysis
    # ---------------------------------------------------------
    def create_analysis_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        layout.addWidget(
            self.page_header(
                "Business Analysis",
                "Data Science workspace: EDA, machine learning, "
                "clustering, forecasting, dan evaluasi."
            )
        )

        # Scroll area mencegah tabrakan ketika jendela pendek.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 4, 0)
        scroll_layout.setSpacing(14)

        setup_card = make_card()
        setup_layout = QVBoxLayout(setup_card)
        setup_layout.setContentsMargins(18, 16, 18, 16)
        setup_layout.setSpacing(14)

        title = QLabel("Analysis setup")
        title.setObjectName("sectionTitle")
        setup_layout.addWidget(title)

        description = QLabel(
            "Pilih teknik analisis, variabel, dan parameter yang sesuai."
        )
        description.setObjectName("muted")
        description.setWordWrap(True)
        setup_layout.addWidget(description)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(12)
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Exploratory Data Analysis (EDA)",
            "Regression comparison",
            "Classification comparison",
            "Clustering",
            "Forecasting",
            "Anomaly Detection",
            "Correlation analysis",
        ])
        self.method_combo.currentTextChanged.connect(
            self.update_analysis_controls
        )

        self.date_combo = QComboBox()
        self.target_combo = QComboBox()
        self.feature_combo = QComboBox()

        self.period_spin = QSpinBox()
        self.period_spin.setRange(1, 365)
        self.period_spin.setValue(30)
        self.period_spin.setSuffix(" periode")

        self.cluster_spin = QSpinBox()
        self.cluster_spin.setRange(2, 12)
        self.cluster_spin.setValue(3)

        form.addRow("Pilih teknik analisis", self.method_combo)
        form.addRow("Kolom tanggal", self.date_combo)
        form.addRow("Target / variabel", self.target_combo)
        form.addRow("Kolom tambahan", self.feature_combo)
        form.addRow("Periode prediksi", self.period_spin)
        form.addRow("Jumlah cluster", self.cluster_spin)

        setup_layout.addLayout(form)

        self.run_analysis_button = QPushButton("Jalankan analisis")
        self.run_analysis_button.setObjectName("primary")
        self.run_analysis_button.setMinimumHeight(40)
        self.run_analysis_button.clicked.connect(self.run_analysis)
        setup_layout.addWidget(self.run_analysis_button)

        scroll_layout.addWidget(setup_card)

        self.analysis_status = QLabel(
            "Impor dataset untuk menjalankan analisis Data Science."
        )
        self.analysis_status.setObjectName("muted")
        self.analysis_status.setWordWrap(True)
        scroll_layout.addWidget(self.analysis_status)

        result_card = make_card()
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(16, 14, 16, 14)
        result_layout.setSpacing(10)

        result_title = QLabel("Interpretasi dan hasil")
        result_title.setObjectName("sectionTitle")
        result_layout.addWidget(result_title)

        self.analysis_result = QTextEdit()
        self.analysis_result.setReadOnly(True)
        self.analysis_result.setMinimumHeight(180)
        self.analysis_result.setPlaceholderText(
            "Hasil analisis, metrik model, dan interpretasi akan muncul di sini."
        )
        result_layout.addWidget(self.analysis_result)

        scroll_layout.addWidget(result_card)

        chart_card = make_card()
        chart_layout = QVBoxLayout(chart_card)
        chart_layout.setContentsMargins(14, 12, 14, 12)

        chart_title = QLabel("Visualisasi analisis")
        chart_title.setObjectName("sectionTitle")
        chart_layout.addWidget(chart_title)

        self.analysis_chart = ChartCanvas()
        chart_layout.addWidget(self.analysis_chart)

        scroll_layout.addWidget(chart_card)
        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        self.update_analysis_controls()
        return page

    def refresh_analysis_columns(self):
        combos = [
            self.date_combo, self.target_combo, self.feature_combo
        ]

        for combo in combos:
            combo.blockSignals(True)
            combo.clear()

        if self.df is not None:
            columns = [str(c) for c in self.df.columns]
            nums = numeric_columns(self.df)

            self.date_combo.addItem("(Tidak digunakan)")
            self.date_combo.addItems(columns)

            self.target_combo.addItem("(Pilih otomatis)")
            self.target_combo.addItems(columns)

            self.feature_combo.addItem("(Semua fitur numerik)")
            self.feature_combo.addItems(nums)

            # Deteksi kolom tanggal berdasarkan nama.
            date_candidates = [
                c for c in columns
                if any(word in c.lower() for word in [
                    "date", "time", "tanggal", "month", "year"
                ])
            ]
            if date_candidates:
                self.date_combo.setCurrentText(date_candidates[0])

            # Pilih target numerik yang umum pada data bisnis.
            preferred = [
                c for c in nums
                if any(word in c.lower() for word in [
                    "sales", "revenue", "amount", "quantity",
                    "profit", "price", "total"
                ])
            ]
            if preferred:
                self.target_combo.setCurrentText(preferred[0])
            elif nums:
                self.target_combo.setCurrentText(nums[0])

        for combo in combos:
            combo.blockSignals(False)

        self.update_analysis_controls()

    def update_analysis_controls(self, *_):
        method = self.method_combo.currentText()

        forecast = method == "Forecasting"
        clustering = method == "Clustering"

        self.date_combo.setEnabled(forecast)
        self.period_spin.setEnabled(forecast)
        self.cluster_spin.setEnabled(clustering)

    def run_analysis(self):
        if self.df is None:
            QMessageBox.warning(
                self, "Tidak ada dataset",
                "Impor dataset sebelum menjalankan analisis."
            )
            return

        method = self.method_combo.currentText()
        self.run_analysis_button.setEnabled(False)
        self.analysis_status.setText("Analisis sedang berjalan...")

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if method == "Exploratory Data Analysis (EDA)":
                result = self.analyze_eda()
            elif method == "Regression comparison":
                result = self.analyze_regression()
            elif method == "Classification comparison":
                result = self.analyze_classification()
            elif method == "Clustering":
                result = self.analyze_clustering()
            elif method == "Forecasting":
                result = self.analyze_forecasting()
            elif method == "Anomaly Detection":
                result = self.analyze_anomalies()
            else:
                result = self.analyze_correlation()

            self.last_analysis = result
            self.analysis_result.setPlainText(result)
            self.analysis_status.setText(
                "Analisis selesai. Periksa hasil, metrik, dan visualisasi."
            )
        except Exception as exc:
            self.analysis_status.setText("Analisis gagal.")
            QMessageBox.warning(
                self, "Analisis tidak dapat dijalankan",
                f"{exc}\n\nPeriksa tipe kolom, jumlah data, "
                "dan nilai target."
            )
        finally:
            QApplication.restoreOverrideCursor()
            self.run_analysis_button.setEnabled(True)

    def analyze_eda(self):
        df = self.df
        numeric = df.select_dtypes(include=np.number)

        lines = [
            "EXPLORATORY DATA ANALYSIS",
            "=" * 42,
            f"Jumlah baris: {len(df):,}",
            f"Jumlah kolom: {len(df.columns)}",
            f"Duplikat: {df.duplicated().sum():,}",
            f"Sel kosong: {df.isna().sum().sum():,}",
            "",
            "TIPE DATA",
        ]

        for col, dtype in df.dtypes.items():
            lines.append(f"- {col}: {dtype}")

        lines.extend(["", "NILAI KOSONG PER KOLOM"])
        missing = df.isna().sum().sort_values(ascending=False)

        for col, count in missing.items():
            pct = 100 * count / max(len(df), 1)
            lines.append(f"- {col}: {count:,} ({pct:.2f}%)")

        if not numeric.empty:
            lines.extend(["", "STATISTIK DESKRIPTIF"])
            lines.append(numeric.describe().round(3).to_string())

            self.analysis_chart.axes.clear()
            col = self.target_combo.currentText()
            if col not in numeric.columns:
                col = numeric.columns[0]

            values = numeric[col].dropna()
            self.analysis_chart.axes.hist(
                values, bins=30, color="#3478f6", alpha=0.85
            )
            self.analysis_chart.axes.set_title(
                f"Distribusi {col}"
            )
            self.analysis_chart.axes.set_xlabel(col)
            self.analysis_chart.axes.set_ylabel("Frekuensi")
            self.analysis_chart.axes.grid(True, alpha=0.2)
            self.analysis_chart.figure.tight_layout()
            self.analysis_chart.draw()

        return "\n".join(lines)

    def prepare_numeric_features(self, exclude_target=None):
        df = self.df.copy()
        numeric = df.select_dtypes(include=np.number).copy()

        if exclude_target in numeric.columns:
            numeric = numeric.drop(columns=[exclude_target])

        # Hilangkan kolom yang seluruh nilainya kosong dan kolom konstan.
        numeric = numeric.dropna(axis=1, how="all")
        numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]

        if numeric.empty:
            raise ValueError(
                "Tidak ditemukan fitur numerik yang cukup untuk model."
            )

        # Batasi ukuran dataset yang digunakan model agar aplikasi desktop
        # tetap responsif pada dataset yang sangat besar.
        if len(numeric) > 100_000:
            numeric = numeric.sample(100_000, random_state=42)

        return numeric

    def analyze_regression(self):
        df = self.df.copy()
        target = self.target_combo.currentText()

        if target not in df.columns:
            raise ValueError("Pilih kolom target yang valid.")

        y = pd.to_numeric(df[target], errors="coerce")
        X = self.prepare_numeric_features(exclude_target=target)

        data = X.join(y.rename("__target__")).dropna()
        X = data.drop(columns="__target__")
        y = data["__target__"]

        if len(data) < 20:
            raise ValueError(
                "Regresi memerlukan setidaknya 20 baris valid "
                "untuk evaluasi dasar."
            )

        if y.nunique() < 2:
            raise ValueError("Target tidak memiliki variasi yang cukup.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(
                max_depth=8, random_state=42
            ),
            "Random Forest": RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            "Extra Trees": ExtraTreesRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                random_state=42
            ),
        }

        if len(data) >= 100:
            models["SVR"] = make_pipeline(
                StandardScaler(), SVR()
            )

        results = []
        scores = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            r2 = r2_score(y_test, pred)

            results.append(
                (name, mae, rmse, r2, model)
            )
            scores[name] = rmse

        results.sort(key=lambda x: x[2])
        best = results[0]

        lines = [
            "REGRESSION MODEL COMPARISON",
            "=" * 42,
            f"Target: {target}",
            f"Baris valid: {len(data):,}",
            f"Fitur: {', '.join(X.columns)}",
            f"Data train: {len(X_train):,}",
            f"Data test: {len(X_test):,}",
            "",
            "METRIK (RMSE lebih kecil umumnya lebih baik)",
            "",
            f"{'Model':24} {'MAE':>12} {'RMSE':>12} {'R²':>10}",
        ]

        for name, mae, rmse, r2, _ in results:
            lines.append(
                f"{name:24} {mae:12.4f} {rmse:12.4f} {r2:10.4f}"
            )

        lines.extend([
            "",
            f"Model terbaik pada split ini: {best[0]}",
            f"MAE: {best[1]:.4f}",
            f"RMSE: {best[2]:.4f}",
            f"R²: {best[3]:.4f}",
            "",
            "Interpretasi:",
            "- MAE mengukur rata-rata besar kesalahan prediksi.",
            "- RMSE memberi penalti lebih besar pada kesalahan besar.",
            "- R² mengukur kecocokan model terhadap variasi target.",
            "- Skor negatif atau buruk dapat menunjukkan fitur "
            "belum cukup menjelaskan target.",
            "- Hasil ini bukan jaminan kinerja pada data masa depan.",
        ])

        self.analysis_chart.axes.clear()
        names = [x[0] for x in results]
        vals = [x[2] for x in results]
        self.analysis_chart.axes.barh(names[::-1], vals[::-1], color="#3478f6")
        self.analysis_chart.axes.set_title("Perbandingan RMSE")
        self.analysis_chart.axes.set_xlabel("RMSE (lebih kecil lebih baik)")
        self.analysis_chart.figure.tight_layout()
        self.analysis_chart.draw()

        return "\n".join(lines)

    def analyze_classification(self):
        df = self.df.copy()
        target = self.target_combo.currentText()

        if target not in df.columns:
            raise ValueError("Pilih kolom target yang valid.")

        y_raw = df[target]
        valid = y_raw.notna()
        y_raw = y_raw[valid]

        if y_raw.nunique() < 2 or y_raw.nunique() > 30:
            raise ValueError(
                "Target klasifikasi harus memiliki 2–30 kelas. "
                "Pilih kolom kategori yang sesuai."
            )

        X_all = self.prepare_numeric_features(exclude_target=target)
        X_all = X_all.loc[valid]
        common = X_all.index.intersection(y_raw.index)
        X_all = X_all.loc[common]
        y_raw = y_raw.loc[common]

        encoder = LabelEncoder()
        y = encoder.fit_transform(y_raw.astype(str))

        if len(y) < 30:
            raise ValueError(
                "Data terlalu sedikit untuk evaluasi klasifikasi yang layak."
            )

        class_counts = pd.Series(y).value_counts()
        if class_counts.min() < 2:
            raise ValueError(
                "Ada kelas dengan kurang dari 2 contoh. "
                "Data belum cukup untuk train/test stratified."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y, test_size=0.2, random_state=42,
            stratify=y
        )

        models = {
            "Logistic Regression": make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1500)
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=8, random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                random_state=42
            ),
        }

        if len(X_train) <= 20_000:
            models["SVM"] = make_pipeline(
                StandardScaler(), SVC()
            )

        results = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            acc = accuracy_score(y_test, pred)
            f1 = f1_score(
                y_test, pred, average="weighted", zero_division=0
            )
            results.append((name, acc, f1))

        results.sort(key=lambda x: x[2], reverse=True)

        lines = [
            "CLASSIFICATION MODEL COMPARISON",
            "=" * 42,
            f"Target: {target}",
            f"Jumlah kelas: {len(encoder.classes_)}",
            f"Label: {', '.join(encoder.classes_.astype(str))}",
            f"Data train: {len(X_train):,}",
            f"Data test: {len(X_test):,}",
            "",
            f"{'Model':24} {'Accuracy':>12} {'Weighted F1':>14}",
        ]

        for name, acc, f1 in results:
            lines.append(f"{name:24} {acc:12.4f} {f1:14.4f}")

        best = results[0]
        lines.extend([
            "",
            f"Model terbaik pada split ini: {best[0]}",
            f"Accuracy: {best[1]:.4f}",
            f"Weighted F1: {best[2]:.4f}",
            "",
            "Interpretasi:",
            "- Accuracy adalah proporsi prediksi yang benar.",
            "- Weighted F1 menimbang F1 berdasarkan jumlah contoh kelas.",
            "- Untuk data tidak seimbang, periksa precision, recall, "
            "confusion matrix, dan performa setiap kelas.",
            "- Hasil model tidak membuktikan hubungan sebab-akibat.",
        ])

        self.analysis_chart.axes.clear()
        self.analysis_chart.axes.bar(
            [x[0] for x in results],
            [x[2] for x in results],
            color="#3478f6"
        )
        self.analysis_chart.axes.set_title("Weighted F1 antar-model")
        self.analysis_chart.axes.set_ylabel("Weighted F1")
        self.analysis_chart.axes.tick_params(axis="x", rotation=25)
        self.analysis_chart.figure.tight_layout()
        self.analysis_chart.draw()

        return "\n".join(lines)

    def analyze_clustering(self):
        X = self.prepare_numeric_features()

        # Isi nilai kosong dan standardisasi fitur.
        X = X.replace([np.inf, -np.inf], np.nan)
        X = pd.DataFrame(
            SimpleImputer(strategy="median").fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        X_scaled = StandardScaler().fit_transform(X)

        k = self.cluster_spin.value()
        if len(X_scaled) <= k:
            raise ValueError("Jumlah baris harus lebih besar dari jumlah cluster.")

        # Batasi sampel evaluasi silhouette untuk dataset besar.
        if len(X_scaled) > 20_000:
            rng = np.random.RandomState(42)
            indices = rng.choice(
                len(X_scaled), size=20_000, replace=False
            )
            X_fit = X_scaled[indices]
        else:
            X_fit = X_scaled

        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)

        silhouette = None
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(
                X_fit,
                model.predict(X_fit)
            )

        counts = pd.Series(labels).value_counts().sort_index()

        lines = [
            "CUSTOMER / BUSINESS SEGMENTATION",
            "=" * 42,
            "Metode: K-Means",
            f"Jumlah cluster: {k}",
            f"Jumlah observasi: {len(X_scaled):,}",
            f"Fitur: {', '.join(X.columns)}",
            "",
            "JUMLAH OBSERVASI PER CLUSTER",
        ]

        for cluster, count in counts.items():
            lines.append(f"- Cluster {cluster}: {count:,} observasi")

        if silhouette is not None:
            lines.extend([
                "",
                f"Silhouette score: {silhouette:.4f}",
                "Nilai lebih tinggi umumnya berarti cluster lebih terpisah.",
            ])

        lines.extend([
            "",
            "CATATAN",
            "K-Means menemukan kelompok berdasarkan kemiripan fitur.",
            "Cluster belum otomatis berarti pelanggan bernilai tinggi/rendah.",
            "Periksa profil rata-rata tiap cluster untuk memberi makna bisnis.",
        ])

        # Profil cluster berdasarkan nilai fitur asli.
        profile = X.copy()
        profile["Cluster"] = labels
        profile = profile.groupby("Cluster").mean(numeric_only=True)

        lines.extend(["", "RATA-RATA FITUR PER CLUSTER"])
        lines.append(profile.round(3).to_string())

        self.analysis_chart.axes.clear()
        self.analysis_chart.axes.bar(
            [f"Cluster {i}" for i in counts.index],
            counts.values,
            color="#3478f6"
        )
        self.analysis_chart.axes.set_title("Ukuran setiap cluster")
        self.analysis_chart.setMinimumHeight(260)
        self.analysis_chart.figure.tight_layout()
        self.analysis_chart.draw()

        return "\n".join(lines)

    def analyze_forecasting(self):
        df = self.df.copy()
        date_col = self.date_combo.currentText()
        target = self.target_combo.currentText()

        if date_col in ("", "(Tidak digunakan)") or date_col not in df.columns:
            raise ValueError("Pilih kolom tanggal untuk forecasting.")

        if target not in df.columns:
            raise ValueError("Pilih kolom target numerik.")

        dates = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        values = pd.to_numeric(df[target], errors="coerce")

        data = pd.DataFrame({
            "date": dates,
            "value": values
        }).dropna()

        if len(data) < 30:
            raise ValueError(
                "Forecasting memerlukan minimal 30 observasi valid."
            )

        data = data.sort_values("date")
        data = data.groupby("date", as_index=False)["value"].sum()

        if len(data) < 20:
            raise ValueError(
                "Terlalu sedikit titik waktu unik setelah agregasi tanggal."
            )

        # Jadikan deret waktu dengan interval median antar tanggal.
        data = data.set_index("date")
        differences = data.index.to_series().diff().dropna()
        if differences.empty:
            raise ValueError("Tidak dapat menentukan interval waktu.")

        median_days = max(1, int(differences.dt.total_seconds().median() / 86400))
        freq = f"{median_days}D"
        series = data["value"].resample(freq).sum().dropna()

        if len(series) < 20:
            raise ValueError("Data deret waktu terlalu pendek setelah resampling.")

        # Fitur lag sederhana untuk forecasting.
        frame = pd.DataFrame({"y": series})
        for lag in (1, 2, 3, 7):
            frame[f"lag_{lag}"] = frame["y"].shift(lag)

        frame["rolling_mean_3"] = frame["y"].shift(1).rolling(3).mean()
        frame = frame.dropna()

        if len(frame) < 15:
            raise ValueError("Tidak cukup observasi untuk fitur lag.")

        X = frame.drop(columns="y")
        y = frame["y"]

        split = int(len(frame) * 0.8)
        if split < 10 or len(frame) - split < 3:
            raise ValueError("Data terlalu sedikit untuk evaluasi forecasting.")

        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))

        # Forecast rekursif menggunakan lag dari hasil prediksi terbaru.
        horizon = self.period_spin.value()
        history = list(series.values)
        future = []

        for _ in range(horizon):
            row = {}
            for lag in (1, 2, 3, 7):
                row[f"lag_{lag}"] = history[-lag] if len(history) >= lag else history[-1]

            row["rolling_mean_3"] = float(np.mean(history[-3:]))
            next_value = float(model.predict(pd.DataFrame([row])[X.columns])[0])
            future.append(next_value)
            history.append(next_value)

        step = pd.Timedelta(days=median_days)
        future_dates = [
            series.index[-1] + step * (i + 1)
            for i in range(horizon)
        ]

        lines = [
            "TIME-SERIES FORECASTING",
            "=" * 42,
            f"Kolom tanggal: {date_col}",
            f"Target: {target}",
            f"Jumlah titik waktu: {len(series):,}",
            f"Interval perkiraan: setiap {median_days} hari",
            f"Model: Random Forest dengan fitur lag",
            f"Train: {len(X_train):,}",
            f"Test: {len(X_test):,}",
            "",
            "EVALUASI PADA BAGIAN TEST TERAKHIR",
            f"MAE: {mae:.4f}",
            f"RMSE: {rmse:.4f}",
            "",
            f"FORECAST {horizon} PERIODE",
        ]

        for dt, value in zip(future_dates, future):
            lines.append(f"- {dt.strftime('%Y-%m-%d')}: {value:.4f}")

        lines.extend([
            "",
            "Catatan:",
            "Forecast ini merupakan baseline berbasis fitur lag, bukan "
            "jaminan hasil bisnis.",
            "Akurasi dapat menurun jika ada musim, tren, promosi, atau "
            "perubahan bisnis yang belum dimodelkan.",
            "Evaluasi menggunakan urutan waktu, bukan split acak.",
        ])

        self.analysis_chart.axes.clear()
        self.analysis_chart.axes.plot(
            series.index, series.values,
            label="Aktual", color="#3478f6"
        )
        self.analysis_chart.axes.plot(
            y_test.index, pred,
            label="Prediksi test", color="#e39b36"
        )
        self.analysis_chart.axes.plot(
            future_dates, future,
            label="Forecast", color="#28a276", linestyle="--"
        )
        self.analysis_chart.axes.set_title(f"Forecast {target}")
        self.analysis_chart.axes.legend()
        self.analysis_chart.axes.grid(True, alpha=0.2)
        self.analysis_chart.figure.autofmt_xdate()
        self.analysis_chart.figure.tight_layout()
        self.analysis_chart.draw()

        return "\n".join(lines)

    def analyze_anomalies(self):
        X = self.prepare_numeric_features()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = pd.DataFrame(
            SimpleImputer(strategy="median").fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        if len(X) < 20:
            raise ValueError(
                "Anomaly detection memerlukan setidaknya 20 baris."
            )

        if len(X) > 100_000:
            X = X.sample(100_000, random_state=42)

        model = make_pipeline(
            StandardScaler(),
            IsolationForest(
                contamination="auto", random_state=42
            )
        )
        labels = model.fit_predict(X)
        anomaly_mask = labels == -1
        count = int(anomaly_mask.sum())

        scores = model.decision_function(X)
        top_indices = np.argsort(scores)[:min(10, count)]

        lines = [
            "ANOMALY DETECTION",
            "=" * 42,
            "Metode: Isolation Forest",
            f"Observasi dianalisis: {len(X):,}",
            f"Anomali terdeteksi: {count:,}",
            f"Proporsi anomali: {100 * count / len(X):.2f}%",
            "",
            "Contoh indeks observasi yang paling tidak biasa:",
        ]

        for idx in X.index[top_indices]:
            lines.append(f"- Indeks data: {idx}")

        lines.extend([
            "",
            "Anomali bukan otomatis kesalahan atau kecurangan.",
            "Periksa transaksi tersebut dengan konteks bisnis sebelum bertindak.",
        ])

        self.analysis_chart.axes.clear()
        self.analysis_chart.axes.bar(
            ["Normal", "Anomali"],
            [int((~anomaly_mask).sum()), count],
            color=["#3478f6", "#e39b36"]
        )
        self.analysis_chart.axes.set_title("Hasil anomaly detection")
        self.analysis_chart.figure.tight_layout()
        self.analysis_chart.draw()

        return "\n".join(lines)

    def analyze_correlation(self):
        numeric = self.df.select_dtypes(include=np.number)

        if numeric.shape[1] < 2:
            raise ValueError(
                "Correlation analysis memerlukan minimal 2 kolom numerik."
            )

        corr = numeric.corr()
        pairs = []

        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                a = corr.columns[i]
                b = corr.columns[j]
                value = corr.iloc[i, j]
                if pd.notna(value):
                    pairs.append((a, b, float(value)))

        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        lines = [
            "CORRELATION ANALYSIS",
            "=" * 42,
            "Pasangan korelasi dengan nilai absolut terbesar:",
            "",
        ]

        for a, b, value in pairs[:20]:
            lines.append(f"- {a} ↔ {b}: {value:.4f}")

        lines.extend([
            "",
            "Korelasi menunjukkan keterkaitan linear, bukan sebab-akibat.",
            "Korelasi juga dapat dipengaruhi outlier, tren, dan data yang bias.",
        ])

        self.analysis_chart.axes.clear()
        image = self.analysis_chart.axes.imshow(
            corr.values, aspect="auto", vmin=-1, vmax=1,
            cmap="coolwarm"
        )
        self.analysis_chart.axes.set_xticks(range(len(corr.columns)))
        self.analysis_chart.axes.set_xticklabels(
            corr.columns, rotation=60, ha="right", fontsize=8
        )
        self.analysis_chart.axes.set_yticks(range(len(corr.columns)))
        self.analysis_chart.axes.set_yticklabels(
            corr.columns, fontsize=8
        )
        self.analysis_chart.axes.set_title("Correlation matrix")
        self.analysis_chart.figure.colorbar(
            image, ax=self.analysis_chart.axes, fraction=0.046, pad=0.04
        )
        self.analysis_chart.figure.tight_layout()
        self.analysis_chart.draw()

        return "\n".join(lines)

    # ---------------------------------------------------------
    # AI Advisor: UI lokal, belum tersambung ke model AI eksternal
    # ---------------------------------------------------------
    def create_ai_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        layout.addWidget(
            self.page_header(
                "AI Business Advisor",
                "Eksplorasi data dan susun pertanyaan bisnis."
            )
        )

        card = make_card()
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 16, 16, 16)

        self.ai_chat = QTextEdit()
        self.ai_chat.setReadOnly(True)
        self.ai_chat.setPlainText(
            "AutoBiz Advisor\n\n"
            "Panel ini belum terhubung ke layanan AI eksternal. "
            "Untuk sementara, gunakan ringkasan statistik dan analisis "
            "di menu Business Analysis."
        )

        self.ai_input = QLineEdit()
        self.ai_input.setPlaceholderText("Tulis pertanyaan bisnis...")
        self.ai_send_button = QPushButton("Kirim")
        self.ai_send_button.setObjectName("primary")
        self.ai_send_button.clicked.connect(self.send_local_ai_message)
        self.ai_input.returnPressed.connect(self.send_local_ai_message)

        row = QHBoxLayout()
        row.addWidget(self.ai_input, 1)
        row.addWidget(self.ai_send_button)

        card_layout.addWidget(self.ai_chat, 1)
        card_layout.addLayout(row)

        layout.addWidget(card, 1)
        return page

    def send_local_ai_message(self):
        question = self.ai_input.text().strip()
        if not question:
            return

        self.ai_chat.append(f"\nKamu: {question}")

        if self.df is None:
            answer = "Silakan impor dataset terlebih dahulu."
        else:
            answer = (
                f"Dataset berisi {len(self.df):,} baris dan "
                f"{len(self.df.columns)} kolom.\n"
                f"Sel kosong: {self.df.isna().sum().sum():,}.\n"
                "Ini adalah ringkasan otomatis lokal, bukan jawaban dari "
                "model AI. Untuk analisis lanjutan, buka Business Analysis."
            )

        self.ai_chat.append(f"\nAutoBiz: {answer}")
        self.ai_input.clear()

    # ---------------------------------------------------------
    # Reports dan Settings
    # ---------------------------------------------------------
    def create_reports_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        layout.addWidget(
            self.page_header(
                "Reports",
                "Simpan ringkasan dataset dan hasil analisis."
            )
        )

        card = make_card()
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 16, 16, 16)

        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setPlainText(
            "Laporan belum tersedia. Impor dataset dan jalankan analisis."
        )

        refresh_btn = QPushButton("Perbarui laporan")
        refresh_btn.clicked.connect(self.refresh_report)

        save_btn = QPushButton("Simpan laporan TXT")
        save_btn.setObjectName("primary")
        save_btn.clicked.connect(self.save_report)

        card_layout.addWidget(self.report_text, 1)
        card_layout.addWidget(refresh_btn)
        card_layout.addWidget(save_btn)

        layout.addWidget(card, 1)
        return page

    def refresh_report(self):
        if self.df is None:
            self.report_text.setPlainText("Belum ada dataset.")
            return

        df = self.df
        lines = [
            "AUTOBIZ DATA — DATASET REPORT",
            "=" * 45,
            f"File: {self.file_path or 'Dataset aktif'}",
            f"Baris: {len(df):,}",
            f"Kolom: {len(df.columns)}",
            f"Duplikat: {df.duplicated().sum():,}",
            f"Sel kosong: {df.isna().sum().sum():,}",
            "",
            "KOLOM",
        ]

        for col in df.columns:
            lines.append(
                f"- {col} | tipe: {df[col].dtype} | "
                f"kosong: {df[col].isna().sum():,}"
            )

        if self.last_analysis:
            lines.extend([
                "",
                "HASIL ANALISIS TERAKHIR",
                self.last_analysis
            ])

        self.report_text.setPlainText("\n".join(lines))

    def save_report(self):
        if self.df is None:
            QMessageBox.warning(
                self, "Tidak ada dataset", "Impor dataset terlebih dahulu."
            )
            return

        self.refresh_report()

        path, _ = QFileDialog.getSaveFileName(
            self, "Simpan laporan", "autobiz_report.txt",
            "Text file (*.txt)"
        )
        if not path:
            return

        try:
            if not path.lower().endswith(".txt"):
                path += ".txt"

            Path(path).write_text(
                self.report_text.toPlainText(),
                encoding="utf-8"
            )
            QMessageBox.information(
                self, "Laporan disimpan", path
            )
        except Exception as exc:
            QMessageBox.critical(self, "Gagal menyimpan laporan", str(exc))

    def create_settings_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        layout.addWidget(
            self.page_header(
                "Settings",
                "Informasi aplikasi dan dependensi."
            )
        )

        card = make_card()
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 18, 18, 18)

        info = QLabel(
            "AutoBiz Data — Business Growth Assistant\n\n"
            "Desktop application berbasis PySide6 dan Pandas.\n\n"
            "Fitur:\n"
            "• Import dan export CSV / Excel\n"
            "• Data cleaning dasar\n"
            "• Exploratory Data Analysis\n"
            "• Regression dan classification model comparison\n"
            "• Clustering dengan K-Means\n"
            "• Forecasting berbasis fitur lag\n"
            "• Anomaly detection dengan Isolation Forest\n"
            "• Correlation analysis\n"
            "• Laporan ringkas dataset dan analisis\n\n"
            "Catatan: hasil model perlu divalidasi sesuai konteks bisnis. "
            "AI eksternal belum diintegrasikan."
        )
        info.setWordWrap(True)
        info.setStyleSheet(
            "background:transparent;border:none;color:#254873;"
        )

        card_layout.addWidget(info)
        card_layout.addStretch()
        layout.addWidget(card, 1)
        return page


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)

    window = AutoBizData()
    window.show()

    sys.exit(app.exec())
