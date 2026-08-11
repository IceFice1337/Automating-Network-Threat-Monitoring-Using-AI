import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import joblib
import pandas as pd
from scapy.all import IP, TCP, UDP, rdpcap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QProgressBar,
)

try:
    from PySide6.QtWidgets import QProgressBar
except ImportError:
    from PySide6.QtWidgets import QProgressBar


APP_DIR = os.path.join(os.path.expanduser("~"), ".ai_network_monitor")
DATASET_FILE = os.path.join(APP_DIR, "training_dataset.csv")
MODEL_FILE = os.path.join(APP_DIR, "rf_model.joblib")
FEATURES_FILE = os.path.join(APP_DIR, "feature_columns.joblib")
ANALYSIS_EXPORT_FILE = os.path.join(APP_DIR, "last_analysis_results.csv")

FEATURE_COLUMNS = [
    "packet_count",
    "byte_count",
    "duration",
    "avg_packet_size",
    "packets_per_second",
    "bytes_per_second",
    "tcp_syn_count",
    "tcp_fin_count",
    "tcp_rst_count",
    "unique_src_ports",
    "unique_dst_ports",
    "protocol_tcp",
    "protocol_udp",
    "protocol_other",
]


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


@dataclass
class FlowStats:
    packet_count: int = 0
    byte_count: int = 0
    first_ts: float = 0.0
    last_ts: float = 0.0
    tcp_syn_count: int = 0
    tcp_fin_count: int = 0
    tcp_rst_count: int = 0
    src_ports: Optional[set] = None
    dst_ports: Optional[set] = None
    protocol: str = "OTHER"

    def __post_init__(self):
        if self.src_ports is None:
            self.src_ports = set()
        if self.dst_ports is None:
            self.dst_ports = set()


class PcapFeatureExtractor:
    def extract(self, pcap_path: str) -> pd.DataFrame:
        packets = rdpcap(pcap_path)
        flows: Dict[Tuple[str, str, int, int, str], FlowStats] = {}

        for pkt in packets:
            if IP not in pkt:
                continue

            ip = pkt[IP]
            src_ip = getattr(ip, "src", "0.0.0.0")
            dst_ip = getattr(ip, "dst", "0.0.0.0")
            protocol = "OTHER"
            src_port = 0
            dst_port = 0

            if TCP in pkt:
                protocol = "TCP"
                src_port = int(pkt[TCP].sport)
                dst_port = int(pkt[TCP].dport)
            elif UDP in pkt:
                protocol = "UDP"
                src_port = int(pkt[UDP].sport)
                dst_port = int(pkt[UDP].dport)

            key = (src_ip, dst_ip, src_port, dst_port, protocol)

            if key not in flows:
                flows[key] = FlowStats()
                flows[key].first_ts = float(pkt.time)
                flows[key].protocol = protocol

            flow = flows[key]
            flow.packet_count += 1
            flow.byte_count += int(len(pkt))
            flow.last_ts = float(pkt.time)
            flow.src_ports.add(src_port)
            flow.dst_ports.add(dst_port)

            if TCP in pkt:
                flags = int(pkt[TCP].flags)
                if flags & 0x02:
                    flow.tcp_syn_count += 1
                if flags & 0x01:
                    flow.tcp_fin_count += 1
                if flags & 0x04:
                    flow.tcp_rst_count += 1

        rows: List[dict] = []

        for (src_ip, dst_ip, src_port, dst_port, protocol), flow in flows.items():
            duration = max(flow.last_ts - flow.first_ts, 0.000001)
            avg_packet_size = flow.byte_count / max(flow.packet_count, 1)

            rows.append(
                {
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "src_port": src_port,
                    "dst_port": dst_port,
                    "protocol": protocol,
                    "packet_count": flow.packet_count,
                    "byte_count": flow.byte_count,
                    "duration": duration,
                    "avg_packet_size": avg_packet_size,
                    "packets_per_second": flow.packet_count / duration,
                    "bytes_per_second": flow.byte_count / duration,
                    "tcp_syn_count": flow.tcp_syn_count,
                    "tcp_fin_count": flow.tcp_fin_count,
                    "tcp_rst_count": flow.tcp_rst_count,
                    "unique_src_ports": len(flow.src_ports),
                    "unique_dst_ports": len(flow.dst_ports),
                    "protocol_tcp": 1 if protocol == "TCP" else 0,
                    "protocol_udp": 1 if protocol == "UDP" else 0,
                    "protocol_other": 1 if protocol == "OTHER" else 0,
                }
            )

        return pd.DataFrame(rows)


class ModelManager:
    def __init__(self):
        os.makedirs(APP_DIR, exist_ok=True)

    def append_training_data(self, features_df: pd.DataFrame, label: int) -> int:
        if features_df.empty:
            raise ValueError("No flows could be extracted from the selected file.")

        training_df = features_df.copy()
        training_df["label"] = label

        if os.path.exists(DATASET_FILE):
            existing_df = pd.read_csv(DATASET_FILE)
            merged_df = pd.concat([existing_df, training_df], ignore_index=True)
        else:
            merged_df = training_df

        merged_df.to_csv(DATASET_FILE, index=False)
        return len(merged_df)

    def get_dataset_info(self) -> str:
        if not os.path.exists(DATASET_FILE):
            return "Training dataset has not been created yet."

        df = pd.read_csv(DATASET_FILE)
        if df.empty:
            return "Training dataset is empty."

        normal_count = int((df["label"] == 0).sum())
        anomaly_count = int((df["label"] == 1).sum())

        return (
            f"Total rows: {len(df)} | "
            f"Normal: {normal_count} | "
            f"Anomaly: {anomaly_count}"
        )

    def clear_dataset(self):
        if os.path.exists(DATASET_FILE):
            os.remove(DATASET_FILE)

    def clear_model(self):
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)
        if os.path.exists(FEATURES_FILE):
            os.remove(FEATURES_FILE)

    def analyze(self, features_df: pd.DataFrame) -> pd.DataFrame:
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError("No trained model found. Train the model first.")

        if features_df.empty:
            raise ValueError("No flows could be extracted from the selected file.")

        model = joblib.load(MODEL_FILE)
        feature_columns = joblib.load(FEATURES_FILE)

        X = features_df[feature_columns]
        preds = model.predict(X)
        probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None

        result_df = features_df[
            ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]
        ].copy()

        result_df["prediction"] = preds
        result_df["prediction_text"] = result_df["prediction"].map(
            {0: "Normal", 1: "Anomaly"}
        )

        if probs is not None and probs.shape[1] >= 2:
            result_df["anomaly_probability"] = probs[:, 1]
        else:
            result_df["anomaly_probability"] = 0.0

        result_df = result_df.sort_values(by="anomaly_probability", ascending=False)
        result_df.to_csv(ANALYSIS_EXPORT_FILE, index=False)

        return result_df


class AddTrainingWorker(QThread):
    success = Signal(str)
    error = Signal(str)

    def __init__(self, extractor: PcapFeatureExtractor, manager: ModelManager, path: str, label: int):
        super().__init__()
        self.extractor = extractor
        self.manager = manager
        self.path = path
        self.label = label

    def run(self):
        try:
            features = self.extractor.extract(self.path)
            total = self.manager.append_training_data(features, self.label)
            label_name = "Normal" if self.label == 0 else "Anomaly"
            self.success.emit(f"File added as {label_name}. Total dataset rows: {total}")
        except Exception as exc:
            self.error.emit(str(exc))


class TrainModelWorker(QThread):
    progress = Signal(int, str, str)
    success = Signal(float, int, float)
    error = Signal(str)

    def __init__(self, dataset_file: str):
        super().__init__()
        self.dataset_file = dataset_file

    def run(self):
        try:
            if not os.path.exists(self.dataset_file):
                raise FileNotFoundError("No training dataset found. Add training files first.")

            df = pd.read_csv(self.dataset_file)

            if df.empty:
                raise ValueError("The training dataset is empty.")
            if "label" not in df.columns:
                raise ValueError("The training dataset does not contain a label column.")
            if df["label"].nunique() < 2:
                raise ValueError("At least two classes are required: normal and anomaly.")

            X = df[FEATURE_COLUMNS]
            y = df["label"]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y,
            )

            n_estimators = 200
            start_time = time.time()

            model = RandomForestClassifier(
                n_estimators=1,
                warm_start=True,
                random_state=42,
                n_jobs=1,
            )

            for i in range(1, n_estimators + 1):
                model.n_estimators = i
                model.fit(X_train, y_train)

                elapsed = time.time() - start_time
                avg_per_tree = elapsed / i
                remaining = avg_per_tree * (n_estimators - i)

                percent = int((i / n_estimators) * 100)
                elapsed_text = format_seconds(elapsed)
                remaining_text = format_seconds(remaining)

                self.progress.emit(
                    percent,
                    elapsed_text,
                    remaining_text,
                )

            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            joblib.dump(model, MODEL_FILE)
            joblib.dump(FEATURE_COLUMNS, FEATURES_FILE)

            total_elapsed = time.time() - start_time
            self.success.emit(accuracy, len(df), total_elapsed)

        except Exception as exc:
            self.error.emit(str(exc))


class AnalyzeWorker(QThread):
    success = Signal(object)
    error = Signal(str)

    def __init__(self, extractor: PcapFeatureExtractor, manager: ModelManager, path: str):
        super().__init__()
        self.extractor = extractor
        self.manager = manager
        self.path = path

    def run(self):
        try:
            features = self.extractor.extract(self.path)
            results = self.manager.analyze(features)
            self.success.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Network Threat Monitor")
        self.resize(1180, 800)

        self.extractor = PcapFeatureExtractor()
        self.manager = ModelManager()

        self.training_worker = None
        self.train_worker = None
        self.analysis_worker = None

        self.training_path_edit = QLineEdit()
        self.analysis_path_edit = QLineEdit()

        self.normal_radio = QRadioButton("Normal traffic")
        self.anomaly_radio = QRadioButton("Anomalous / attack traffic")
        self.normal_radio.setChecked(True)

        self.dataset_info_label = QLabel()
        self.analysis_summary_label = QLabel("")
        self.elapsed_time_label = QLabel("Elapsed: 0s")
        self.remaining_time_label = QLabel("Estimated remaining: --")
        self.training_progress_bar = QProgressBar()
        self.training_progress_bar.setRange(0, 100)
        self.training_progress_bar.setValue(0)

        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)

        self.analysis_table = QTableWidget()
        self.analysis_table.setColumnCount(7)
        self.analysis_table.setHorizontalHeaderLabels(
            [
                "Source IP",
                "Destination IP",
                "Source Port",
                "Destination Port",
                "Protocol",
                "Result",
                "Anomaly Score",
            ]
        )

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self._build_ui()
        self._refresh_dataset_info()

    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        tabs = QTabWidget()
        tabs.addTab(self._build_training_tab(), "Training")
        tabs.addTab(self._build_analysis_tab(), "Analysis")

        main_layout.addWidget(tabs)

    def _build_training_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        file_group = QGroupBox("Training file")
        file_layout = QGridLayout(file_group)

        file_layout.addWidget(QLabel("PCAP/PCAPNG file:"), 0, 0)
        file_layout.addWidget(self.training_path_edit, 1, 0)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_training_file)
        file_layout.addWidget(browse_button, 1, 1)

        layout.addWidget(file_group)

        label_group = QGroupBox("Training label")
        label_layout = QVBoxLayout(label_group)
        label_layout.addWidget(self.normal_radio)
        label_layout.addWidget(self.anomaly_radio)
        layout.addWidget(label_group)

        button_row = QHBoxLayout()

        add_button = QPushButton("Add file to training dataset")
        add_button.clicked.connect(self.add_training_file)

        train_button = QPushButton("Train model")
        train_button.clicked.connect(self.train_model)

        clear_dataset_button = QPushButton("Clear dataset")
        clear_dataset_button.clicked.connect(self.clear_dataset)

        clear_model_button = QPushButton("Delete model")
        clear_model_button.clicked.connect(self.delete_model)

        button_row.addWidget(add_button)
        button_row.addWidget(train_button)
        button_row.addWidget(clear_dataset_button)
        button_row.addWidget(clear_model_button)

        layout.addLayout(button_row)

        progress_group = QGroupBox("Training progress")
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.addWidget(self.training_progress_bar)
        progress_layout.addWidget(self.elapsed_time_label)
        progress_layout.addWidget(self.remaining_time_label)
        layout.addWidget(progress_group)

        dataset_group = QGroupBox("Training dataset status")
        dataset_layout = QVBoxLayout(dataset_group)
        dataset_layout.addWidget(self.dataset_info_label)
        layout.addWidget(dataset_group)

        log_group = QGroupBox("Training log")
        log_layout = QVBoxLayout(log_group)
        log_layout.addWidget(self.training_log)
        layout.addWidget(log_group)

        return tab

    def _build_analysis_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        file_group = QGroupBox("Analysis file")
        file_layout = QGridLayout(file_group)

        file_layout.addWidget(QLabel("PCAP/PCAPNG file:"), 0, 0)
        file_layout.addWidget(self.analysis_path_edit, 1, 0)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_analysis_file)
        file_layout.addWidget(browse_button, 1, 1)

        analyze_button = QPushButton("Analyze")
        analyze_button.clicked.connect(self.analyze_file)
        file_layout.addWidget(analyze_button, 2, 1)

        layout.addWidget(file_group)
        layout.addWidget(self.analysis_summary_label)
        layout.addWidget(self.analysis_table)

        return tab

    def browse_training_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a PCAP/PCAPNG file",
            "",
            "PCAP Files (*.pcap *.pcapng);;All Files (*.*)",
        )
        if file_path:
            self.training_path_edit.setText(file_path)

    def browse_analysis_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a PCAP/PCAPNG file",
            "",
            "PCAP Files (*.pcap *.pcapng);;All Files (*.*)",
        )
        if file_path:
            self.analysis_path_edit.setText(file_path)

    def add_training_file(self):
        path = self.training_path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "No file selected", "Please select a training file first.")
            return

        label = 0 if self.normal_radio.isChecked() else 1

        self.status_bar.showMessage("Extracting features from training file...")
        self.training_worker = AddTrainingWorker(self.extractor, self.manager, path, label)
        self.training_worker.success.connect(self.on_add_training_success)
        self.training_worker.error.connect(self.on_worker_error)
        self.training_worker.start()

    def on_add_training_success(self, message: str):
        self.training_log.append(message)
        self._refresh_dataset_info()
        self.status_bar.showMessage("Training file added successfully.", 5000)

    def train_model(self):
        self.training_progress_bar.setValue(0)
        self.elapsed_time_label.setText("Elapsed: 0s")
        self.remaining_time_label.setText("Estimated remaining: calculating...")
        self.status_bar.showMessage("Training model...")

        self.train_worker = TrainModelWorker(DATASET_FILE)
        self.train_worker.progress.connect(self.on_train_progress)
        self.train_worker.success.connect(self.on_train_success)
        self.train_worker.error.connect(self.on_worker_error)
        self.train_worker.start()

    def on_train_progress(self, percent: int, elapsed_text: str, remaining_text: str):
        self.training_progress_bar.setValue(percent)
        self.elapsed_time_label.setText(f"Elapsed: {elapsed_text}")
        self.remaining_time_label.setText(f"Estimated remaining: {remaining_text}")

    def on_train_success(self, accuracy: float, total_rows: int, total_elapsed: float):
        self.training_progress_bar.setValue(100)
        self.elapsed_time_label.setText(f"Elapsed: {format_seconds(total_elapsed)}")
        self.remaining_time_label.setText("Estimated remaining: 0s")

        message = (
            f"Model trained successfully. "
            f"Test accuracy: {accuracy:.4f}. "
            f"Total rows: {total_rows}. "
            f"Training time: {format_seconds(total_elapsed)}"
        )
        self.training_log.append(message)
        self.status_bar.showMessage("Model trained successfully.", 5000)
        QMessageBox.information(
            self,
            "Training complete",
            f"Model trained successfully.\n"
            f"Test accuracy: {accuracy:.4f}\n"
            f"Training time: {format_seconds(total_elapsed)}",
        )

    def analyze_file(self):
        path = self.analysis_path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "No file selected", "Please select an analysis file first.")
            return

        self.status_bar.showMessage("Extracting features and analyzing file...")
        self.analysis_worker = AnalyzeWorker(self.extractor, self.manager, path)
        self.analysis_worker.success.connect(self.on_analysis_success)
        self.analysis_worker.error.connect(self.on_worker_error)
        self.analysis_worker.start()

    def on_analysis_success(self, results: pd.DataFrame):
        self.populate_analysis_table(results)
        anomalies = int((results["prediction"] == 1).sum())
        total = len(results)

        self.analysis_summary_label.setText(
            f"Analyzed flows: {total}. Suspicious flows: {anomalies}. "
            f"Results saved to: {ANALYSIS_EXPORT_FILE}"
        )
        self.status_bar.showMessage("Analysis completed successfully.", 5000)

    def on_worker_error(self, error_message: str):
        self.status_bar.showMessage("Operation failed.", 5000)
        QMessageBox.critical(self, "Error", error_message)

    def populate_analysis_table(self, results: pd.DataFrame):
        top_rows = results.head(300).reset_index(drop=True)
        self.analysis_table.setRowCount(len(top_rows))

        for row_index, (_, row) in enumerate(top_rows.iterrows()):
            values = [
                str(row["src_ip"]),
                str(row["dst_ip"]),
                str(row["src_port"]),
                str(row["dst_port"]),
                str(row["protocol"]),
                str(row["prediction_text"]),
                f"{row['anomaly_probability']:.4f}",
            ]

            for col_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.analysis_table.setItem(row_index, col_index, item)

        self.analysis_table.resizeColumnsToContents()

    def _refresh_dataset_info(self):
        self.dataset_info_label.setText(self.manager.get_dataset_info())

    def clear_dataset(self):
        reply = QMessageBox.question(
            self,
            "Confirm",
            "Delete the training dataset?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.manager.clear_dataset()
            self._refresh_dataset_info()
            self.training_log.append("Training dataset deleted.")
            self.training_progress_bar.setValue(0)
            self.elapsed_time_label.setText("Elapsed: 0s")
            self.remaining_time_label.setText("Estimated remaining: --")
            self.status_bar.showMessage("Training dataset deleted.", 5000)

    def delete_model(self):
        reply = QMessageBox.question(
            self,
            "Confirm",
            "Delete the trained model?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.manager.clear_model()
            self.training_log.append("Trained model deleted.")
            self.status_bar.showMessage("Trained model deleted.", 5000)


def main():
    os.makedirs(APP_DIR, exist_ok=True)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()