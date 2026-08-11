import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
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
    QCheckBox,
    QPlainTextEdit,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


def normalize_protocol(value: str) -> str:
    if pd.isna(value):
        return "OTHER"
    value = str(value).strip().upper()
    if value in {"TCP", "UDP", "ICMP"}:
        return value
    return value if value else "OTHER"


def safe_int(value) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(float(value))
    except Exception:
        return 0


class CompareEngine:
    def load_ml_results(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        required = {"src_ip", "dst_ip", "src_port", "dst_port", "protocol", "prediction"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"ML results file is missing columns: {missing}")

        df = df.copy()
        df["src_ip"] = df["src_ip"].astype(str)
        df["dst_ip"] = df["dst_ip"].astype(str)
        df["src_port"] = df["src_port"].apply(safe_int)
        df["dst_port"] = df["dst_port"].apply(safe_int)
        df["protocol"] = df["protocol"].apply(normalize_protocol)
        df["ml_predicted_anomaly"] = df["prediction"].apply(lambda x: 1 if int(x) == 1 else 0)

        if "anomaly_probability" not in df.columns:
            df["anomaly_probability"] = 0.0

        grouped = (
            df.groupby(["src_ip", "dst_ip", "src_port", "dst_port", "protocol"], as_index=False)
            .agg(
                ml_predicted_anomaly=("ml_predicted_anomaly", "max"),
                ml_anomaly_score=("anomaly_probability", "max"),
            )
        )
        return grouped

    def load_ground_truth(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
        required = {"id.orig_h", "id.resp_h", "id.orig_p", "id.resp_p", "proto", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Ground truth file is missing columns: {missing}")

        df = df.copy()
        df["src_ip"] = df["id.orig_h"].astype(str)
        df["dst_ip"] = df["id.resp_h"].astype(str)
        df["src_port"] = df["id.orig_p"].apply(safe_int)
        df["dst_port"] = df["id.resp_p"].apply(safe_int)
        df["protocol"] = df["proto"].apply(normalize_protocol)
        df["gt_is_anomaly"] = df["label"].astype(str).str.strip().str.lower().ne("benign").astype(int)

        grouped = (
            df.groupby(["src_ip", "dst_ip", "src_port", "dst_port", "protocol"], as_index=False)
            .agg(
                gt_is_anomaly=("gt_is_anomaly", "max"),
                gt_label=("label", "first"),
            )
        )
        return grouped

    def load_suricata_alerts(self, eve_path: str) -> pd.DataFrame:
        records = []
        with open(eve_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event_type") != "alert":
                    continue

                records.append(
                    {
                        "src_ip": str(event.get("src_ip", "")),
                        "dst_ip": str(event.get("dest_ip", "")),
                        "src_port": safe_int(event.get("src_port", 0)),
                        "dst_port": safe_int(event.get("dest_port", 0)),
                        "protocol": normalize_protocol(event.get("proto", "OTHER")),
                        "suricata_alert": 1,
                        "suricata_signature": event.get("alert", {}).get("signature", ""),
                    }
                )

        if not records:
            return pd.DataFrame(
                columns=[
                    "src_ip",
                    "dst_ip",
                    "src_port",
                    "dst_port",
                    "protocol",
                    "suricata_alert",
                    "suricata_signature",
                ]
            )

        df = pd.DataFrame(records)
        grouped = (
            df.groupby(["src_ip", "dst_ip", "src_port", "dst_port", "protocol"], as_index=False)
            .agg(
                suricata_alert=("suricata_alert", "max"),
                suricata_signature=("suricata_signature", "first"),
            )
        )
        return grouped

    def compute_metrics(self, pred_col: str, truth_col: str, df: pd.DataFrame) -> dict:
        tp = int(((df[pred_col] == 1) & (df[truth_col] == 1)).sum())
        fp = int(((df[pred_col] == 1) & (df[truth_col] == 0)).sum())
        tn = int(((df[pred_col] == 0) & (df[truth_col] == 0)).sum())
        fn = int(((df[pred_col] == 0) & (df[truth_col] == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(df) if len(df) > 0 else 0.0

        return {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "Accuracy": round(accuracy, 4),
        }

    def run_suricata(self, suricata_bin: str, pcap_path: str, output_dir: str) -> str:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        eve_json = output_path / "eve.json"
        if eve_json.exists():
            eve_json.unlink()

        cmd = [suricata_bin, "-r", pcap_path, "-l", str(output_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "Suricata failed.\n\n"
                f"STDOUT:\n{result.stdout}\n\n"
                f"STDERR:\n{result.stderr}"
            )
        if not eve_json.exists():
            raise FileNotFoundError(f"Suricata finished, but eve.json was not found: {eve_json}")
        return str(eve_json)

    def compare(self, ml_results_file: str, eve_file: str, output_dir: str, gt_file: str = ""):
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        summary_file = output_dir_path / "comparison_summary.csv"
        details_file = output_dir_path / "comparison_details.csv"

        ml_df = self.load_ml_results(ml_results_file)
        suri_df = self.load_suricata_alerts(eve_file)
        use_gt = bool(gt_file)

        if use_gt:
            gt_df = self.load_ground_truth(gt_file)
            merged = gt_df.merge(
                ml_df,
                on=["src_ip", "dst_ip", "src_port", "dst_port", "protocol"],
                how="left",
            ).merge(
                suri_df,
                on=["src_ip", "dst_ip", "src_port", "dst_port", "protocol"],
                how="left",
            )
            merged["ml_predicted_anomaly"] = merged["ml_predicted_anomaly"].fillna(0).astype(int)
            merged["ml_anomaly_score"] = merged["ml_anomaly_score"].fillna(0.0)
            merged["suricata_alert"] = merged["suricata_alert"].fillna(0).astype(int)
            merged["suricata_signature"] = merged["suricata_signature"].fillna("")

            ml_metrics = self.compute_metrics("ml_predicted_anomaly", "gt_is_anomaly", merged)
            suri_metrics = self.compute_metrics("suricata_alert", "gt_is_anomaly", merged)

            total_flows = len(merged)
            true_anomalies = int((merged["gt_is_anomaly"] == 1).sum())
            true_benign = int((merged["gt_is_anomaly"] == 0).sum())
            ml_detected = int((merged["ml_predicted_anomaly"] == 1).sum())
            suri_detected = int((merged["suricata_alert"] == 1).sum())

            summary_df = pd.DataFrame(
                [
                    {
                        "Method": "ML Model",
                        "Total Flows": total_flows,
                        "True Anomalies": true_anomalies,
                        "True Benign": true_benign,
                        "Detected Anomalies": ml_detected,
                        **ml_metrics,
                    },
                    {
                        "Method": "Suricata",
                        "Total Flows": total_flows,
                        "True Anomalies": true_anomalies,
                        "True Benign": true_benign,
                        "Detected Anomalies": suri_detected,
                        **suri_metrics,
                    },
                ]
            )
        else:
            merged = ml_df.merge(
                suri_df,
                on=["src_ip", "dst_ip", "src_port", "dst_port", "protocol"],
                how="outer",
            )
            merged["ml_predicted_anomaly"] = merged["ml_predicted_anomaly"].fillna(0).astype(int)
            merged["ml_anomaly_score"] = merged["ml_anomaly_score"].fillna(0.0)
            merged["suricata_alert"] = merged["suricata_alert"].fillna(0).astype(int)
            merged["suricata_signature"] = merged["suricata_signature"].fillna("")

            summary_df = pd.DataFrame(
                [
                    {
                        "Method": "ML Model",
                        "Detected Anomalies": int((merged["ml_predicted_anomaly"] == 1).sum()),
                        "Ground Truth Available": "No",
                    },
                    {
                        "Method": "Suricata",
                        "Detected Anomalies": int((merged["suricata_alert"] == 1).sum()),
                        "Ground Truth Available": "No",
                    },
                ]
            )

        summary_df.to_csv(summary_file, index=False)
        merged.to_csv(details_file, index=False)
        return summary_df, str(summary_file), str(details_file)


class CompareWorker(QThread):
    success = Signal(object, str, str)
    error = Signal(str)
    log = Signal(str)

    def __init__(self, ml_csv: str, pcap_path: str, output_dir: str, suricata_path: str, gt_path: str = "", skip_suricata: bool = False):
        super().__init__()
        self.engine = CompareEngine()
        self.ml_csv = ml_csv
        self.pcap_path = pcap_path
        self.output_dir = output_dir
        self.suricata_path = suricata_path
        self.gt_path = gt_path
        self.skip_suricata = skip_suricata

    def run(self):
        try:
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            eve_path = output_dir / "eve.json"

            if self.skip_suricata:
                if not eve_path.exists():
                    raise FileNotFoundError(f"Skip Suricata is enabled, but eve.json was not found in: {eve_path}")
                self.log.emit(f"Using existing Suricata file: {eve_path}")
            else:
                suri_bin = self.suricata_path.strip()
                if not suri_bin:
                    suri_bin = r"C:\Program Files\Suricata\suricata.exe"
                if not Path(suri_bin).exists():
                    if shutil.which(suri_bin) is None:
                        raise FileNotFoundError(f"Suricata executable not found: {suri_bin}")
                self.log.emit("Running Suricata on the selected PCAP...")
                eve_file = self.engine.run_suricata(suri_bin, self.pcap_path, self.output_dir)
                eve_path = Path(eve_file)
                self.log.emit(f"Suricata finished: {eve_path}")

            summary_df, summary_file, details_file = self.engine.compare(
                self.ml_csv,
                str(eve_path),
                self.output_dir,
                self.gt_path.strip(),
            )
            self.success.emit(summary_df, summary_file, details_file)
        except Exception as exc:
            self.error.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML vs Suricata Comparison")
        self.resize(1050, 720)
        self.worker = None

        self.ml_csv_edit = QLineEdit(r"C:\Users\user\.ai_network_monitor\last_analysis_results.csv")
        self.pcap_edit = QLineEdit()
        self.output_dir_edit = QLineEdit(str(Path.home() / "Desktop" / "Thesis"))
        self.suricata_edit = QLineEdit(r"C:\Program Files\Suricata\suricata.exe")
        self.gt_edit = QLineEdit()
        self.skip_suricata_checkbox = QCheckBox("Skip Suricata run and use existing eve.json from the output folder")

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)

        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(0)
        self.summary_table.setRowCount(0)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        files_group = QGroupBox("Files and settings")
        files_layout = QGridLayout(files_group)

        self._add_file_row(files_layout, 0, "ML results CSV:", self.ml_csv_edit, self.browse_ml_csv)
        self._add_file_row(files_layout, 1, "Analyzed PCAP:", self.pcap_edit, self.browse_pcap)
        self._add_file_row(files_layout, 2, "Output folder:", self.output_dir_edit, self.browse_output_dir)
        self._add_file_row(files_layout, 3, "Suricata executable:", self.suricata_edit, self.browse_suricata)
        self._add_file_row(files_layout, 4, "conn.log.labeled (optional):", self.gt_edit, self.browse_gt)

        files_layout.addWidget(self.skip_suricata_checkbox, 5, 0, 1, 3)
        layout.addWidget(files_group)

        buttons = QHBoxLayout()
        run_button = QPushButton("Run comparison")
        run_button.clicked.connect(self.run_comparison)
        buttons.addWidget(run_button)
        layout.addLayout(buttons)

        summary_group = QGroupBox("Summary")
        summary_layout = QVBoxLayout(summary_group)
        summary_layout.addWidget(self.summary_table)
        layout.addWidget(summary_group)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        log_layout.addWidget(self.log_box)
        layout.addWidget(log_group)

    def _add_file_row(self, grid, row, label_text, line_edit, browse_callback):
        grid.addWidget(QLabel(label_text), row, 0)
        grid.addWidget(line_edit, row, 1)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(browse_callback)
        grid.addWidget(browse_button, row, 2)

    def browse_ml_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select ML results CSV", "", "CSV Files (*.csv);;All Files (*.*)")
        if path:
            self.ml_csv_edit.setText(path)

    def browse_pcap(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select analyzed PCAP", "", "PCAP Files (*.pcap *.pcapng);;All Files (*.*)")
        if path:
            self.pcap_edit.setText(path)

    def browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self.output_dir_edit.setText(path)

    def browse_suricata(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select suricata executable", "", "Executable (*.exe);;All Files (*.*)")
        if path:
            self.suricata_edit.setText(path)

    def browse_gt(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select conn.log.labeled", "", "All Files (*.*)")
        if path:
            self.gt_edit.setText(path)

    def append_log(self, message: str):
        self.log_box.appendPlainText(message)

    def run_comparison(self):
        ml_csv = self.ml_csv_edit.text().strip()
        pcap = self.pcap_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()
        suricata = self.suricata_edit.text().strip()
        gt = self.gt_edit.text().strip()
        skip_suricata = self.skip_suricata_checkbox.isChecked()

        if not ml_csv or not Path(ml_csv).exists():
            QMessageBox.warning(self, "Missing file", "Please select a valid ML results CSV file.")
            return
        if not pcap or not Path(pcap).exists():
            QMessageBox.warning(self, "Missing file", "Please select a valid PCAP file.")
            return
        if not output_dir:
            QMessageBox.warning(self, "Missing folder", "Please select an output folder.")
            return
        if not skip_suricata and not suricata:
            QMessageBox.warning(self, "Missing file", "Please select the Suricata executable.")
            return
        if gt and not Path(gt).exists():
            QMessageBox.warning(self, "Missing file", "Ground truth file does not exist.")
            return

        self.summary_table.clear()
        self.summary_table.setRowCount(0)
        self.summary_table.setColumnCount(0)
        self.log_box.clear()
        self.status_bar.showMessage("Running comparison...")

        self.worker = CompareWorker(ml_csv, pcap, output_dir, suricata, gt, skip_suricata)
        self.worker.log.connect(self.append_log)
        self.worker.success.connect(self.on_success)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_success(self, summary_df: pd.DataFrame, summary_file: str, details_file: str):
        self.populate_summary_table(summary_df)
        self.append_log(f"Summary saved to: {summary_file}")
        self.append_log(f"Details saved to: {details_file}")
        self.status_bar.showMessage("Comparison completed.", 5000)
        QMessageBox.information(
            self,
            "Done",
            f"Comparison completed.\n\nSummary: {summary_file}\nDetails: {details_file}",
        )

    def on_error(self, message: str):
        self.status_bar.showMessage("Comparison failed.", 5000)
        QMessageBox.critical(self, "Error", message)

    def populate_summary_table(self, df: pd.DataFrame):
        self.summary_table.setColumnCount(len(df.columns))
        self.summary_table.setRowCount(len(df))
        self.summary_table.setHorizontalHeaderLabels([str(c) for c in df.columns])

        for row_idx, (_, row) in enumerate(df.iterrows()):
            for col_idx, col_name in enumerate(df.columns):
                value = str(row[col_name])
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.summary_table.setItem(row_idx, col_idx, item)

        self.summary_table.resizeColumnsToContents()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
