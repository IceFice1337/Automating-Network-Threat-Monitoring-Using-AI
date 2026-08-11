import pandas as pd
from pathlib import Path
from collections import defaultdict

from scapy.all import rdpcap, IP, TCP, UDP

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "packet_count",
    "byte_count",
    "duration",
    "avg_packet_size",
    "packets_per_second",
    "bytes_per_second",
    "tcp_syn_count",
    "tcp_fin_count",
    "tcp_rst_count"
]


def extract_features_from_pcap(pcap_path):
    print(f"Extracting features from: {pcap_path}")

    packets = rdpcap(str(pcap_path))
    flows = defaultdict(list)

    for pkt in packets:
        if IP not in pkt:
            continue

        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        protocol = "OTHER"
        src_port = 0
        dst_port = 0

        if TCP in pkt:
            protocol = "TCP"
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
        elif UDP in pkt:
            protocol = "UDP"
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport

        flow_key = (src_ip, dst_ip, src_port, dst_port, protocol)
        flows[flow_key].append(pkt)

    rows = []

    for (src_ip, dst_ip, src_port, dst_port, protocol), pkts in flows.items():
        times = [float(pkt.time) for pkt in pkts]
        lengths = [len(pkt) for pkt in pkts]

        packet_count = len(pkts)
        byte_count = sum(lengths)

        if len(times) > 1:
            duration = max(times) - min(times)
        else:
            duration = 0.000001

        avg_packet_size = byte_count / packet_count if packet_count > 0 else 0
        packets_per_second = packet_count / duration if duration > 0 else 0
        bytes_per_second = byte_count / duration if duration > 0 else 0

        tcp_syn_count = 0
        tcp_fin_count = 0
        tcp_rst_count = 0

        for pkt in pkts:
            if TCP in pkt:
                flags = pkt[TCP].flags

                if flags & 0x02:
                    tcp_syn_count += 1
                if flags & 0x01:
                    tcp_fin_count += 1
                if flags & 0x04:
                    tcp_rst_count += 1

        rows.append({
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": protocol,
            "packet_count": packet_count,
            "byte_count": byte_count,
            "duration": duration,
            "avg_packet_size": avg_packet_size,
            "packets_per_second": packets_per_second,
            "bytes_per_second": bytes_per_second,
            "tcp_syn_count": tcp_syn_count,
            "tcp_fin_count": tcp_fin_count,
            "tcp_rst_count": tcp_rst_count
        })

    return pd.DataFrame(rows)


def ask_files(title, allowed_extensions):
    print(f"\n{title}")
    print("Add files one by one.")
    print("Press ENTER on empty line when finished.\n")

    files = []

    while True:
        path = input("File path: ").strip().replace('"', "")

        if path == "":
            break

        file_path = Path(path)

        if not file_path.exists():
            print("File not found. Try again.")
            continue

        if file_path.suffix.lower() not in allowed_extensions:
            print(f"Unsupported file type: {file_path.suffix}")
            print(f"Allowed: {', '.join(allowed_extensions)}")
            continue

        files.append(file_path)
        print(f"Added: {file_path}")

    return files


def load_training_data(training_files):
    dataframes = []

    for file in training_files:
        suffix = file.suffix.lower()

        if suffix == ".csv":
            print(f"Loading training CSV: {file}")
            df = pd.read_csv(file)
        elif suffix in [".pcap", ".pcapng"]:
            df = extract_features_from_pcap(file)
        else:
            print(f"Skipping unsupported file: {file}")
            continue

        missing = [col for col in FEATURE_COLUMNS if col not in df.columns]

        if missing:
            print(f"Skipping file because required columns are missing: {missing}")
            continue

        dataframes.append(df[FEATURE_COLUMNS].fillna(0))

    if not dataframes:
        raise ValueError("No valid training data was loaded.")

    combined = pd.concat(dataframes, ignore_index=True)
    return combined


def train_isolation_forest(training_files):
    X = load_training_data(training_files)

    print(f"\nTraining rows: {len(X)}")
    print("Scaling features...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    contamination_input = input("\nContamination [default 0.02]: ").strip()
    contamination = float(contamination_input) if contamination_input else 0.02

    trees_input = input("Number of trees [default 200]: ").strip()
    n_estimators = int(trees_input) if trees_input else 200

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        random_state=42
    )

    print("\nTraining Isolation Forest model...")
    model.fit(X_scaled)

    print("Training completed.")

    return model, scaler


def analyze_file(model, scaler, pcap_path):
    df = extract_features_from_pcap(pcap_path)

    if df.empty:
        return {
            "file": pcap_path.name,
            "total_flows": 0,
            "anomalies": 0,
            "normal": 0,
            "anomaly_percent": 0,
            "saved_to": ""
        }

    X = df[FEATURE_COLUMNS].fillna(0)
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)

    df["isolation_forest_result"] = [
        "Anomaly" if pred == -1 else "Normal"
        for pred in predictions
    ]

    df["isolation_forest_score"] = scores

    anomaly_count = (df["isolation_forest_result"] == "Anomaly").sum()
    normal_count = (df["isolation_forest_result"] == "Normal").sum()
    total_flows = len(df)

    anomaly_percent = (anomaly_count / total_flows) * 100 if total_flows > 0 else 0

    output_csv = pcap_path.with_name(pcap_path.stem + "_isolation_forest_results.csv")
    df.to_csv(output_csv, index=False)

    return {
        "file": pcap_path.name,
        "total_flows": total_flows,
        "anomalies": anomaly_count,
        "normal": normal_count,
        "anomaly_percent": round(anomaly_percent, 2),
        "saved_to": str(output_csv)
    }


def print_results_table(results):
    print("\nIsolation Forest Analysis Results")
    print("-" * 90)
    print(f"{'File':<28} {'Total flows':<15} {'Anomalies':<15} {'Normal':<15} {'Anomaly %':<10}")
    print("-" * 90)

    for r in results:
        print(
            f"{r['file']:<28} "
            f"{r['total_flows']:<15} "
            f"{r['anomalies']:<15} "
            f"{r['normal']:<15} "
            f"{r['anomaly_percent']:<10}"
        )

    print("-" * 90)


def main():
    print("Isolation Forest PCAP Analyzer")

    training_files = ask_files(
        title="Select TRAINING files (.pcap, .pcapng, or .csv)",
        allowed_extensions=[".pcap", ".pcapng", ".csv"]
    )

    if not training_files:
        print("No training files selected. Exiting.")
        return

    model, scaler = train_isolation_forest(training_files)

    analysis_files = ask_files(
        title="Select PCAP files for ANALYSIS (.pcap or .pcapng)",
        allowed_extensions=[".pcap", ".pcapng"]
    )

    if not analysis_files:
        print("No analysis files selected. Exiting.")
        return

    results = []

    for pcap_file in analysis_files:
        print(f"\nAnalyzing: {pcap_file}")
        result = analyze_file(model, scaler, pcap_file)
        results.append(result)

    print_results_table(results)

    print("\nDetailed CSV files saved:")
    for r in results:
        if r["saved_to"]:
            print(r["saved_to"])


if __name__ == "__main__":
    main()