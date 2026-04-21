# Real-Time Multi-Threaded Financial Transaction Analyzer with Fraud Detection

A robust, multi-threaded Python application designed to simulate and process streams of financial transactions, detecting fraudulent and anomalous behavior in real-time.

## Features

* **Data Generator**: Simulates a high-volume financial transaction stream.
* **Multi-Threaded Processing**: Utilizes `ThreadPoolExecutor` and thread-safe queues to concurrently process transactions.
* **Advanced Fraud Detection**:
  * High-value transaction thresholding
  * Rapid successive transaction detection
  * Location anomaly (impossible travel) detection
  * Statistical anomaly detection using standard deviation (z-score)
* **Performance Benchmarking**: Automatically compares single-threaded vs multi-threaded execution times.
* **Interactive Dashboard (Bonus)**: Includes a full `Streamlit` UI to visually monitor transactions and fraud alerts in real-time.

## Requirements

* Python 3.8+
* `streamlit` (Optional, for the dashboard)
* `pandas` (Optional, for the dashboard)

```bash
pip install streamlit pandas
```

## Running the CLI Analyzer

You can run the core multi-threaded analyzer via the CLI. It will benchmark both single and multi-threaded performance.

```bash
# Run with default settings (20,000 transactions, 4 threads)
python main.py

# Run a large scale simulation with custom thread counts and export results to CSV
python main.py --count 100000 --threads 8 --export
```

### CLI Arguments

* `--count`: Number of simulated transactions to generate (default: 20000).
* `--threads`: Number of worker threads for multi-threaded processing (default: 4).
* `--export`: Flag to export all flagged fraudulent transactions to `flagged_transactions.csv`.

## Running the Interactive Dashboard

To launch the real-time Streamlit dashboard:

```bash
streamlit run app.py
```

This will open a web interface where you can configure the batch size and thread count, trigger live transaction generation, and view real-time metrics and dynamic graphs of the fraud detection system.

## Project Structure

* `generator.py` - Contains `DataGenerator` for producing the simulated financial data.
* `worker.py` - Contains the `WorkManager` and worker threads pulling from queues.
* `analyzer.py` - Contains `FraudAnalyzer` with thread-safe locks and anomaly detection logic.
* `main.py` - The CLI entry point and performance benchmarking script.
* `app.py` - The Streamlit interactive dashboard.
