import argparse
import time
import csv
from generator import DataGenerator
from analyzer import FraudAnalyzer
from worker import WorkManager
from ml_trainer import MLTrainer

def _print_results(analyzer, execution_time, thread_type):
    print(f"\n--- {thread_type} Execution ---")
    print(f"Execution Time: {execution_time:.4f} seconds")
    summary = analyzer.get_summary()
    print(f"Total Processed: {summary['total_processed']}")
    print(f"Total Flagged:   {summary['total_flagged']} ({summary['fraud_rate']})")
    print(f"Amount Mean:     ${summary['mean_amount']}")
    print(f"Amount Std Dev:  ${summary['std_dev_amount']}")

def export_to_csv(analyzer, filename="results.csv"):
    if not analyzer.flagged_transactions:
        print("No flagged transactions to export.")
        return
        
    print(f"\nExporting flagged transactions to {filename}...")
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Transaction ID", "User ID", "Amount", "Location", "Timestamp", "Risk Score", "Reason"])
        for item in analyzer.flagged_transactions:
            tx = item["tx"]
            analysis = item["analysis"]
            writer.writerow([
                tx["transaction_id"],
                tx["user_id"],
                f"{tx['amount']:.2f}",
                tx["location"],
                tx["timestamp"],
                analysis["risk_score"],
                analysis["reason"]
            ])
    print("Export complete.")

def main():
    parser = argparse.ArgumentParser(description="Real-Time Multi-Threaded Financial Transaction Analyzer")
    parser.add_argument("--count", type=int, default=20000, help="Number of simulation transactions to generate")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for multi-threaded processing")
    parser.add_argument("--export", action="store_true", help="Export flagged transactions to CSV")
    parser.add_argument("--ml", action="store_true", help="Enable Machine Learning (Isolation Forest) Anomaly Detection")
    
    args = parser.parse_args()
    
    print(f"Generating {args.count} transactions...")
    # Using a slightly small user pool to ensure multiple repeat transactions from the same user (triggering anomaly rules)
    gen = DataGenerator(num_users=max(10, args.count // 100)) 
    transactions = gen.generate_batch(args.count)
    print("Generation complete.\n")
    
    # --- ML Training ---
    ml_model = None
    if args.ml:
        # Use first 2000 transactions (or up to count) to train
        train_pool = transactions[:min(2000, len(transactions))]
        ml_model = MLTrainer.train_isolation_forest(train_pool)
    
    # --- Single Threaded Evaluation ---
    analyzer_single = FraudAnalyzer(ml_model=ml_model)
    start_time = time.time()
    WorkManager.process_single_threaded(transactions, analyzer_single)
    single_time = time.time() - start_time
    _print_results(analyzer_single, single_time, "Single-Threaded")

    # --- Multi Threaded Evaluation ---
    analyzer_multi = FraudAnalyzer(ml_model=ml_model)
    start_time = time.time()
    WorkManager.process_multi_threaded(transactions, analyzer_multi, thread_count=args.threads)
    multi_time = time.time() - start_time
    _print_results(analyzer_multi, multi_time, f"Multi-Threaded ({args.threads} Threads)")
    
    print("\n--- Performance Comparison ---")
    if multi_time < single_time:
        speedup = single_time / multi_time
        print(f"Multi-threading was {speedup:.2f}x FASTER than single-threading.")
    else:
        slowdown = multi_time / single_time
        print(f"Multi-threading was {slowdown:.2f}x SLOWER than single-threading.")
        print("(Note: In basic CPU-bound Python loop, multi-threading might be slower due to GIL lock overhead.)")

    # Print a few sample flagged items
    print("\n--- Sample Flagged Transactions ---")
    for item in analyzer_multi.flagged_transactions[:10]: 
        tx = item['tx']
        an = item['analysis']
        print(f"TxID: {tx['transaction_id'][:8]}... | User: {tx['user_id']} | Amount: ${tx['amount']:.2f} | Reason: {an['reason']}")

    if args.export:
        export_to_csv(analyzer_multi, "flagged_transactions.csv")

if __name__ == "__main__":
    main()
