import streamlit as st
import time
import pandas as pd
from generator import DataGenerator
from analyzer import FraudAnalyzer
from worker import WorkManager

st.set_page_config(page_title="Fraud Detection Simulation", layout="wide")
st.title("🛡️ Real-Time Multi-Threaded Fraud Detection")

# Initialize session state for persistence
if "analyzer" not in st.session_state:
    st.session_state.analyzer = FraudAnalyzer()
    st.session_state.generator = DataGenerator(num_users=200)
    st.session_state.total_tx_data = []
    st.session_state.performance_log = []

st.sidebar.header("⚙️ Simulation Controls")
batch_size = st.sidebar.slider("Batch Size", 50, 2000, 500)
threads = st.sidebar.slider("Worker Threads", 1, 16, 4)

if st.sidebar.button("🚀 Generate & Process Data"):
    with st.spinner(f"Generating and processing {batch_size} transactions..."):
        # Generate data
        tx_batch = st.session_state.generator.generate_batch(batch_size)
        st.session_state.total_tx_data.extend(tx_batch)
        
        # 1. Benchmark Single-Threaded (using a dummy analyzer to not double-count stats)
        dummy_analyzer = FraudAnalyzer()
        start = time.time()
        WorkManager.process_single_threaded(tx_batch, dummy_analyzer)
        single_time = time.time() - start
        
        # 2. Process Multi-Threaded (updating the actual dashboard state)
        start = time.time()
        WorkManager.process_multi_threaded(tx_batch, st.session_state.analyzer, thread_count=threads)
        multi_time = time.time() - start
        
        speedup = single_time / multi_time if multi_time > 0 else 0
        
        st.session_state.performance_log.append({
            "Batch": batch_size,
            "Threads": threads,
            "Single Thread (s)": single_time,
            "Multi Thread (s)": multi_time,
            "Speedup": speedup
        })
        st.sidebar.success(f"Multi-thread processed in {multi_time:.4f}s ({speedup:.2f}x speedup)")

# Dashboard UI
summary = st.session_state.analyzer.get_summary()

st.markdown("### 📊 Live Statistics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", summary["total_processed"])
col2.metric("Flagged as Fraud", summary["total_flagged"])
col3.metric("Fraud Rate", summary["fraud_rate"])
col4.metric("Avg Transaction", f"${summary['mean_amount']:.2f}")

st.markdown("---")

col_left, col_right = st.columns([2, 2])

with col_left:
    st.subheader("🚨 Recent Fraud Alerts")
    if st.session_state.analyzer.flagged_transactions:
        flagged_data = []
        # Show top 20 recent
        for item in reversed(st.session_state.analyzer.flagged_transactions[-20:]):
            tx = item['tx']
            an = item['analysis']
            flagged_data.append({
                "User": tx['user_id'],
                "Amount": tx['amount'],
                "Location": tx['location'],
                "Risk": an['risk_score'],
                "Reason": an['reason']
            })
        
        df = pd.DataFrame(flagged_data)
        st.dataframe(df.style.highlight_max(subset=['Risk'], color='red'), use_container_width=True)
    else:
        st.info("No fraudulent transactions detected yet.")

with col_right:
    st.subheader("📈 Performance Comparison")
    if st.session_state.performance_log:
        perf_df = pd.DataFrame(st.session_state.performance_log)
        
        # Show data table for details
        st.dataframe(
            perf_df[["Batch", "Threads", "Speedup"]].style.format({"Speedup": "{:.2f}x"}), 
            use_container_width=True
        )
        
        # Show bar chart comparing run times
        chart_data = perf_df[['Single Thread (s)', 'Multi Thread (s)']]
        st.bar_chart(chart_data)
    else:
        st.info("Run a simulation batch to see performance.")

if st.session_state.total_tx_data:
    st.markdown("### 💸 Transaction Amount Distribution (Last 200)")
    # Show history of amounts
    recent_txs = st.session_state.total_tx_data[-200:]
    amounts_df = pd.DataFrame({
        "Amount": [t["amount"] for t in recent_txs]
    })
    st.line_chart(amounts_df)
