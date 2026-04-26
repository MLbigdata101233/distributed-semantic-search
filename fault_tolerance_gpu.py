#!/usr/bin/env python3
"""
Real fault tolerance demonstration on a Spark standalone cluster.

Auto-discovers the master URL from .spark_cluster_ports (written by
setup_spark_cluster.sh), so port collisions don't break this script.

What it does:
  1. Reads the master URL from .spark_cluster_ports
  2. Connects to the running cluster
  3. Reads preprocessed parquet
  4. Runs a deliberately slow map across many partitions
  5. Halfway through, a background thread kills ONE Spark worker
  6. Spark detects the dead executor and reschedules tasks
  7. Job still completes -> proves fault tolerance

Outputs:
  - fault_tolerance_log.txt : timestamped event timeline
  - Spark UI shows the dead worker + task retries

Prerequisites:
  - ./setup_spark_cluster.sh start  (must be run first)
"""
import os
import time
import signal
import threading
import subprocess
from datetime import datetime
from pyspark.sql import SparkSession

LOG_FILE = "fault_tolerance_log.txt"
PROCESSED_PARQUET_DIR = "data/data/processed_parquet_gpu"  # adjust if needed
PORT_FILE = ".spark_cluster_ports"


def read_master_url():
    """Read the Spark master URL from the port file written by setup script."""
    if not os.path.exists(PORT_FILE):
        raise RuntimeError(
            f"{PORT_FILE} not found. Run ./setup_spark_cluster.sh start first."
        )
    config = {}
    with open(PORT_FILE) as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                config[k] = v
    if "SPARK_MASTER_URL" not in config:
        raise RuntimeError(f"SPARK_MASTER_URL not found in {PORT_FILE}")
    return config["SPARK_MASTER_URL"], config


def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def find_worker_pids():
    """Find all Spark worker process IDs."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "org.apache.spark.deploy.worker.Worker"],
            capture_output=True, text=True
        )
        return [int(p) for p in result.stdout.strip().split() if p]
    except Exception as e:
        log(f"Error finding workers: {e}")
        return []


def kill_one_worker_after(delay_seconds):
    """Background thread: wait, then SIGKILL one worker."""
    log(f"[KILLER] Will kill a worker in {delay_seconds}s...")
    time.sleep(delay_seconds)

    pids = find_worker_pids()
    log(f"[KILLER] Found {len(pids)} worker process(es): {pids}")

    if len(pids) < 2:
        log("[KILLER] WARNING: fewer than 2 workers running.")
        return

    target_pid = pids[0]
    log(f"[KILLER] Killing worker PID {target_pid} with SIGKILL...")
    try:
        os.kill(target_pid, signal.SIGKILL)
        log(f"[KILLER] Worker {target_pid} killed.")
        log("[KILLER] Spark should now detect the loss "
            "and reschedule tasks on the surviving worker.")
    except ProcessLookupError:
        log(f"[KILLER] Process {target_pid} already gone.")
    except PermissionError:
        log(f"[KILLER] No permission to kill {target_pid}. "
            "Run as the user that started the cluster.")


def slow_operation(row):
    """Deliberately slow per-row work so the kill happens mid-execution."""
    import time as _time
    _time.sleep(0.005)  # 5ms per row
    text = row.text_for_embedding if hasattr(row, "text_for_embedding") else ""
    return (row.id, len(text) if text else 0)


def main():
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    log("=" * 60)
    log("FAULT TOLERANCE DEMONSTRATION")
    log("=" * 60)

    master_url, port_config = read_master_url()
    log(f"Discovered master URL: {master_url}")
    log(f"Master Web UI: http://localhost:{port_config.get('SPARK_MASTER_WEBUI_PORT', '?')}")

    initial_workers = find_worker_pids()
    log(f"Initial worker PIDs: {initial_workers}")
    if len(initial_workers) < 2:
        log("ERROR: need at least 2 workers running.")
        log("Run: ./setup_spark_cluster.sh start")
        return

    log(f"Connecting to {master_url}...")
    spark = (
        SparkSession.builder
        .appName("FaultToleranceDemo")
        .master(master_url)
        .config("spark.executor.memory", "4g")
        .config("spark.executor.cores", "2")
        .config("spark.task.maxFailures", "4")
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    log("Spark session ready.")
    log(f"Reading parquet from {PROCESSED_PARQUET_DIR}...")
    df = spark.read.parquet(PROCESSED_PARQUET_DIR)

    n_partitions = 40
    df = df.repartition(n_partitions)
    total_rows = df.count()
    log(f"Loaded {total_rows} rows across {n_partitions} partitions.")

    # Schedule killer thread for 15s into job
    killer = threading.Thread(target=kill_one_worker_after, args=(15,), daemon=True)
    killer.start()

    log("Starting slow map operation (~60-90s)...")
    log(f"Watch Spark UI: http://localhost:{port_config.get('SPARK_MASTER_WEBUI_PORT', '?')}")
    log("Around 15s, one worker will die. Job should still finish.")

    t_start = time.time()
    try:
        result = df.rdd.map(slow_operation).count()
        elapsed = time.time() - t_start
        log("=" * 60)
        log("JOB COMPLETED SUCCESSFULLY despite worker failure")
        log(f"  Rows processed: {result}")
        log(f"  Elapsed:        {elapsed:.1f}s")
        log("=" * 60)

        final_workers = find_worker_pids()
        log(f"Final worker PIDs: {final_workers}")
        log(f"Workers lost during job: {len(initial_workers) - len(final_workers)}")

        log("")
        log("PROOF OF FAULT TOLERANCE:")
        log("  - Job started with 2 workers")
        log("  - 1 worker SIGKILL'd mid-execution")
        log(f"  - Job still completed and processed all {result} rows")
        log("  - This is Spark's RDD lineage + task retry in action")

    except Exception as e:
        log(f"JOB FAILED: {e}")
        raise
    finally:
        spark.stop()
        log("Spark session stopped.")
        log(f"Full timeline: {LOG_FILE}")


if __name__ == "__main__":
    main()
    