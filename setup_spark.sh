#!/bin/bash
# ============================================================
# Spark Standalone Cluster Setup v2 (more robust)
# ============================================================
# Uses unique SPARK_IDENT_STRING per worker so PIDs don't collide.
# Each worker gets its own log/pid directory.
#
# Use this version if v1 keeps losing worker 2.
# ============================================================

set -e

MASTER_HOST="localhost"
DEFAULT_MASTER_PORT=7077
DEFAULT_MASTER_WEBUI_PORT=8080
WORKER_CORES=4
WORKER_MEMORY="4G"
PORT_FILE=".spark_cluster_ports"

# Auto-detect SPARK_HOME
if [ -z "$SPARK_HOME" ]; then
    if command -v pyspark &> /dev/null; then
        SPARK_HOME=$(python -c "import pyspark; import os; print(os.path.dirname(pyspark.__file__))")
    else
        echo "ERROR: SPARK_HOME not set"
        exit 1
    fi
fi

# Use writable dirs for everything Spark needs to write
BASE_DIR="${SPARK_LOG_DIR:-/tmp/spark-${USER}}"
mkdir -p "$BASE_DIR"

find_free_port() {
    local start_port=$1
    local port=$start_port
    for ((i=0; i<200; i++)); do
        if python -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(('', $port))
    s.close()
    exit(0)
except OSError:
    exit(1)
" 2>/dev/null; then
            echo $port
            return 0
        fi
        port=$((port + 1))
    done
    return 1
}

start_one_worker() {
    local worker_num=$1
    local master_url=$2
    local webui_port=$3

    # Each worker gets its OWN ident string -> own PID file
    local ident="worker${worker_num}"
    local worker_dir="$BASE_DIR/${ident}"
    mkdir -p "$worker_dir/logs" "$worker_dir/pids" "$worker_dir/work"

    echo "  Starting worker $worker_num on Web UI port $webui_port..."

    # Run worker in background with isolated env
    SPARK_IDENT_STRING="$ident" \
    SPARK_LOG_DIR="$worker_dir/logs" \
    SPARK_PID_DIR="$worker_dir/pids" \
    SPARK_WORKER_DIR="$worker_dir/work" \
    "$SPARK_HOME/sbin/start-worker.sh" "$master_url" \
        --cores "$WORKER_CORES" \
        --memory "$WORKER_MEMORY" \
        --webui-port "$webui_port"

    sleep 2

    # Verify it actually started
    if [ ! -f "$worker_dir/pids/spark-${ident}-org.apache.spark.deploy.worker.Worker-1.pid" ]; then
        echo "  WARNING: worker $worker_num PID file not created. Check $worker_dir/logs/"
    fi
}

start_cluster() {
    echo "================================================"
    echo "Starting Spark cluster (v2)"
    echo "================================================"

    MASTER_PORT=$(find_free_port $DEFAULT_MASTER_PORT)
    MASTER_WEBUI_PORT=$(find_free_port $DEFAULT_MASTER_WEBUI_PORT)
    WORKER1_WEBUI_PORT=$(find_free_port $((MASTER_WEBUI_PORT + 1)))
    WORKER2_WEBUI_PORT=$(find_free_port $((WORKER1_WEBUI_PORT + 1)))

    MASTER_URL="spark://${MASTER_HOST}:${MASTER_PORT}"

    echo "  Master:    port $MASTER_PORT, UI $MASTER_WEBUI_PORT"
    echo "  Worker 1:  UI port $WORKER1_WEBUI_PORT"
    echo "  Worker 2:  UI port $WORKER2_WEBUI_PORT"
    echo "  Memory:    $WORKER_MEMORY each"
    echo "  Cores:     $WORKER_CORES each"
    echo

    # Persist ports
    cat > "$PORT_FILE" <<EOF
SPARK_MASTER_HOST=$MASTER_HOST
SPARK_MASTER_PORT=$MASTER_PORT
SPARK_MASTER_WEBUI_PORT=$MASTER_WEBUI_PORT
SPARK_WORKER1_WEBUI_PORT=$WORKER1_WEBUI_PORT
SPARK_WORKER2_WEBUI_PORT=$WORKER2_WEBUI_PORT
SPARK_MASTER_URL=$MASTER_URL
EOF

    # Start master with its own ident
    echo "[1/3] Starting master..."
    SPARK_IDENT_STRING="master" \
    SPARK_LOG_DIR="$BASE_DIR/master/logs" \
    SPARK_PID_DIR="$BASE_DIR/master/pids" \
    "$SPARK_HOME/sbin/start-master.sh" \
        --host "$MASTER_HOST" \
        --port "$MASTER_PORT" \
        --webui-port "$MASTER_WEBUI_PORT"
    mkdir -p "$BASE_DIR/master/logs" "$BASE_DIR/master/pids"
    sleep 3

    echo "[2/3] Starting worker 1..."
    start_one_worker 1 "$MASTER_URL" "$WORKER1_WEBUI_PORT"

    echo "[3/3] Starting worker 2..."
    start_one_worker 2 "$MASTER_URL" "$WORKER2_WEBUI_PORT"

    sleep 3
    echo
    echo "================================================"
    echo "Verification:"
    echo "================================================"

    N_WORKERS=$(pgrep -f "org.apache.spark.deploy.worker.Worker" | wc -l)
    echo "Worker processes running: $N_WORKERS"

    if [ "$N_WORKERS" -lt 2 ]; then
        echo "WARNING: fewer than 2 workers running."
        echo "Check logs in: $BASE_DIR/worker*/logs/"
        echo
        for d in "$BASE_DIR"/worker*/logs; do
            echo "--- $d ---"
            ls -lh "$d" 2>/dev/null
            echo
            for f in "$d"/*.out; do
                [ -f "$f" ] || continue
                echo "Last lines of $f:"
                tail -20 "$f"
                echo
            done
        done
    else
        echo "SUCCESS: $N_WORKERS workers running"
        echo
        echo "Master Web UI: http://localhost:$MASTER_WEBUI_PORT"
        echo "Worker 1 UI:   http://localhost:$WORKER1_WEBUI_PORT"
        echo "Worker 2 UI:   http://localhost:$WORKER2_WEBUI_PORT"
    fi
    echo "================================================"
}

stop_cluster() {
    echo "Stopping all Spark processes..."
    pkill -f "org.apache.spark.deploy.worker.Worker" 2>/dev/null || true
    pkill -f "org.apache.spark.deploy.master.Master" 2>/dev/null || true
    sleep 2
    rm -f "$PORT_FILE"
    rm -f "$BASE_DIR"//pids/.pid 2>/dev/null
    echo "Stopped."
}

status_cluster() {
    echo "================================================"
    echo "Cluster status"
    echo "================================================"
    if [ -f "$PORT_FILE" ]; then
        cat "$PORT_FILE" | sed 's/^/  /'
        echo
    fi
    MASTERS=$(pgrep -fa "org.apache.spark.deploy.master.Master" || echo "")
    WORKERS=$(pgrep -fa "org.apache.spark.deploy.worker.Worker" || echo "")
    [ -z "$MASTERS" ] && echo "Master:  NOT RUNNING" || echo "Master:  RUNNING ($MASTERS)"
    if [ -z "$WORKERS" ]; then
        echo "Workers: NOT RUNNING"
    else
        N=$(echo "$WORKERS" | wc -l)
        echo "Workers: $N running"
        echo "$WORKERS" | sed 's/^/         /'
    fi
}

case "${1:-}" in
    start)   start_cluster ;;
    stop)    stop_cluster ;;
    status)  status_cluster ;;
    restart) stop_cluster; sleep 2; start_cluster ;;
    *)
        echo "Usage: $0 {start|stop|status|restart}"
        exit 1
        ;;
esac