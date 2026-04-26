#!/usr/bin/env python3
"""
Flask query interface with:
  - Actual question text in results (not just IDs)
  - Proper e5 query prefix for retrieval quality
  - Error handling
  - /stats endpoint showing index info
  - /health endpoint
  - Latency timing shown to user
"""
from flask import Flask, request, jsonify, render_template_string
import faiss
import numpy as np
import pandas as pd
import time
import logging
from sentence_transformers import SentenceTransformer
from config_gpu import *

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)

log.info("Loading sentence transformer model...")
model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, device="cuda")

log.info("Loading FAISS index...")
cpu_index = faiss.read_index(FAISS_INDEX_PATH)
gpu_index = cpu_index  # use CPU directly to avoid GPU FAISS issues
gpu_index.nprobe = 32

log.info("Loading metadata...")
metadata = pd.read_parquet(METADATA_PATH)
# Build id -> text lookup
id_to_text = dict(zip(metadata['id'].values, metadata['text'].values))
log.info(f"Ready: {gpu_index.ntotal} vectors indexed")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Semantic Search - Stack Exchange</title>
    <style>
        body { font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }
        h1 { color: #2c3e50; }
        .search-box { display: flex; gap: 10px; margin: 20px 0; }
        input[type=text] { flex: 1; padding: 12px; font-size: 16px; border: 2px solid #ddd; border-radius: 6px; }
        input[type=submit] { padding: 12px 24px; font-size: 16px; background: #3498db; color: white; border: none; border-radius: 6px; cursor: pointer; }
        input[type=submit]:hover { background: #2980b9; }
        .stats { color: #888; font-size: 13px; margin: 10px 0; }
        .result { padding: 15px; margin: 12px 0; border-left: 3px solid #3498db; background: #f9f9f9; border-radius: 4px; }
        .result-meta { font-size: 13px; color: #888; margin-bottom: 5px; }
        .result-text { color: #333; line-height: 1.5; }
        .score { background: #3498db; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; }
        .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; color: #888; font-size: 13px; }
    </style>
</head>
<body>
    <h1>Semantic Search</h1>
    <p style="color: #666;">Search {{ index_size }} Stack Exchange posts using dense embeddings (e5-large-v2 + FAISS IVF on GPU)</p>

    <form action="/search" method="get">
        <div class="search-box">
            <input type="text" name="q" value="{{ query|default('') }}" placeholder="Ask a technical question..." autofocus>
            <input type="submit" value="Search">
        </div>
    </form>

    {% if query %}
    <div class="stats">
        Found {{ results|length }} results in {{ "%.1f"|format(latency_ms) }} ms
    </div>
    {% endif %}

    {% if results %}
        {% for r in results %}
        <div class="result">
            <div class="result-meta">
                Post #{{ r.id }} <span class="score">score: {{ "%.3f"|format(r.score) }}</span>
            </div>
            <div class="result-text">{{ r.text[:400] }}{% if r.text|length > 400 %}...{% endif %}</div>
        </div>
        {% endfor %}
    {% endif %}

    <div class="footer">
        <a href="/stats">/stats</a> | <a href="/health">/health</a>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML, index_size=f"{gpu_index.ntotal:,}")

@app.route('/search')
def search():
    query = request.args.get('q', '').strip()
    if not query:
        return render_template_string(HTML, index_size=f"{gpu_index.ntotal:,}")

    try:
        t0 = time.perf_counter()

        # CRITICAL: e5 requires "query: " prefix on queries
        prefixed_query = f"query: {query}"
        vec = model.encode([prefixed_query], normalize_embeddings=True).astype(np.float32)

        distances, indices = gpu_index.search(vec, 10)
        latency_ms = (time.perf_counter() - t0) * 1000

        results = []
        for i, dist in zip(indices[0], distances[0]):
            if i == -1:
                continue
            row = metadata.iloc[i]
            results.append({
                "id": int(row['id']),
                "score": float(dist),
                "text": str(row['text'])
            })

        return render_template_string(HTML,
                                      query=query,
                                      results=results,
                                      latency_ms=latency_ms,
                                      index_size=f"{gpu_index.ntotal:,}")
    except Exception as e:
        log.exception("Search failed")
        return jsonify({"error": str(e)}), 500

@app.route('/api/search')
def api_search():
    """JSON API for programmatic access."""
    query = request.args.get('q', '').strip()
    k = int(request.args.get('k', 10))
    if not query:
        return jsonify({"error": "missing query parameter 'q'"}), 400

    t0 = time.perf_counter()
    prefixed_query = f"query: {query}"
    vec = model.encode([prefixed_query], normalize_embeddings=True).astype(np.float32)
    distances, indices = gpu_index.search(vec, k)
    latency_ms = (time.perf_counter() - t0) * 1000

    results = [
        {"id": int(metadata.iloc[i]['id']),
         "score": float(d),
         "text": str(metadata.iloc[i]['text'])}
        for i, d in zip(indices[0], distances[0]) if i != -1
    ]
    return jsonify({"query": query, "latency_ms": latency_ms, "results": results})

@app.route('/stats')
def stats():
    return jsonify({
        "total_vectors": int(gpu_index.ntotal),
        "embedding_dim": int(gpu_index.d),
        "nprobe": int(gpu_index.nprobe),
        "model": SENTENCE_TRANSFORMER_MODEL,
        "index_type": "IVF1024,Flat",
        "metric": "inner_product",
        "gpu_id": GPU_ID
    })

@app.route('/health')
def health():
    return jsonify({"status": "ok", "vectors": int(gpu_index.ntotal)})

def find_free_port(start_port=5000, max_attempts=100):
    """Find first free port >= start_port."""
    import socket
    for port in range(start_port, start_port + max_attempts):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(('', port))
            s.close()
            return port
        except OSError:
            continue
    raise RuntimeError(f"No free port found from {start_port}")

if __name__ == '__main__':
    port = find_free_port(start_port=5000)
    print(f"\n{'='*60}")
    print(f"Flask app starting on port {port}")
    print(f"  Local:    http://localhost:{port}")
    print(f"  Network:  http://0.0.0.0:{port}")
    print(f"\nFor SSH tunnel from your laptop:")
    print(f"  ssh -L {port}:localhost:{port} m25_jahanvi@iit-jodhpur")
    print(f"{'='*60}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
