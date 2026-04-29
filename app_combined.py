#!/usr/bin/env python3
"""
Flask query interface for multi-source semantic search.

Each result's URL is constructed from its row-level source_site field,
so AskUbuntu posts link to askubuntu.com and SuperUser posts link to
superuser.com automatically.
"""
from flask import Flask, request, jsonify, render_template_string
import faiss
import numpy as np
import pandas as pd
import time
import re
import html as html_module
import socket
import logging
from sentence_transformers import SentenceTransformer
from config_gpu import *

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)


def post_url(post_id, source_site):
    """Build URL for a post given its source site."""
    return f"https://{source_site}/q/{post_id}"


def clean_text(text):
    """Strip HTML tags and decode entities."""
    if not text:
        return ""
    text = html_module.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = html_module.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def find_free_port(start_port=5000, max_attempts=100):
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


# ---------------- Load model and index ----------------
log.info("Loading sentence transformer model...")
model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, device="cuda")

log.info("Loading FAISS index...")
cpu_index = faiss.read_index("/data/m25_jahanvi/ML_big/combined_data/faiss_index_gpu.index")
search_index = cpu_index
search_index.nprobe = 32

log.info("Loading metadata...")
metadata = pd.read_parquet("/data/m25_jahanvi/ML_big/combined_data/metadata_gpu.parquet")
log.info(f"Ready: {search_index.ntotal} vectors indexed")

# Show source distribution at startup
if 'source_site' in metadata.columns:
    dist = metadata['source_site'].value_counts().to_dict()
    log.info(f"Source distribution: {dist}")
    SOURCE_DIST = dist
else:
    log.warning("No source_site column in metadata! Falling back to single-source.")
    SOURCE_DIST = {"unknown": len(metadata)}


# ---------------- HTML template ----------------
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Semantic Search - Multi-Site</title>
    <style>
        body { font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }
        h1 { color: #2c3e50; }
        .source-info { color: #888; font-size: 13px; margin: 10px 0; }
        .search-box { display: flex; gap: 10px; margin: 20px 0; }
        input[type=text] { flex: 1; padding: 12px; font-size: 16px; border: 2px solid #ddd; border-radius: 6px; }
        input[type=submit] { padding: 12px 24px; font-size: 16px; background: #3498db; color: white; border: none; border-radius: 6px; cursor: pointer; }
        input[type=submit]:hover { background: #2980b9; }
        .stats { color: #888; font-size: 13px; margin: 10px 0; }
        .result { padding: 15px; margin: 12px 0; border-left: 3px solid #3498db; background: #f9f9f9; border-radius: 4px; }
        .result-meta { font-size: 13px; color: #888; margin-bottom: 8px; }
        .result-text { color: #333; line-height: 1.5; }
        .score { background: #3498db; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; }
        .source-tag { background: #2ecc71; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px; margin-left: 6px; }
        .post-link { margin-left: 10px; color: #2980b9; text-decoration: none; font-weight: 500; }
        .post-link:hover { text-decoration: underline; }
        .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; color: #888; font-size: 13px; }
    </style>
</head>
<body>
    <h1>Multi-Site Semantic Search</h1>
    <div class="source-info">
        Searching {{ index_size }} posts across multiple Stack Exchange sites:
        {% for site, count in source_dist.items() %}
            <span class="source-tag">{{ site }} ({{ count }})</span>
        {% endfor %}
    </div>

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
                Post #{{ r.id }}
                <span class="score">score: {{ "%.3f"|format(r.score) }}</span>
                <span class="source-tag">{{ r.source_site }}</span>
                <a class="post-link" href="{{ r.url }}" target="_blank" rel="noopener">View original post &rarr;</a>
            </div>
            <div class="result-text">{{ r.text[:400] }}{% if r.text|length > 400 %}...{% endif %}</div>
        </div>
        {% endfor %}
    {% endif %}

    <div class="footer">
        <a href="/stats">/stats</a> | <a href="/health">/health</a> | <a href="/api/search?q=test">/api/search</a>
    </div>
</body>
</html>
"""


# ---------------- Routes ----------------
@app.route('/')
def home():
    return render_template_string(HTML,
                                  index_size=f"{search_index.ntotal:,}",
                                  source_dist=SOURCE_DIST)


@app.route('/search')
def search():
    query = request.args.get('q', '').strip()
    if not query:
        return render_template_string(HTML,
                                      index_size=f"{search_index.ntotal:,}",
                                      source_dist=SOURCE_DIST)

    try:
        t0 = time.perf_counter()
        prefixed_query = f"query: {query}"
        vec = model.encode([prefixed_query], normalize_embeddings=True).astype(np.float32)
        distances, indices = search_index.search(vec, 10)
        latency_ms = (time.perf_counter() - t0) * 1000

        results = []
        for i, dist in zip(indices[0], distances[0]):
            if i == -1:
                continue
            row = metadata.iloc[i]
            post_id = int(row['id'])
            source_site = str(row.get('source_site', 'unknown'))
            results.append({
                "id": post_id,
                "score": float(dist),
                "text": clean_text(str(row['text'])),
                "source_site": source_site,
                "url": post_url(post_id, source_site)
            })

        return render_template_string(HTML,
                                      query=query,
                                      results=results,
                                      latency_ms=latency_ms,
                                      index_size=f"{search_index.ntotal:,}",
                                      source_dist=SOURCE_DIST)
    except Exception as e:
        log.exception("Search failed")
        return jsonify({"error": str(e)}), 500


@app.route('/api/search')
def api_search():
    query = request.args.get('q', '').strip()
    k = int(request.args.get('k', 10))
    if not query:
        return jsonify({"error": "missing query parameter 'q'"}), 400

    t0 = time.perf_counter()
    prefixed_query = f"query: {query}"
    vec = model.encode([prefixed_query], normalize_embeddings=True).astype(np.float32)
    distances, indices = search_index.search(vec, k)
    latency_ms = (time.perf_counter() - t0) * 1000

    results = []
    for i, d in zip(indices[0], distances[0]):
        if i == -1:
            continue
        row = metadata.iloc[i]
        post_id = int(row['id'])
        source_site = str(row.get('source_site', 'unknown'))
        results.append({
            "id": post_id,
            "score": float(d),
            "text": clean_text(str(row['text'])),
            "source_site": source_site,
            "url": post_url(post_id, source_site)
        })
    return jsonify({"query": query, "latency_ms": latency_ms, "results": results})


@app.route('/stats')
def stats():
    return jsonify({
        "total_vectors": int(search_index.ntotal),
        "embedding_dim": int(search_index.d),
        "nprobe": int(search_index.nprobe),
        "model": SENTENCE_TRANSFORMER_MODEL,
        "index_type": "IVF1024,Flat",
        "metric": "inner_product",
        "compute": "CPU FAISS",
        "source_distribution": SOURCE_DIST
    })


@app.route('/health')
def health():
    return jsonify({"status": "ok", "vectors": int(search_index.ntotal)})


if __name__ == '__main__':
    port = find_free_port(start_port=5000)
    print(f"\n{'='*60}")
    print(f"Multi-site Flask app starting on port {port}")
    print(f"  Local on server: http://localhost:{port}")
    print(f"  Sources: {SOURCE_DIST}")
    print(f"\nFor SSH tunnel from your laptop:")
    print(f"  ssh -L {port}:localhost:{port} m25_jahanvi@iit-jodhpur")
    print(f"\nThen open: http://localhost:{port}")
    print(f"{'='*60}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
