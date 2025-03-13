from flask import Flask, request, render_template, jsonify
from query_understanding import parse_user_query
from es_query import search_elasticsearch
from retrieval_feiss import retrieve_similar_places
from werkzeug.serving import WSGIRequestHandler
import threading
import time



app = Flask(__name__)

# 設定 WSGI 請求的 `timeout`
WSGIRequestHandler.timeout = 300  # 設為 5 分鐘，防止 Flask 請求超時

# 存放 FAISS 結果的字典（非同步查詢用）
faiss_results_cache = {}

def async_faiss_search(query_id, es_results, semantic_keywords):
    """FAISS 以非同步方式執行，並將結果存入快取"""
    print(f"🚀 [FAISS] 開始處理 Query ID: {query_id}")
    
    # 先預設 FAISS 結果為 `processing`
    faiss_results_cache[query_id] = "processing"
    
    faiss_results = retrieve_similar_places(es_results, semantic_keywords)
    
    if faiss_results is None:
        print(f"❌ [FAISS] Query ID: {query_id} 執行失敗，返回 None")
        faiss_results_cache[query_id] = []
    else:
        print(f"✅ [FAISS] 完成處理 Query ID: {query_id}, 返回 {len(faiss_results)} 條結果")
        faiss_results_cache[query_id] = faiss_results

    # **檢查快取內容**
    print(f"📌 [FAISS] 查詢結果已存入快取: {query_id} -> {faiss_results_cache.get(query_id, '不存在')}")


@app.route("/", methods=["GET", "POST"])
def index():
    """渲染 HTML 首頁"""
    if request.method == "POST":
        user_query = request.form["query"]
        return search(user_query)
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    """處理 API 查詢"""
    user_query = request.json.get("query", "")
    if not user_query:
        return jsonify({"error": "請輸入查詢內容"}), 400

    query_id = str(int(time.time()))  # 產生唯一 Query ID
    parsed_query = parse_user_query(user_query)
    print(f"🔍 解析後的查詢: {parsed_query}")

    # **執行 Elasticsearch 查詢**
    es_results = search_elasticsearch(parsed_query)
    if not es_results:
        return jsonify({"message": "沒有找到符合條件的地點"}), 404

    response_data = {"query_id": query_id, "elasticsearch_results": es_results}

    # **如果 `semantic_keywords` 存在，則 FAISS 以非同步方式執行**
    if "semantic_keywords" in parsed_query:
        faiss_thread = threading.Thread(target=async_faiss_search, args=(query_id, es_results, parsed_query["semantic_keywords"]))
        faiss_thread.start()

    return jsonify(response_data)

import time

@app.route("/faiss_result/<query_id>", methods=["GET"])
def get_faiss_result(query_id):
    """查詢 FAISS 結果，等待結果完成"""
    for _ in range(10):  # 最多重試 10 次，每次等 1 秒
        if query_id in faiss_results_cache:
            result = faiss_results_cache[query_id]
            if result != "processing":
                return jsonify({"query_id": query_id, "faiss_results": result})
        time.sleep(1)  # 等待 FAISS 結果完成
    
    return jsonify({"query_id": query_id, "status": "not_found"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6262, debug=True)

