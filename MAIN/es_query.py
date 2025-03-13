import json
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import BadRequestError
from query_understanding import parse_user_query
from retrieval_feiss import retrieve_similar_places  # 確保 FAISS 可用

# 連接 Elasticsearch
es = Elasticsearch("http://localhost:9200")

# 設定 Elasticsearch Index 名稱
INDEX_NAME = "poi_data"


def search_elasticsearch(parsed_query):
    """
    執行 Elasticsearch 搜尋，根據 `gmap_location`、`address`、`class` 等條件查找 POI。
    如果 `semantic_keywords` 存在，則將 `location_id` 傳給 FAISS 進行語意檢索。
    """
    es_query = {"query": {"bool": {"must": [], "should": [], "filter": []}}}

    # 地址或 Google Maps 地點名稱
    if "gmap_location" in parsed_query:
        es_query["query"]["bool"]["must"].append(
            {"match": {"gmap_location": {"query": parsed_query["gmap_location"], "fuzziness": "AUTO"}}}
        )
    if "address" in parsed_query:
        es_query["query"]["bool"]["must"].append(
            {"match_phrase": {"address": {"query": parsed_query["address"]}}}
        )

    # 類別
    if "class" in parsed_query:
        es_query["query"]["bool"]["must"].append(
            {"match": {"class": {"query": parsed_query["class"], "fuzziness": "AUTO", "boost": 3.0}}}
        )

    # 其他條件
    if "tags" in parsed_query:
        es_query["query"]["bool"]["must"].append(
            {"match": {"tags": {"query": parsed_query["tags"], "fuzziness": "AUTO"}}}
        )
    if "opening_hours" in parsed_query:
        es_query["query"]["bool"]["must"].append(
            {"match": {"opening_hours": {"query": parsed_query["opening_hours"]}}}
        )
    if "entrance_fee" in parsed_query:
        es_query["query"]["bool"]["must"].append(
            {"range": {"entrance_fee": {"lt": parsed_query["entrance_fee"]}}}
        )

    try:
        response = es.search(index=INDEX_NAME, body=es_query, size=20)
        hits = response["hits"]["hits"]

        if not hits:
            return []

        # **回傳 `location_id` 清單**
        results = [{"location_id": hit["_source"]["location_id"]} for hit in hits]
        return results

    except BadRequestError as e:
        print(f"❌ Elasticsearch BadRequestError: {e}")
        return []


if __name__ == "__main__":
    # 測試 `query_understanding.py` 解析的輸入
    user_input = "我想找台北市擁有豐富館藏的博物館"
    parsed_query = parse_user_query(user_input)

    print(f"\n🔍 測試 Elasticsearch 查詢: {parsed_query}")
    es_results = search_elasticsearch(parsed_query)

    if es_results:
        print("✅ Elasticsearch 結果:", json.dumps(es_results, ensure_ascii=False, indent=2))

        # **如果 `semantic_keywords` 存在，執行 FAISS 語意檢索**
        if "semantic_keywords" in parsed_query:
            faiss_results = retrieve_similar_places(es_results, parsed_query["semantic_keywords"])
            print("✅ FAISS 檢索結果:", json.dumps(faiss_results, ensure_ascii=False, indent=2))
    else:
        print("❌ 沒有找到符合條件的地點")
