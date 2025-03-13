import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# **載入語意模型**
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', device='cuda')

# **讀取 FAISS 索引**
index = faiss.read_index("faiss_index_ivf_new.bin")

# **讀取評論數據**
df_comments = pd.read_csv("cleaned_comments_new.csv", usecols=["review_id", "location_id", "comments", "language"])

# **讀取地點資訊**
df_info = pd.read_json("E_838_location_info.json")
df_info = df_info.rename(columns={"location_id": "location_id", "gmap_location": "gmap_location"})

# **讀取 FAISS review_id 對應**
review_ids = np.load("review_ids_new.npy")  # FAISS 對應的 review_id


def retrieve_similar_places(locations, semantic_keywords):
    """
    針對 `locations` 清單，使用 `semantic_keywords` 進行 FAISS 檢索，確保 `gmap_location` 只出現一次
    """
    print("📝 FAISS 檢索開始...")
    print(f"🗺 限制檢索的地點數量: {len(locations)}")
    print(f"🔑 使用語意關鍵詞: {semantic_keywords}")

    if not locations or not semantic_keywords:
        print("🚨 沒有地點或語意關鍵字，跳過 FAISS 檢索。")
        return []

    selected_locations = [loc["location_id"] for loc in locations]

    # **篩選符合的評論**
    filtered_comments = df_comments[df_comments["location_id"].isin(selected_locations)]
    print(f"🔍 符合條件的評論數量: {len(filtered_comments)}")

    if filtered_comments.empty:
        print("🚨 沒有符合條件的評論，FAISS 跳過！")
        return []

    # **向量化 `semantic_keywords`**
    query_vector = model.encode([" ".join(semantic_keywords)]).astype('float32')

    # **設定最大檢索評論數量**
    k = min(100, len(filtered_comments))

    # **執行 FAISS 語意檢索**
    distances, indices = index.search(query_vector, k)

    # **確保 FAISS 檢索結果數量正確**
    valid_indices = [i for i in indices[0] if i >= 0]
    if not valid_indices:
        print("🚨 FAISS 沒有找到匹配的評論。")
        return []

    # **使用 review_id 來確保索引對應**
    retrieved_review_ids = [review_ids[i] for i in valid_indices]

    # **篩選符合的評論**
    similar_reviews = filtered_comments[filtered_comments["review_id"].isin(retrieved_review_ids)].copy()

    # **合併地點名稱**
    df_merged = similar_reviews.merge(df_info[['location_id', 'gmap_location']], on='location_id', how='left')

    # **計算相似度分數**
    df_merged["similarity_score"] = 1 / (1 + distances[0][:len(df_merged)])

    # **應用 Softmax 加權**
    alpha = 5
    exp_scores = np.exp(df_merged["similarity_score"] * alpha)
    df_merged["weight"] = exp_scores / exp_scores.sum()

    # **合併相同 `gmap_location`，確保地點只出現一次**
    location_scores = df_merged.groupby(["gmap_location", "location_id"])["weight"].sum().reset_index()
    location_scores = location_scores.sort_values(by="weight", ascending=False)

    # **保留最相似的評論（但不重複地點）**
    best_reviews = df_merged.sort_values(by="weight", ascending=False).drop_duplicates(subset=["gmap_location"])

    # **回傳結果（最多 10 筆）**
    results = []
    for idx, row in location_scores.head(10).iterrows():
        matched_review = best_reviews[best_reviews["gmap_location"] == row["gmap_location"]].iloc[0]
        results.append({
            "location_id": row["location_id"],
            "gmap_location": row["gmap_location"],
            "comments": matched_review["comments"],  # 只保留最高匹配的評論
            "weight": row["weight"]
        })

    return results


if __name__ == "__main__":
    # **測試 FAISS 檢索**
    test_locations = []
    test_keywords = []

    print("\n🔍 測試 FAISS 語意檢索...")
    faiss_results = retrieve_similar_places(test_locations, test_keywords)

    if faiss_results:
        print("\n✅ FAISS 檢索結果（最多 10 條）：")
        for idx, row in enumerate(faiss_results):
            print(f"{idx+1}. 📍 {row['gmap_location']}")
            print(f"   💬 {row['comments']}")
            print(f"   ⚖️  相似度權重: {row['weight']:.4f}")
            print("-" * 60)
    else:
        print("❌ 沒有找到匹配的評論。")
