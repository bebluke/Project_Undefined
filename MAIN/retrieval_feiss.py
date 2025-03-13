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
    針對 `locations` 清單，使用 `semantic_keywords` 進行 FAISS 檢索，確保 `gmap_location` 只出現一次，並返回最多 5 則評論。
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

    # **提高 `k` 值，讓 FAISS 可以返回更多評論**
    k = min(300, len(filtered_comments))
    distances, indices = index.search(query_vector, k)

    # **篩選有效的索引**
    valid_mask = indices[0] >= 0  # 過濾掉 -1
    valid_indices = indices[0][valid_mask]
    
    # **如果沒有有效索引，直接返回**
    if len(valid_indices) == 0:
        print("🚨 FAISS 沒有找到匹配的評論。")
        return []
    
    valid_distances = distances[0][valid_mask]

    # **使用 review_id 來確保索引對應**
    retrieved_review_ids = [review_ids[i] for i in valid_indices]

    # **篩選符合的評論**
    similar_reviews = filtered_comments[filtered_comments["review_id"].isin(retrieved_review_ids)].copy()

    # **合併地點名稱**
    df_merged = similar_reviews.merge(df_info[['location_id', 'gmap_location', 'address', 'summary_2']], on='location_id', how='left')

    # **確保 `distances` 和 `df_merged` 行數一致**
    valid_distances = valid_distances[: len(df_merged)]
    df_merged["similarity_score"] = 1 / (1 + valid_distances)

    # **應用 Softmax 加權**
    alpha = 5
    exp_scores = np.exp(df_merged["similarity_score"] * alpha)
    df_merged["weight"] = exp_scores / exp_scores.sum()

    # **統計地點的加權分數**
    location_scores = df_merged.groupby(["gmap_location", "location_id", "address", "summary_2"])["weight"].sum().reset_index()
    location_scores = location_scores.sort_values(by="weight", ascending=False)

    # **確保每個地點最多返回 5 則評論**
    grouped_reviews = df_merged.groupby("gmap_location").apply(lambda x: x.head(5)).reset_index(drop=True)

    # **回傳結果**
    results = []
    for idx, row in location_scores.head(10).iterrows():
        matched_reviews = grouped_reviews[grouped_reviews["gmap_location"] == row["gmap_location"]]

        results.append({
            "location_id": row["location_id"],
            "gmap_location": row["gmap_location"],
            "address": row["address"],
            "summary_2": row["summary_2"],
            "comments": matched_reviews["comments"].tolist(),  # 取最多 5 則評論
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
