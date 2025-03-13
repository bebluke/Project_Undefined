import json
import time
import google.generativeai as genai

# 設定 Gemini API Key
genai.configure(api_key="AIzaSyC1xJFic-KVm3bfhd6f53YFv5BBYjc5oXg")

# 初始化 Gemini 模型
GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-pro")

def clean_gemini_output(response_text):
    """
    清理 Gemini API 回傳的 JSON 格式，去除 Markdown 代碼塊標記。
    """
    response_text = response_text.strip()

    # **移除開頭和結尾的 Markdown 標籤**
    if response_text.startswith("```json"):
        response_text = response_text[7:]  # 移除 ```json
    elif response_text.startswith("```"):
        response_text = response_text[3:]  # 移除 ```
    
    if response_text.endswith("```"):
        response_text = response_text[:-3]  # 移除結尾的 ```
    
    return response_text.strip()  # 確保不含多餘換行

def parse_user_query(query):
    """
    解析使用者查詢，確保 `gmap_location` 和 `class` 使用中文名稱，`geo_distance` 使用標準距離格式。
    """
    chat = GEMINI_MODEL.start_chat()

    prompt = f"""
    解析使用者查詢，並確保輸出符合以下要求：
    
    - **地點名稱 (`gmap_location`)** 必須是**中文名稱**（例如："士林夜市"）。
    - **地理位置 (`address`)** 台灣所有的縣市都要解析為address，如台中=台中市、新竹="新竹縣"、"新竹市。
    - **地點類別 (`class`)** 必須對應 Google Maps **中文類別**（例如："博物館"、"餐廳"）。
    - **地理距離 (`geo_distance`)** 需轉換為標準格式（例如："5km"、"10km"）。
    - **地點標籤 (`tags`)** 如"適合兒童"、"無障礙停車場"、"洗手間"、Wi-Fi"。
    - **營業時間 (`opening_hours`)** 需轉換為標準格式（例如："週二","下午"）。
    - **語意關鍵詞 (`semantic_keywords`)** 若存在，請保留原文。
    
    **"gmap_location"和"address"不同時出現在同一輸出，沒有明確地點才選擇"address"

    **範例輸出 (請確保 JSON 格式正確，不要包裹 Markdown 標記)**：
    {{
      "gmap_location": "士林夜市",
      "class": "博物館",
      "geo_distance": "5km"
    }}

    請解析以下查詢：
    「{query}」
    """

    response = chat.send_message(prompt)

    if response.parts:
        try:
            cleaned_text = clean_gemini_output(response.text)
            parsed_query = json.loads(cleaned_text)

            # **修正 geo_distance**
            if "geo_distance" in parsed_query and isinstance(parsed_query["geo_distance"], str):
                if parsed_query["geo_distance"].lower() in ["附近", "nearest"]:
                    parsed_query["geo_distance"] = "5km"

            return parsed_query
        except json.JSONDecodeError:
            raise ValueError(f"❌ JSON 解析失敗，原始輸出:\n{cleaned_text}")
    else:
        raise ValueError("❌ Gemini 無法解析查詢")


if __name__ == "__main__":
    test_queries = [
        "I would like find the nearest museum of Shilin Night Market",
        "有哪些免費的藝術博物館",
        "大安區適合親子的景點",
    ]
    
    for query in test_queries:
        print(f"\n🔍 測試 Query: {query}")
        try:
            result = parse_user_query(query)
            print("✅ 解析結果:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except ValueError as e:
            print(e)
        time.sleep(1.5)
