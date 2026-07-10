import os
import yaml
import requests
from vector_store import QdrantStore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_DIR = os.path.join(BASE_DIR, "OKF")   # 知識包路徑（Parent Document 的來源）
QDRANT_PATH = os.path.join(BASE_DIR, "db", "qdrant")  # Qdrant 持久化儲存路徑
COLLECTION_NAME = "it_support_dsi"  # Qdrant collection 名稱
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama 服務位址
EMBED_MODEL = "nomic-embed-text"  # Embedding 模型（需先在 Ollama pull）
EXCLUDE_FILES = {"index.md"}  # 跳過的檔案，index.md 不參與檢索


def parse_markdown(md_path):
    """
    解析單一 OKF 概念文件，拆出 YAML frontmatter 與 Markdown 正文。
    """
    with open(md_path, "r", encoding="utf-8") as f:
        raw = f.read()
    metadata = {}
    body = raw
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            try:
                metadata = yaml.safe_load(parts[1]) or {}
            except Exception as e:
                print(f"[錯誤] 解析 frontmatter 失敗 ({md_path}): {e}")
            body = parts[2]
    rel_path = os.path.relpath(md_path, BUNDLE_DIR).replace(os.sep, "/")
    concept_id = rel_path[:-3] if rel_path.endswith(".md") else rel_path
    return concept_id, metadata, body.strip()


def build_catalog_text(concept_id, metadata):
    """
    只用 frontmatter 的介紹性欄位（type、title、description、tags）組成一段文字，
    做為這份文件在 Stage 1 向量檢索中唯一的一筆 embedding 來源。
    不包含正文內容，也不切 chunk，一份檔案只對應一筆向量。
    """
    title = metadata.get("title", "")
    description = metadata.get("description", "")
    okf_type = metadata.get("type", "")
    tags = metadata.get("tags", []) or []
    tags_text = ", ".join(tags) if isinstance(tags, list) else str(tags)

    return f"{title}\n{description}\n類型: {okf_type}\n標籤: {tags_text}"


def embed_texts(texts):
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/embed", json={"model": EMBED_MODEL, "input": texts})
    resp.raise_for_status()
    return resp.json()["embeddings"]


def main():
    print("=== 開始建立 Document Summary Index 向量索引（目錄層級） ===\n")
    md_files = []
    for root, dirs, files in os.walk(BUNDLE_DIR):
        for f in files:
            if f.endswith(".md") and f not in EXCLUDE_FILES:
                md_files.append(os.path.join(root, f))

    if not md_files:
        print("找不到任何 .md 檔案 (排除 index.md)")
        return

    print(f"找到 {len(md_files)} 份概念檔案 (parent documents):")
    for f in md_files:
        print(f"  - {os.path.relpath(f, BUNDLE_DIR)}")
    print()

    documents = []
    metadatas = []
    ids = []

    for md_path in sorted(md_files):
        concept_id, metadata, body = parse_markdown(md_path)
        filename = os.path.basename(md_path)

        catalog_text = build_catalog_text(concept_id, metadata)

        documents.append(catalog_text)
        metadatas.append({
            "source": concept_id,     # 概念識別碼，相對於 BUNDLE_DIR 的路徑
            "filename": filename,     # 原始檔名，Stage 2 用來讀取完整 parent document
            "title": metadata.get("title", ""),
            "type": metadata.get("type", ""),
            "tags": ", ".join(metadata.get("tags", [])) if isinstance(metadata.get("tags"), list) else str(metadata.get("tags", "")),
            "description": metadata.get("description", ""),
        })
        ids.append(concept_id)

    print(f"共 {len(documents)} 份文件目錄（每份檔案只對應一筆向量，不切 chunk）\n")

    print("正在產生 Embedding (Ollama nomic-embed-text)...")
    embeddings = embed_texts(documents)
    print(f"Embedding 完成，維度: {len(embeddings[0]) if embeddings else 0}\n")

    print("正在寫入 Qdrant...")
    store = QdrantStore(COLLECTION_NAME, vector_size=len(embeddings[0]), path=QDRANT_PATH)
    store.create_collection(force=True)
    store.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)

    print(f"\n=== 完成 ===")
    print(f"文件數量: {store.count()}")
    print(f"儲存位置: {QDRANT_PATH}")
    print("\n--- 目錄摘要 ---")
    for doc, meta in zip(documents, metadatas):
        preview = doc.replace("\n", " ")[:80]
        print(f"  [{meta['filename']}]  →  {preview}...")


if __name__ == "__main__":
    main()
