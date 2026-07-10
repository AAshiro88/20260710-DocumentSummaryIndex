import os
import re
import base64
import mimetypes
import yaml
import requests
import gradio as gr
from dotenv import load_dotenv
from vector_store import QdrantStore

OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama 服務位址
EMBED_MODEL = "nomic-embed-text"  # Stage 1 向量檢索用的 Embedding 模型
OLLAMA_STAGE2_MODEL = "DedeProGames/open-reasoner:1.7b"  # Stage 2 回答模型（STAGE2_MODEL="ollama" 時使用）
# OLLAMA_STAGE2_MODEL = "qwen3.6"  # Ollama 回答模型（STAGE2_MODEL="ollama" 時使用）

OLLAMA_TIMEOUT = 600  # Ollama 請求逾時秒數

STAGE2_MODEL = "ollama"  # Stage 2 回答模型："ollama" 或 "gemini"

# ============================================================
# Document Summary Index 相關設定（目錄層級版本）
# ============================================================
# Stage 1 向量庫裡「一份檔案只對應一筆向量」（用 frontmatter 的 title/description/tags/type 組成），
# 所以這裡取回的每一筆結果本身就對應一份完整的 parent document，不需要再做 chunk 去重複。
TOP_K_DOCS = 3  # 最多取回幾份相關的 parent document

# Rerank 相關設定：Stage 1 先召回較多候選文件，再用 CrossEncoder 依語意相關性重新排序，
# 篩到 TOP_K_DOCS 份最相關的文件才進入 Stage 2，藉此補強純向量比對摘要文字時精準度不足的問題。
RERANK_ENABLE = True  # 是否啟用 Rerank 重新排序
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"  # Rerank 模型（第一次啟動會自動下載）
RERANK_RETRIEVE_K = 15  # 第一階段召回的候選文件數量（Rerank 前，需大於 TOP_K_DOCS）

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_DIR = os.path.join(BASE_DIR, "OKF", "it-support")
QDRANT_PATH = os.path.join(BASE_DIR, "db", "qdrant")
COLLECTION_NAME = "it_support_dsi"
VECTOR_SIZE = 768  # nomic-embed-text 向量維度

store = QdrantStore(COLLECTION_NAME, VECTOR_SIZE, path=QDRANT_PATH)

env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=env_path)
api_key = os.getenv("GOOGLE_API_KEY")

gemini_client = None
gemini_types = None
if STAGE2_MODEL == "gemini" and api_key:
    from google import genai
    from google.genai import types as _types
    gemini_client = genai.Client(api_key=api_key)
    gemini_types = _types

reranker = None
if RERANK_ENABLE:
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(RERANK_MODEL)
        print(f"[Rerank] 模型載入完成: {RERANK_MODEL}")
    except Exception as e:
        print(f"[警告] Rerank 模型載入失敗，已自動停用: {e}")


# ============================================================
# State Machine - 對話狀態常數
# ============================================================
S_IDLE = "idle"
S_WAIT_CONFIRM = "wait_confirm"
S_WAIT_REPHRASE = "wait_rephrase"
S_SELECTING = "selecting"
S_WAIT_QUESTION = "wait_question"

CONFIRM_WORDS = {"是", "對", "yes", "對的", "是的", "正確", "沒錯", "要", "好"}
REJECT_WORDS = {"不對", "不是", "否", "no", "錯誤", "不正確", "不是這個", "重新", "換"}

def _is_confirm(text):
    t = text.strip().lower()
    return t in {w.lower() for w in CONFIRM_WORDS}

def _is_reject(text):
    t = text.strip().lower()
    return any(t == w.lower() or t.startswith(w.lower()) for w in REJECT_WORDS)


# ============================================================
# OKF 概念文件解析（Stage 2 用，讀取完整 parent document）
# ============================================================
def parse_okf_markdown(md_path, bundle_dir):
    """
    解析單一 OKF 概念文件（.md），拆出 YAML frontmatter 與 Markdown 正文。
    回傳 (概念識別碼, metadata, 正文內容)。
    """
    with open(md_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    metadata = {}
    body = raw_text

    if raw_text.startswith("---"):
        parts = raw_text.split("---", 2)
        if len(parts) >= 3:
            try:
                metadata = yaml.safe_load(parts[1]) or {}
            except Exception as e:
                print(f"[錯誤] 解析 frontmatter 失敗 ({md_path}): {e}")
            body = parts[2]

    rel_path = os.path.relpath(md_path, bundle_dir).replace(os.sep, "/")
    concept_id = rel_path[:-3] if rel_path.endswith(".md") else rel_path

    return concept_id, metadata, body.strip()


def format_concept_segment(concept_id, metadata, body, filename):
    """
    將單一概念文件的解析結果，組裝成給回答模型閱讀的結構化文字區塊。
    """
    okf_type = metadata.get("type", "Index" if filename == "index.md" else "Unknown")
    title = metadata.get("title", concept_id)
    yaml_str = yaml.dump(metadata, allow_unicode=True).strip() if metadata else "(無 frontmatter)"

    segment = f"=== OKF 概念檔案開始 ===\n"
    segment += f"概念識別碼 (Concept ID): {concept_id}\n"
    segment += f"知識類型 (Type): {okf_type}\n"
    segment += f"標題 (Title): {title}\n"
    segment += f"元數據 (Metadata):\n{yaml_str}\n"
    segment += f"詳細內容 (Content):\n{body}\n"
    segment += f"=== OKF 概念檔案結束 ===\n"
    return segment


def build_markdown_file_index(bundle_dir):
    """
    走訪 bundle_dir 底下所有 .md 檔案，建立「檔名 -> 完整路徑」的對照表，
    供 Stage 2 依 Stage 1 定位出的檔名讀取完整原文。
    """
    md_index = {}
    for root, dirs, files in os.walk(bundle_dir):
        for filename in files:
            if filename.endswith(".md"):
                md_index[filename] = os.path.join(root, filename)
    return md_index


def build_image_index(bundle_dir):
    idx = {}
    for root, dirs, files in os.walk(bundle_dir):
        for f in files:
            if f.lower().endswith(IMAGE_EXTENSIONS):
                idx[f] = os.path.join(root, f)
    return idx


md_index = build_markdown_file_index(BUNDLE_DIR)
image_index = build_image_index(BUNDLE_DIR)


def images_to_markdown(text, allowed_images=None):
    """
    將回答文字中 [圖片:檔名.png] 標記，轉換為內嵌 base64 圖片，
    讓圖片直接顯示在 Gradio 對話框內，而不是另外跳出檢視器視窗。
    傳入 allowed_images 時只允許被召回文件實際參照的圖片。
    """
    img_source = allowed_images if allowed_images is not None else image_index

    def replacer(m):
        fp = img_source.get(m.group(1))
        if fp:
            mime_type = mimetypes.guess_type(fp)[0] or "image/png"
            with open(fp, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode("ascii")
            return f"![](data:{mime_type};base64,{encoded})"
        return f"*(圖片 {m.group(1)} 未在知識包中找到)*"
    return re.sub(r"\[圖片:([^\]]+)\]", replacer, text)


# ============================================================
# Stage 1：向量搜尋「目錄」（title/description/tags/type），
# 每一筆結果本身就對應一份完整檔案，不需要 chunk 去重複
# ============================================================
def embed_text(text):
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/embed", json={"model": EMBED_MODEL, "input": [text]})
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def search_catalog(query, top_k=TOP_K_DOCS):
    embedding = embed_text(query)
    try:
        return store.search(query_embedding=embedding, top_k=top_k)
    except Exception:
        return None


def rerank_catalog(query, hits, top_k):
    """
    用 CrossEncoder 針對 Stage 1 召回的文件摘要（title/description/tags 組成的 document 欄位）
    重新計算與問題的相關性分數，取分數最高的 top_k 份文件，藉此補足純向量比對在候選文件
    變多時精準度下降的問題。
    """
    pairs = [(query, hit.payload.get("document", "")) for hit in hits]
    scores = reranker.predict(pairs)
    scored = list(zip(scores, hits))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in scored[:top_k]]


def select_parent_documents(hits):
    """
    從向量搜尋回傳的結果中取出 filename，依分數高低排序，
    保留原始順序去重複（理論上這裡本來就不會重複，因為一份檔案只有一筆向量，
    保留這道手續只是為了保險）。
    """
    seen = []
    for hit in hits:
        filename = hit.payload.get("filename")
        if filename and filename not in seen:
            seen.append(filename)
    return seen


# ============================================================
# Stage 2：讀取完整原始 OKF 檔案（不是目錄摘要），
# 把全部內容交給 AI 自行分析回答
# ============================================================
def load_parent_documents(bundle_dir, filenames):
    """
    只讀取 Stage 1 定位出的 parent document 完整內容（frontmatter + 正文），
    確保送給回答模型的是完整、未被切斷的原文。
    回傳 (context_text, allowed_images)，
    allowed_images 是 {basename: abs_path} 的 dict，只有被召回文件中實際參照的圖片。
    """
    context_segments = []
    allowed_images = {}
    IMG_RE = re.compile(r"!\[.*?\]\(img/([^)]+)\)")
    for filename in filenames:
        md_path = md_index.get(filename)
        if not md_path:
            print(f"[警告] Stage 1 定位出「{filename}」，但在知識包內找不到對應檔案。")
            continue
        concept_id, metadata, body = parse_okf_markdown(md_path, bundle_dir)
        context_segments.append(format_concept_segment(concept_id, metadata, body, filename))
        for m in IMG_RE.finditer(body):
            img_name = m.group(1)
            img_dir = os.path.dirname(md_path)
            img_full = os.path.normpath(os.path.join(img_dir, "img", img_name))
            if os.path.isfile(img_full):
                allowed_images[img_name] = img_full
    return "\n".join(context_segments), allowed_images


SYSTEM_INSTRUCTION = (
    "你是一個企業內部的 IT 支援 AI 助手。請依據下方提供的 OKF (Open Knowledge Format) 知識包內容來回答用戶問題。\n"
    "回答時請依知識包中的步驟順序說明，不可省略或自行調整步驟。\n"
    "如果用戶的問題超出知識包內容範圍，必須明確告知知識包中沒有相關資訊，不可自行編造答案。\n"
    "知識包內容中，某些步驟附有對應的示意圖，圖片檔名會以 Markdown 圖片語法出現，"
    "例如 ![authenticator-02](img/authenticator-02.png)。\n"
    "當你的回答提到某個步驟且該步驟在知識包中附有示意圖時，"
    "請在該步驟說明文字之後，加上格式為 [圖片:檔名.png] 的標記，例如 [圖片:authenticator-02.png]。\n"
    "只能使用知識包中實際出現過的檔名，不可自行編造檔名，也不要使用其他格式標記圖片。\n"
    "回答時請務必涵蓋知識包中所有步驟與對應圖片。"
)


def call_ollama_for_answer(context, question):
    user_prompt = f"請依據以下載入的 OKF 知識庫回答問題。\n\n{context}\n\n用戶問題：{question}"
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_STAGE2_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except requests.exceptions.Timeout:
        return f"❌ Ollama 請求逾時（{OLLAMA_TIMEOUT} 秒）"
    except Exception as e:
        return f"❌ Ollama API 呼叫失敗: {e}"


def call_gemini_for_answer(context, question):
    if gemini_client is None:
        return "❌ Gemini 用戶端未初始化（請檢查 .env 中的 GOOGLE_API_KEY）"
    prompt = f"請依據以下載入的 OKF 知識庫回答問題。\n\n{context}\n\n用戶問題：{question}"
    try:
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=gemini_types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2,
            ),
        )
        return resp.text
    except Exception as e:
        return f"❌ Gemini API 請求失敗: {e}"


def _call_llm(context, question):
    if STAGE2_MODEL == "ollama":
        return call_ollama_for_answer(context, question)
    elif STAGE2_MODEL == "gemini":
        return call_gemini_for_answer(context, question)
    else:
        return f"❌ 未知的 STAGE2_MODEL：{STAGE2_MODEL}"


def _search_and_load(query, top_k):
    retrieve_k = RERANK_RETRIEVE_K if (RERANK_ENABLE and reranker is not None) else top_k
    hits = search_catalog(query, top_k=retrieve_k)
    if not hits:
        return None, None, None
    if RERANK_ENABLE and reranker is not None:
        hits = rerank_catalog(query, hits, top_k)
    filenames = select_parent_documents(hits)
    if not filenames:
        return None, None, None
    ctx, images = load_parent_documents(BUNDLE_DIR, filenames)
    return ctx, images, filenames


def respond(message, history, state):
    if not isinstance(state, dict) or "stage" not in state:
        state.clear()
        state.update(stage=S_IDLE)

    stage = state.get("stage", S_IDLE)

    # ── S_IDLE：初始搜尋 ──
    if stage == S_IDLE:
        ctx, images, filenames = _search_and_load(message, TOP_K_DOCS)
        if ctx is None:
            return "⚠️ 無法在知識庫中找到相關內容（請先執行 train_dsi.py 建立索引）"

        state["stage"] = S_WAIT_CONFIRM
        state["context"] = ctx
        state["allowed_images"] = images
        state["filenames"] = filenames

        output = f"**找到 {len(filenames)} 份相關文件：**\n\n{ctx}"
        output += "\n\n---\n✅ **這些文件是否正確？**\n"
        output += "• 如果是，請直接輸入你想問的問題\n"
        output += "• 如果不是，請輸入「不對」或「重新搜尋」"
        return output

    # ── S_WAIT_CONFIRM：等待確認搜尋結果 ──
    if stage == S_WAIT_CONFIRM:
        if _is_confirm(message):
            state["stage"] = S_WAIT_QUESTION
            return "請輸入你想問的問題："

        if _is_reject(message):
            state["stage"] = S_WAIT_REPHRASE
            return "請重新描述你的問題："

        ctx = state.get("context", "")
        if not ctx:
            state["stage"] = S_IDLE
            return "⚠️ 文件內容遺失，請重新輸入問題。"
        ans = _call_llm(ctx, message)
        ans = images_to_markdown(ans, state.get("allowed_images", {}))
        state["stage"] = S_IDLE
        return ans

    # ── S_WAIT_REPHRASE：等待重新描述問題 ──
    if stage == S_WAIT_REPHRASE:
        ctx, images, filenames = _search_and_load(message, TOP_K_DOCS * 2)
        if not filenames:
            state["stage"] = S_IDLE
            return "⚠️ 重新搜尋後仍找不到相關內容，請更精確描述後再試。"

        candidates = []
        for fn in filenames[:3]:
            md_path = md_index.get(fn)
            if not md_path:
                continue
            cid, meta, body = parse_okf_markdown(md_path, BUNDLE_DIR)
            candidates.append({
                "filename": fn, "title": meta.get("title", ""),
                "description": meta.get("description", ""),
                "body": body, "metadata": meta, "concept_id": cid,
                "md_path": md_path,
            })

        if not candidates:
            state["stage"] = S_IDLE
            return "⚠️ 無法讀取候選文件，請重新輸入問題。"

        state["stage"] = S_SELECTING
        state["candidates"] = candidates

        lines = ["**重新搜尋結果，請選擇最相關的文件：**\n"]
        for i, c in enumerate(candidates, 1):
            desc = (c["description"][:120] + "…") if c["description"] and len(c["description"]) > 120 else (c["description"] or "（無描述）")
            lines.append(f"**{i}.** {c['title']}")
            lines.append(f"   {desc}")
        lines.append("\n請輸入編號(1/2/3)選擇，或輸入「都不對」重新搜尋。")
        return "\n".join(lines)

    # ── S_SELECTING：等待選擇文件編號 ──
    if stage == S_SELECTING:
        msg_s = message.strip()
        if msg_s in {"都不對", "都不是", "重新", "取消", "返回"}:
            state["stage"] = S_IDLE
            return "請重新輸入你的問題："

        try:
            choice = int(msg_s)
        except ValueError:
            return "請輸入有效的編號（1/2/3），或輸入「都不對」重新搜尋。"

        candidates = state.get("candidates", [])
        if choice < 1 or choice > len(candidates):
            return "請輸入 1~{} 之間的編號。".format(len(candidates))

        sel = candidates[choice - 1]
        full = format_concept_segment(sel["concept_id"], sel["metadata"], sel["body"], sel["filename"])

        img_re = re.compile(r"!\[.*?\]\(img/([^)]+)\)")
        imgs = {}
        for m in img_re.finditer(sel["body"]):
            img_name = m.group(1)
            img_full = os.path.normpath(os.path.join(os.path.dirname(sel["md_path"]), "img", img_name))
            if os.path.isfile(img_full):
                imgs[img_name] = img_full

        state["stage"] = S_WAIT_QUESTION
        state["context"] = full
        state["allowed_images"] = imgs
        state["filenames"] = [sel["filename"]]

        return f"**已選擇：{sel['title']}**\n\n{full}\n\n---\n請輸入你想問的問題："

    # ── S_WAIT_QUESTION：等待使用者提問 ──
    if stage == S_WAIT_QUESTION:
        ctx = state.get("context", "")
        if not ctx:
            state["stage"] = S_IDLE
            return "⚠️ 文件內容遺失，請重新輸入問題。"
        ans = _call_llm(ctx, message)
        ans = images_to_markdown(ans, state.get("allowed_images", {}))
        state["stage"] = S_IDLE
        return ans

    state["stage"] = S_IDLE
    return "請輸入問題。"


with gr.Blocks(title="OKF IT 支援助手 (Document Summary Index)") as demo:
    gr.Markdown(
        "# OKF IT 支援助手（Document Summary Index）\n"
        "多輪檢索流程：搜尋 → 確認 → 選擇 → 提問"
    )
    state = gr.State({"stage": S_IDLE})
    gr.ChatInterface(
        fn=respond,
        additional_inputs=[state],
        textbox=gr.Textbox(placeholder="請輸入你的問題…", scale=7),
        title=None,
        description=None,
    )
    gr.Markdown(
        f"**設定：** Embedding `{EMBED_MODEL}` | 回答 `{'Gemini' if STAGE2_MODEL == 'gemini' else OLLAMA_STAGE2_MODEL}` "
        f"| 目錄比對 Top-{TOP_K_DOCS} 份完整文件 | Rerank `{RERANK_MODEL if RERANK_ENABLE and reranker is not None else '✕'}`"
    )

if __name__ == "__main__":
    demo.launch()
