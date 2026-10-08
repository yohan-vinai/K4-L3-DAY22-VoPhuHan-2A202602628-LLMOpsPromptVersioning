"""
Tiện ích để tải và xử lý dữ liệu cho RAG pipeline.

Cách dùng:
    from utils.data_loader import load_knowledge_base, split_text, build_vectorstore

    text        = load_knowledge_base()
    chunks      = split_text(text, chunk_size=500, chunk_overlap=50)
    vectorstore = build_vectorstore(chunks, embeddings)
"""
from pathlib import Path
import time


def load_knowledge_base(path: str = None) -> str:
    """
    Đọc file knowledge base và trả về nội dung dạng chuỗi.

    Args:
        path: đường dẫn tới file text.
              Mặc định: data/knowledge_base.txt (thư mục gốc của project)

    Returns:
        Nội dung file dưới dạng str
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "knowledge_base.txt"
    return Path(path).read_text(encoding="utf-8")


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Chia văn bản thành các đoạn nhỏ (chunks) để index.

    Dùng RecursiveCharacterTextSplitter — tách ưu tiên theo đoạn văn, câu, rồi ký tự.

    Args:
        text         : văn bản cần chia
        chunk_size   : số ký tự tối đa mỗi chunk (mặc định: 500)
        chunk_overlap: số ký tự chồng lên nhau giữa 2 chunks liên tiếp (mặc định: 50)

    Returns:
        list[str] — danh sách các chuỗi chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def build_vectorstore(chunks: list, embeddings):
    """
    Tạo FAISS vectorstore từ danh sách chunks và embeddings.

    Args:
        chunks    : list[str] — danh sách text chunks đã chia
        embeddings: Embeddings instance (từ get_embeddings())

    Returns:
        FAISS vectorstore đã được index và sẵn sàng dùng để retrieve
    """
    from langchain_community.vectorstores import FAISS

    print(f"🔨 Đang tạo FAISS index từ {len(chunks)} chunks ...")
    try:
        from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
    except ImportError:
        GoogleGenerativeAIEmbeddings = ()

    if isinstance(embeddings, GoogleGenerativeAIEmbeddings):
        # Gemini's free tier is limited to 100 embedded inputs per minute.
        # Keep the lab's required 500/50 chunking and pace batches when needed.
        batch_size = 100
        vectors = []
        for start in range(0, len(chunks), batch_size):
            if start:
                print("⏳ Đợi 62 giây để tuân thủ quota embedding Gemini ...")
                time.sleep(62)
            batch = chunks[start : start + batch_size]
            for attempt in range(3):
                try:
                    vectors.extend(embeddings.embed_documents(batch))
                    break
                except Exception as exc:
                    if "RESOURCE_EXHAUSTED" not in str(exc) or attempt == 2:
                        raise
                    print("⏳ Gemini embedding đang chạm quota; đợi 62 giây rồi thử lại ...")
                    time.sleep(62)
        vectorstore = FAISS.from_embeddings(zip(chunks, vectors), embeddings)
        print("✅ FAISS vectorstore đã sẵn sàng.")
        return vectorstore

    vectorstore = FAISS.from_texts(chunks, embeddings)
    print("✅ FAISS vectorstore đã sẵn sàng.")
    return vectorstore
