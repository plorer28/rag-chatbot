"""화성시 민원 문서만 근거로 답하는 Streamlit RAG 챗봇."""

import hashlib
import os
from pathlib import Path
from typing import Any

import chromadb
import streamlit as st
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "hwaseong_civil_complaints"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4
# Chroma cosine distance: 0에 가까울수록 질문과 문서 내용이 유사합니다.
MAX_RELEVANCE_DISTANCE = 0.55
NO_ANSWER = "자료에서 확인할 수 없습니다"


def require_api_key() -> str:
    """환경변수에서 API 키를 읽고, 없으면 이해하기 쉬운 오류를 냅니다."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY 환경변수가 없습니다. README의 설정 방법을 확인하세요."
        )
    return api_key


def split_text(text: str) -> list[str]:
    """문단 경계를 우선 유지하며 긴 문서를 겹치는 조각으로 나눕니다."""
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if boundary > start + CHUNK_SIZE // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def load_documents() -> tuple[list[str], list[dict[str, str]], list[str]]:
    """data 폴더의 모든 TXT를 읽어 Chroma에 넣을 자료를 만듭니다."""
    documents: list[str] = []
    metadatas: list[dict[str, str]] = []
    ids: list[str] = []

    for file_path in sorted(DATA_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        for index, chunk in enumerate(split_text(text)):
            digest = hashlib.sha1(
                f"{file_path.name}:{index}:{chunk}".encode("utf-8")
            ).hexdigest()
            documents.append(chunk)
            metadatas.append({"source": file_path.name, "chunk": str(index + 1)})
            ids.append(digest)

    if not documents:
        raise RuntimeError("data 폴더에서 TXT 문서를 찾을 수 없습니다.")
    return documents, metadatas, ids


def get_collection(reset: bool = False) -> Any:
    """필요할 때만 문서를 OpenAI 임베딩으로 변환해 ChromaDB에 저장합니다."""
    api_key = require_api_key()
    embedding_function = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBEDDING_MODEL,
    )
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except ValueError:
            pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == 0:
        documents, metadatas, ids = load_documents()
        # 임베딩 API의 요청 크기를 작게 나누어 안정적으로 색인합니다.
        for start in range(0, len(documents), 64):
            collection.add(
                documents=documents[start : start + 64],
                metadatas=metadatas[start : start + 64],
                ids=ids[start : start + 64],
            )
    return collection


@st.cache_resource(show_spinner=False)
def cached_collection() -> Any:
    return get_collection()


def retrieve(question: str) -> list[dict[str, Any]]:
    results = cached_collection().query(
        query_texts=[question],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    matches = []
    for document, metadata, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        matches.append(
            {"document": document, "metadata": metadata, "distance": float(distance)}
        )
    return matches


def generate_answer(question: str, matches: list[dict[str, Any]]) -> str:
    """검색된 문서 조각 외의 지식은 쓰지 않도록 답변 모델에 제한합니다."""
    if not matches or matches[0]["distance"] > MAX_RELEVANCE_DISTANCE:
        return NO_ANSWER

    context = "\n\n".join(
        f"[출처: {match['metadata']['source']} / 조각 {match['metadata']['chunk']}]\n"
        f"{match['document']}"
        for match in matches
    )
    instructions = f"""당신은 화성시 민원 문서 안내 도우미입니다.
반드시 아래 '검색 문서'에 있는 사실만 이용해 한국어로 답하세요.
문서에 질문의 답이 명확히 없으면 정확히 '{NO_ANSWER}'만 답하세요.
문서 밖의 일반 지식, 추측, 인터넷 정보, 전화번호를 덧붙이지 마세요.
답변은 간결하고 이해하기 쉽게 작성하세요.

검색 문서:
{context}"""
    client = OpenAI(api_key=require_api_key())
    response = client.responses.create(
        model=CHAT_MODEL,
        instructions=instructions,
        input=question,
    )
    answer = response.output_text.strip()
    if answer.strip(" .") == NO_ANSWER:
        return NO_ANSWER
    return answer or NO_ANSWER


def main() -> None:
    st.set_page_config(page_title="화성시 민원 챗봇", page_icon="🏙️", layout="centered")
    st.title("🏙️ 화성시 민원 챗봇")
    st.caption("등록된 화성시 민원 문서만 검색해 답변합니다.")

    with st.sidebar:
        st.subheader("문서 관리")
        if st.button("문서 다시 색인", use_container_width=True):
            try:
                with st.spinner("문서를 임베딩해 ChromaDB에 저장하고 있습니다..."):
                    get_collection(reset=True)
                    cached_collection.clear()
                st.success("문서 색인이 완료됐습니다.")
            except Exception as error:
                st.error(f"색인에 실패했습니다: {error}")
        st.caption(f"임베딩: {EMBEDDING_MODEL}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.caption("출처: " + ", ".join(message["sources"]))

    question = st.chat_input("민원 관련 궁금한 점을 입력하세요")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("관련 문서를 찾고 있습니다..."):
                matches = retrieve(question)
                answer = generate_answer(question, matches)
            st.markdown(answer)
            sources = sorted(
                {match["metadata"]["source"] for match in matches}
            ) if answer != NO_ANSWER else []
            if sources:
                st.caption("출처: " + ", ".join(sources))
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as error:
            st.error(f"처리 중 오류가 발생했습니다: {error}")


if __name__ == "__main__":
    main()
