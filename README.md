# RAG 기반 화성시 민원 챗봇

`data/` 폴더의 화성시 민원 TXT 문서를 OpenAI 임베딩으로 벡터화해 ChromaDB에 저장하고, 질문과 유사한 문서 내용만 근거로 답하는 Streamlit 앱입니다.

## 기능

- `data/`의 모든 `.txt` 문서를 조각으로 나누어 ChromaDB에 영속 저장
- OpenAI `text-embedding-3-small` 임베딩으로 유사 문서 검색
- 검색 문서 내용만 바탕으로 답변 생성
- 충분히 관련된 문서를 찾지 못하면 `자료에서 확인할 수 없습니다`라고 답변
- 답변 아래에 참고한 출처 파일명 표시
- 사이드바의 **문서 다시 색인** 버튼으로 데이터 변경사항 반영

## 설치

Python 3.10 이상을 권장합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## OpenAI API 키 설정

API 키를 코드에 넣지 말고, 실행 전에 환경변수로 설정하세요.

PowerShell에서 현재 창에만 설정하려면:

```powershell
$env:OPENAI_API_KEY = "발급받은_API_키"
```

Windows 사용자 환경변수로 저장하려면:

```powershell
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "발급받은_API_키", "User")
```

두 번째 방법을 썼다면 새 PowerShell 창을 열어야 적용됩니다. 또는 `.env.example`을 복사해 `.env` 파일을 만들고 키를 입력할 수 있습니다. `.env`는 Git에서 제외되며, 앱이 실행될 때 자동으로 읽습니다.

답변 모델은 기본값 `gpt-4.1-mini`를 사용합니다. 다른 모델을 쓰려면 `OPENAI_CHAT_MODEL` 환경변수를 설정하세요.

## 실행

```powershell
python -m streamlit run app.py
```

처음 실행하거나 **문서 다시 색인**을 누르면 OpenAI 임베딩 API가 호출되고, 결과는 로컬 `chroma_db/`에 저장됩니다. 이후 같은 문서로 실행할 때는 저장된 벡터 DB를 재사용합니다.

## 프로젝트 구조

```text
.
├── app.py              # Streamlit RAG 앱
├── requirements.txt    # 필요한 패키지
├── data/               # 검색 대상 화성시 민원 문서
└── chroma_db/          # 실행 후 생성되는 벡터 DB (Git 제외)
```
