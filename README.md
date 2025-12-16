# Clio Service – 감정 다이어리

Flask 기반으로 구현된 인터랙티브 일기장입니다. 사용자는 로그인 후 감정 분석 기반 일기를 작성·편집하고, 고서 느낌의 UI에서 자신의 기록을 아카이브 형태로 탐색할 수 있습니다. 페이지를 이동해도 잔잔한 BGM이 이어지며, 감정이 무거울 때는 마음 돌봄 안내를 제공합니다.

## 주요 기능
- 🔐 **인증 & 세션 관리** – JSON 파일을 이용한 회원가입/로그인, 비밀번호 해시 및 CSRF 보호 적용.
- 📝 **AI 감정 분석** – `our_model/improved_analyzer`가 감정, 색상 HEX/이름, 톤, 등장인물을 추출.
- 📚 **개인 아카이브** – 감정/날짜/인물로 필터링하고, 3D 뷰어에서 관련 기록과 함께 열람.
- ✏️ **수정/삭제** – 뷰어에서 바로 편집 및 삭제 가능하며, 수정 시 AI 분석 재실행.
- 🎧 **배경 음악 지속** – `static/js/bgm-controller.js`가 페이지 이동 시에도 음악을 끊김 없이 유지.
- 🛡️ **마음 케어 알림** – 무거운 감정 감지 시 위기 대응 연락처, 주변 상담소 검색 링크 노출.

## 프로젝트 구조
```
clio-service/
├─ app.py                # Flask 진입점 및 라우트
├─ requirements.txt      # 파이썬 의존성
├─ templates/            # HTML 템플릿 (로그인, 작성, 아카이브, 뷰어, 수정 등)
├─ static/bgm.mp3        # 배경 음악 파일
├─ static/js/bgm-controller.js  # 공용 BGM 제어 스크립트
├─ our_model/            # 감정/색상 분석 모델 및 데이터셋
├─ users.json            # 사용자 계정(아이디: 해시 비밀번호)
└─ diaries.json          # 사용자별 일기 데이터
```

## 준비 사항
- Python 3.10 이상 권장
- `pip` 패키지 관리자
- (선택) `venv` 또는 Conda 가상환경

## 설치 방법
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 서버 실행
```bash
python app.py
# 또는 Windows에서 start.bat 실행
```
기본 접속 주소는 `http://127.0.0.1:5001` 입니다. 포트를 바꾸려면 `app.py` 하단을 수정하세요.

## 데이터 파일 설명
- `users.json`: `{"username": "<hashed-password>"}` 형태로 사용자 정보를 저장.
- `diaries.json`: 사용자별 일기 배열을 저장하는 JSON.
- 분석용 CSV: `emotion_sentimen_dataset.csv`, `your_file_name.csv`, `name_gender_dataset.csv` 등을 `our_model/`에 배치해야 전체 기능이 동작합니다. 없으면 기본값으로 대체됩니다.

배포 전에 `users.json`, `diaries.json` 백업을 권장합니다. 앱에서 바로 덮어쓰므로 복구 지점이 필요합니다.

## 커스터마이징 팁
- **테마/BGM**: `static/bgm.mp3`를 교체하거나 템플릿 CSS를 수정해 분위기를 바꿀 수 있습니다.
- **분석 모델**: `our_model/emotion_model.py`에서 알고리즘이나 데이터셋을 교체하세요.
- **보안/포트**: `app.secret_key`를 환경 변수로 분리하고, 실서비스에서는 gunicorn/uwsgi 같은 WSGI 서버 뒤에 배치하는 것이 좋습니다.

## 자주 묻는 문제
| 증상 | 해결 방법 |
| --- | --- |
| `ModuleNotFoundError` | 가상환경 활성화 여부와 `pip install -r requirements.txt` 실행을 확인하세요. |
| 분석 결과가 기본값만 출력됨 | 필요한 CSV가 `our_model/`에 있는지 확인 후 서버를 재실행하세요. |
| 페이지 이동 시 음악이 끊김 | 브라우저 자동재생 허용 여부와 콘솔 로그를 확인하세요. `sessionStorage`가 차단되면 지속 재생이 안 됩니다. |
| JSON 파일 손상 | 사전에 백업한 `users.json`/`diaries.json`으로 복원하세요. |

## 빠른 사용 흐름
1. 서버 실행 후 `http://127.0.0.1:5001` 접속.
2. 회원가입(정보는 `users.json`에 저장) 후 로그인.
3. `/write`에서 일기 작성 → `/result`에서 AI 분석 확인.
4. `/history`에서 감정 혹은 등장인물로 필터링, 특정 일기는 `/viewer/<entry_id>`로 열람.
5. 뷰어에서 바로 수정 및 삭제 가능.

## 주요 라우트
| 경로 | 메서드 | 설명 |
| --- | --- | --- |
| `/` | GET | 로그인 페이지 (로그인 상태면 `/write`로 리다이렉트). |
| `/login` | POST | 로그인 처리. |
| `/register` | POST | 회원가입 후 사용자 전용 일기 배열 생성. |
| `/write` | GET | 일기 작성 UI + BGM 시작. |
| `/analyze` | POST | 일기 저장 및 AI 분석 후 `/result`로 이동. |
| `/result` | GET | 최신 분석 결과 뷰. |
| `/history` | GET | 아카이브 목록, 쿼리 파라미터 `emotion`, `person`, `date` 지원. |
| `/viewer/<entry_id>` | GET | 책 넘김 스타일의 상세 뷰어. |
| `/edit/<entry_id>` | GET/POST | 일기 편집 후 재분석. |
| `/delete/<entry_id>` | POST | 일기 삭제(뷰어에서 버튼 제공). |
| `/logout` | GET | 세션 종료 후 로그인 페이지로 이동. |

## 데이터 스키마
- 기본 일기 객체 예시:
  ```json
  {
    "id": "랜덤 HEX",
    "date": "January 15, 2024",
    "text": "일기 내용",
    "emotion": "Happiness",
    "color": "#aabbcc",
    "color_name": "Cornflower",
    "tone": "Calm",
    "people": [
      {"name": "Alice", "color": "#f06292"}
    ]
  }
  ```
- 날짜 포맷 변환은 `format_english_date` / `parse_english_date`에 정의되어 있으니 일관성을 유지하세요.
- 스키마를 변경할 경우 `app.py`의 저장/로드 로직과 관련 템플릿(History, Viewer, Result, Edit)을 모두 업데이트해야 합니다.

## AI 분석 파이프라인 자세히 보기
1. `/analyze` 라우트(`app.py`)는 날짜를 영문 표기로 바꾼 뒤, `our_model/emotion_model.py`의 `ImprovedEmotionAnalyzer`에 원문을 전달합니다. 빈값 검증과 CSRF 토큰 확인을 가장 먼저 수행해 악의적 요청을 차단합니다.
2. 모델은 본문을 문장 단위로 분할한 뒤 길이·순서를 기반으로 가중치를 두고 흐름(`emotion_flow`)을 계산합니다. 각 블록은 세 가지 엔진의 투표를 받습니다.
   - **Keyword 규칙 엔진**: 감정을 대표하는 단어 목록과 출현 빈도를 계산합니다.
   - **TF-IDF + 로지스틱 회귀**: `emotion_sentimen_dataset.csv`에서 추출된 텍스트를 벡터화하여 감정 레이블을 예측합니다(`max_features=5000`, `class_weight='balanced'`).
   - **DistilBERT 파인튜닝**: `bert_finetuned/` 폴더에 저장된 체크포인트를 불러오거나, 없을 경우 동일 데이터셋으로 미세조정합니다. 토큰은 256자로 잘리고 GPU 사용이 가능하면 자동으로 CUDA로 옮겨집니다.
   - 보조로 **이모지 가중치**(`emoji_emotion_large.csv`), **리스크 키워드**(`depression_risk_words.csv`), **이름 사전**(`name_gender_dataset.csv`)을 불러와 감정과 등장인물, Mind-Care 플래그를 산출합니다.
3. 최종 감정은 가중치 합산으로 결정되고, 같은 가중치를 바탕으로 0~100% 그라데이션(`emotion_gradient`)이 만들어져 뷰어 차트에 사용됩니다.
4. 색상은 `your_file_name.csv`의 HSV → 감정 RandomForest 모델을 먼저 조회하고, 실패하면 사전에 정의된 팔레트로 폴백합니다. 부정적인 감정은 명도·채도를 자동 조정해서 다크 톤으로, 긍정은 파스텔 톤으로 재조정합니다.
5. `analyze_people()`는 문장에서 이름처럼 보이는 토큰을 찾아 해당 부분만 다시 감정 분석하여 등장인물 목록을 만들어냅니다. 호칭·대명사는 `pronoun_aliases`로 정규화됩니다.
6. `check_mind_care_needed()`는 CSV + 기본 세트에 포함된 위험 어휘를 정규화해 비교하고, 일치하면 결과 페이지 모달에 위기대응 가이드를 띄웁니다.
7. 분석 결과는 항상 먼저 `diaries.json`에 저장된 후 뷰에 전달되어, 추후 재분석 버튼(`/reanalyze`)으로 같은 알고리즘을 일괄 재실행할 수 있습니다.

## 학습 & 모델 리소스
- 텍스트 분류: `TfidfVectorizer`와 로지스틱 회귀를 `train_test_split(stratify)`로 학습/검증하고, 학습 정확도를 서버 부팅 시 로그로 남깁니다.
- 색상 추천: `your_file_name.csv`의 HSV 값과 감정 라벨을 LabelEncoder로 정수화한 뒤, `RandomForestClassifier(n_estimators=100)`로 감정→색상 매핑을 학습합니다.
- DistilBERT 파인튜닝: `EMOTION_BERT_*` 환경 변수로 epoch, batch, lr를 조정할 수 있으며, 10%~20%를 검증셋으로 남겨 loss/val_acc를 매 epoch마다 출력합니다. 학습 완료 후 `bert_finetuned/config.json`이 존재하면 이후에는 캐시를 그대로 불러옵니다.
- 리스크 어휘/이모지/이름 데이터는 CSV를 느슨하게 파싱하도록 되어 있어, 헤더명이 바뀌면 `emotion_model.py`에서 열 이름만 수정하면 됩니다.
- `model_cache.pkl` 등 캐시 파일을 만들어두면 재부팅 시 로딩 시간이 짧아지고, 감정 사전은 `_emotion_dataset_cache`로 메모리에 유지됩니다.

## 일기 저장·수정·삭제 및 백그라운드 처리
- **저장**: `/analyze`는 AI 결과를 얻은 뒤 사용자별 배열에 `id`를 붙여 `diaries.json`에 즉시 덮어씁니다. 저장 시점에 Mind-Care 여부와 감정 흐름도 함께 기록되어 뷰어에서 재활용됩니다.
- **수정**: `/entry/<id>/edit` GET에서 ISO 날짜를 돌려주고, POST에서는 다시 분석을 실행해 감정·색상·인물 정보를 업데이트합니다. 기존 `id`는 유지되어 뷰어 링크가 깨지지 않습니다.
- **삭제**: `/entry/<id>/delete`는 `find_entry_index()`로 원소를 찾아낸 뒤 리스트에서 제거하고 JSON 전체를 다시 저장합니다. CSRF 토큰을 필수로 요구해 외부 사이트에서 삭제를 트리거할 수 없습니다.
- **재분석**: `/reanalyze`는 AJAX로 호출되며, 저장된 모든 일기에 대해 최신 모델을 다시 돌립니다. 응답에는 업데이트된 건수/총 건수가 포함되어 있어 프런트에서 진행 상황을 표시할 수 있습니다.
- **백그라운드 동작**: Flask는 세션 쿠키와 `session['csrf_token']`으로 요청별 상태를 추적하고, `static/js/bgm-controller.js`는 `sessionStorage`에 재생 위치를 저장/복원하여 페이지 이동 중에도 음악을 이어줍니다.

## 뷰어와 테마 색상 제어
- `/view/<id>`는 대상 일기를 찾은 뒤, 같은 감정을 공유하는 다른 일기를 `related` 목록으로 제공하여 3D 뷰어 우측 탭에서 빠르게 탐색하게 합니다.
- 테마색은 저장된 `color`를 기반으로 `blend_with_white()`로 배경을 밝게 만들고, `invert_hex_color()`로 텍스트 대비를 자동 결정합니다. 감정 흐름에서 생성한 `emotion_gradient`는 배경 리본/차트에 사용되고, Mind-Care 플래그가 true일 경우 모달을 즉시 띄웁니다.
- Viewer 템플릿은 페이지 입장 시 선택된 테마를 body data-attribute로 박아 CSS 전환 효과를 주며, `info/list` 탭을 통해 연관 일기를 같은 팔레트 안에서 열람합니다.

## 분석 리소스
- `our_model/emotion_model.py`가 사전 학습된 모델과 보조 CSV를 로드합니다.
- CSV는 Git에 포함되지 않으므로 운영 환경에도 동일하게 배포하세요.
- API: `improved_analyzer.analyze_emotion_and_color(text: str)` → `/analyze` 엔드포인트에서 바로 사용.

## 프런트엔드 메모
- 템플릿은 Google Fonts와 인라인 CSS로 빈티지 책 디자인을 구현했습니다.
- `static/js/bgm-controller.js`는 BGM 시간대를 `sessionStorage`에 저장하고, 페이지 복귀 시 자동으로 이어서 재생합니다.
- BGM을 교체하려면 `static/bgm.mp3`만 바꾸면 됩니다.
- 다국어 지원이 필요하면 `templates/` 내 문구를 수정하고 날짜 포맷 로직을 점검하세요.

### BGM 컨트롤러 작동 원리
1. 페이지 진입 시 `BgmController.init()`이 실행되어 `sessionStorage`에 저장된 재생 위치/의도를 복구합니다.
2. `autoplay` 허용 시 즉시 재생하고, 차단되면 사용자 제스처(터치/클릭)를 기다립니다.
3. 1.6초 간격으로 현재 시간을 저장하고, `visibilitychange`, `pagehide`, `beforeunload` 이벤트에서 추가 저장을 수행합니다.
4. `pageshow` 이벤트에서 자동으로 재시도하여 히스토리 이동(back/forward) 후에도 음악이 연결됩니다.

### UI 세부 팁
- `/write` 페이지에서 표지를 클릭하면 3D 책이 열리고 곧바로 음악이 재생됩니다.
- `/history` 카드에 마우스를 올리면 집중 모드가 켜져 주변이 어두워집니다.
- `/viewer` 우측 탭에서 Info/List를 빠르게 전환하며 관련 일기를 탐색할 수 있습니다.
- Mind-Care 모달은 감정 분석 결과에 따라 자동 표출되며, 오버레이를 클릭하면 닫힙니다.

## 배포 권장 사항
- 실제 서비스에서는 gunicorn/waitress 등으로 Flask를 감싸고, 정적 파일은 Nginx 혹은 CDN에서 서빙하세요.
- `users.json`/`diaries.json`을 영구 스토리지(EFS, NAS 등)에 두거나, 추후 SQLite/Postgres로 이전을 고려하세요.
- `app.secret_key`는 환경 변수로 관리하고 주기적으로 교체하세요.
- HTTPS를 활성화해 폰트·음악 등 외부 리소스가 안전하게 로드되도록 합니다.

### 환경 변수/설정
| 키 | 설명 | 예시 |
| --- | --- | --- |
| `FLASK_APP` | Flask 엔트리 포인트 지정 | `export FLASK_APP=app.py` |
| `FLASK_ENV` | 개발/프로덕션 모드 | `development` |
| `SECRET_KEY` | `app.secret_key` 대체용 환경 변수 (직접 코드에 주입 필요) | `export SECRET_KEY='...'` |
| `PORT` | 필요 시 `app.run(port=PORT)` 형태로 전달 | `.env` 파일에 저장 후 불러오기 |

환경 변수를 쓰려면 `python-dotenv`를 설치하고 `app.py`에서 `os.getenv()`로 읽어오면 됩니다.

### 백업 & 마이그레이션
1. 서버 중지 후 `users.json`, `diaries.json`을 안전한 위치에 복사합니다.
2. 버전 관리를 위해 날짜와 커밋 해시를 파일명에 함께 기록하세요.
3. SQLite/Postgres로 옮길 경우 JSON 구조를 테이블 스키마로 변환한 뒤, Flask에서 ORM(SQLAlchemy 등)을 사용하도록 리팩토링하면 됩니다.

### 로그 & 모니터링
- Flask 기본 로거는 stdout으로 출력되므로, 운영 환경에서는 `gunicorn --access-logfile - --error-logfile -` 형식으로 수집합니다.
- 분석 실패나 CSV 누락 등의 예외를 `try/except`로 포착하고 `logging.warning`을 활용하세요.
- 음악 재생 문제는 브라우저 콘솔 로그에 `BGM autoplay blocked`가 찍히므로, 해당 메시지를 모니터링하여 UX 이슈를 파악할 수 있습니다.

## 개발 스크립트
- `python app.py` – 기본 개발 서버 실행(자동 리로드 포함).
- `python -m unittest` – 테스트 폴더를 추가했다면 여기서 실행.
- 린팅: `flake8`, `black` 등을 `requirements.txt`에 추가한 뒤 수동으로 돌립니다.

행복한 기록 되세요! ✨
