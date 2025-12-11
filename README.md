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
