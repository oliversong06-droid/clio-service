# Clio Service – Emotion Diary

A Flask-based journaling experience that analyzes each entry’s emotion, color tone, and notable people. Users log in, write or edit diary entries, and browse their personal archive through a hardcover-book inspired UI with looping BGM.

## Features
- 🔐 **Authentication & Sessions** – Register/login with JSON-backed credentials and CSRF protection.
- 📝 **AI-Assisted Writing** – `our_model/improved_analyzer` infers emotion, color hex/name, tone, and people per entry.
- 📚 **Personal Archive** – Filter by emotion/date/person, open entries in a 3D viewer, and jump between related memories.
- ✏️ **Edit & Delete** – Update or remove any entry from the viewer; edits re-run the analyzer.
- 🎧 **Persistent Atmosphere** – Background music continues across pages via `sessionStorage`.

## Project Structure
```
clio-service/
├─ app.py                # Flask app + routes
├─ requirements.txt      # Python dependencies
├─ templates/            # HTML templates (login, writer, viewer, edit, etc.)
├─ static/bgm.mp3        # Background music
├─ our_model/            # Emotion/color analyzer & datasets
├─ users.json            # Plain JSON user credentials
└─ diaries.json          # Per-user diary entries
```

## Prerequisites
- Python 3.10+ recommended
- `pip` for dependency installation
- Optional: virtualenv/conda for isolated environments

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Server
```bash
python app.py
# or, on Windows, double-click start.bat
```
The Flask server defaults to `http://127.0.0.1:5001` (see `app.py`).

## Data Files
- `users.json`: `{ "username": "hashed-password" }`
- `diaries.json`: `{ "username": [ { entry }, ... ] }`
- Analyzer CSVs: place `emotion_sentimen_dataset.csv`, `your_file_name.csv`, and `name_gender_dataset.csv` inside `our_model/` for full ML/color/person detection. Without them the analyzer falls back to defaults.

Back up `users.json` and `diaries.json` before deployments; the app writes to them directly.

## Customization Tips
- **Theme/BGM**: replace `static/bgm.mp3` or tweak template CSS for new vibes.
- **Analyzer**: adjust logic in `our_model/emotion_model.py` to plug in different models or datasets.
- **Ports/Secrets**: change `app.secret_key` or run behind a production WSGI server (gunicorn/uwsgi) for deployments.

## Troubleshooting
| Issue | Fix |
| --- | --- |
| `ModuleNotFoundError` | Ensure virtualenv active and `pip install -r requirements.txt` succeeded. |
| Analyzer fallback | Missing CSVs? Place required datasets under `our_model/` and restart. |
| Music restarts mid-navigation | Browser must allow autoplay + localStorage; check console for blocked audio logs. |
| JSON corrupted | Restore backups of `users.json` / `diaries.json`. |

Happy journaling! ✨
