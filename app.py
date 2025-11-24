from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import json
import os
from datetime import datetime
# 님 기존 코드의 AI 모델 불러오기
from our_model.emotion_model import improved_analyzer 

app = Flask(__name__)
app.secret_key = 'acdt_secret_key_1234'  # 세션 암호화 키 (필수)

# --- 경로 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIARY_FILE = os.path.join(BASE_DIR, 'diaries.json')
USER_FILE = os.path.join(BASE_DIR, 'users.json') # 사용자 정보 저장 파일

# --- [Helper] 데이터 로드/저장 함수 (통합됨) ---
def load_data(filename, default_type=dict):
    if not os.path.exists(filename):
        return default_type()
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return default_type()

def save_data(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# --- [Helper] 날짜 변환 ---
def format_english_date(date_str):
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.strftime('%B %d, %Y')
    except ValueError:
        return date_str 

def get_contrast_text(hex_color):
    if not hex_color:
        return '#111111', False
    value = hex_color.lstrip('#')
    if len(value) != 6:
        return '#111111', False
    try:
        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)
    except ValueError:
        return '#111111', False
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    is_dark_bg = luminance < 0.45
    text_color = '#FFFFFF' if is_dark_bg else '#111111'
    return text_color, is_dark_bg

# --- [Helper] 감정 이모지 매핑 ---
EMOTION_EMOJI = {
    'Happiness': '😊',
    'Sadness': '😢',
    'Anger': '😠',
    'Fear': '😨',
    'Disgust': '🤢',
    'Surprise': '😲'
}

def get_user_people(user_diaries):
    people_set = set()
    for diary in user_diaries:
        if 'people' in diary:
            for p in diary['people']:
                people_set.add(p['name'])
    return sorted(list(people_set))


# ================= 라우팅 (Routes) =================

# 1. [로그인 페이지] 앱 접속 시 첫 화면
@app.route('/', methods=['GET', 'POST'])
def login_page():
    # 이미 로그인 되어있으면 바로 글쓰기 화면으로
    if 'user' in session:
        return redirect(url_for('write_diary'))
    return render_template('login.html') # 새로 만든 login.html

# 2. [기능] 로그인 처리
@app.route('/login', methods=['POST'])
def login():
    users = load_data(USER_FILE, dict)
    username = request.form.get('username')
    password = request.form.get('password')
    
    if username in users and users[username] == password:
        session['user'] = username
        return redirect(url_for('write_diary'))
    else:
        return "<script>alert('아이디 또는 비밀번호가 틀렸습니다.'); location.href='/';</script>"

# 3. [기능] 회원가입 처리
@app.route('/register', methods=['POST'])
def register():
    users = load_data(USER_FILE, dict)
    username = request.form.get('username')
    password = request.form.get('password')
    
    if username in users:
        return "<script>alert('이미 존재하는 이름입니다.'); location.href='/';</script>"
    
    # 사용자 저장
    users[username] = password
    save_data(USER_FILE, users)
    
    # 다이어리 데이터에도 빈 방(List) 만들기
    all_diaries = load_data(DIARY_FILE, dict)
    if username not in all_diaries:
        all_diaries[username] = []
    save_data(DIARY_FILE, all_diaries)
    
    return "<script>alert('등록 완료! 로그인해주세요.'); location.href='/';</script>"

# 4. [기능] 로그아웃
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login_page'))

# 5. [메인] 일기 작성 페이지 (로그인 한 사람만 접근 가능)
@app.route('/write')
def write_diary():
    if 'user' not in session:
        return redirect(url_for('login_page'))
    
    # index.html에 사용자 이름도 같이 보내줌 (헤더 표시용)
    return render_template('index.html', user=session['user'])

# 6. [기능] 분석 및 저장 (핵심 로직 통합)
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'user' not in session:
        return redirect(url_for('login_page'))

    current_user = session['user']
    
    # 폼 데이터 받기
    raw_date = request.form['date']
    diary_text = request.form['diary']
    english_date = format_english_date(raw_date)
    
    # --- [AI 모델 사용] 님 코드 그대로 유지 ---
    result = improved_analyzer.analyze_emotion_and_color(diary_text)
    
    # 저장할 데이터 구조 생성
    new_entry = {
        'date': english_date,
        'text': diary_text,
        'emotion': result['emotion'],
        'color': result['color_hex'],
        'color_name': result['color_name'],
        'tone': result['tone'],
        'people': result['people']
    }
    
    # --- [저장 로직 변경] 사용자별 방에 저장 ---
    all_diaries = load_data(DIARY_FILE, dict)
    
    # 혹시 사용자 방이 없으면 생성 (에러 방지)
    if current_user not in all_diaries:
        all_diaries[current_user] = []
        
    all_diaries[current_user].append(new_entry) # 리스트에 추가
    save_data(DIARY_FILE, all_diaries) # 전체 저장
    
    print(f"✅ Saved for {current_user}: {english_date}")
    
    return render_template('result.html', 
                           date=english_date,
                           text=diary_text,
                           emotion=result['emotion'],
                           color=result['color_hex'],
                           color_name=result['color_name'],
                           tone=result['tone'],
                           people=result['people'])

# 7. [히스토리] 내 일기만 보기 + 필터링
@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login_page'))
        
    current_user = session['user']
    
    # 1. 전체 데이터 불러오기
    all_data = load_data(DIARY_FILE, dict)
    
    # 2. [중요] '내 일기'만 꺼내오기 (없으면 빈 리스트)
    user_diaries = all_data.get(current_user, [])
    indexed_diaries = list(enumerate(user_diaries))
    indexed_diaries.reverse() # 최신순 정렬
    
    # 3. 필터 조건 받기
    filter_emotion = request.args.get('emotion')
    filter_date = request.args.get('date')
    filter_person = request.args.get('person')

    # 4. 내 일기에서 등장인물 목록 뽑기
    my_people_list = get_user_people(user_diaries)

    filtered_diaries = []
    
    # 5. 필터링 로직 (님 코드 로직 유지)
    for original_idx, diary in indexed_diaries:
        # 감정 필터
        if filter_emotion and filter_emotion != "All" and diary['emotion'] != filter_emotion:
            continue
        
        # 날짜 필터
        if filter_date:
            english_filter_date = format_english_date(filter_date)
            if diary['date'] != english_filter_date:
                continue
        
        # 인물 필터
        if filter_person and filter_person != "All":
            diary_people_names = [p['name'] for p in diary.get('people', [])]
            if filter_person not in diary_people_names:
                continue
            
        decorated_diary = dict(diary)
        decorated_diary['entry_index'] = original_idx
        text_color_for_card, _ = get_contrast_text(diary.get('color', '#f5f1e6'))
        decorated_diary['text_color'] = text_color_for_card
        decorated_diary['emotion_icon'] = EMOTION_EMOJI.get(diary.get('emotion'), '🎭')
        filtered_diaries.append(decorated_diary)

    return render_template('history.html', 
                           diaries=filtered_diaries,
                           all_people=my_people_list, # 내 친구 목록만 전달
                           current_emotion=filter_emotion,
                           current_date=filter_date,
                           current_person=filter_person,
                           user=current_user) # 사용자 이름도 전달

@app.route('/viewer/<int:entry_index>')
def view_diary(entry_index):
    if 'user' not in session:
        return redirect(url_for('login_page'))

    current_user = session['user']
    all_diaries = load_data(DIARY_FILE, dict)
    user_diaries = all_diaries.get(current_user, [])

    if entry_index < 0 or entry_index >= len(user_diaries):
        return redirect(url_for('history'))

    entry = user_diaries[entry_index]
    base_color = entry.get('color', '#3e2723')
    text_color, is_dark_bg = get_contrast_text(base_color)

    related_entries = []
    for idx in range(len(user_diaries) - 1, -1, -1):
        if idx == entry_index:
            continue
        diary = user_diaries[idx]
        if diary.get('emotion') == entry.get('emotion'):
            related_entries.append({
                'date': diary.get('date'),
                'text': diary.get('text'),
                'color': diary.get('color'),
                'entry_index': idx
            })

    return render_template('viewer.html',
                           entry=entry,
                           text_color=text_color,
                           is_dark_bg=is_dark_bg,
                           related_entries=related_entries,
                           entry_index=entry_index,
                           user=current_user)

if __name__ == '__main__':
    app.run(debug=True, port=5001) # 포트 충돌 방지 5001
