from flask import Flask, render_template, request, jsonify, session, redirect, url_for, abort
import json
import os
from datetime import datetime
import secrets
# 님 기존 코드의 AI 모델 불러오기
from our_model.emotion_model import improved_analyzer 
from werkzeug.security import check_password_hash, generate_password_hash

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

# --- [Helper] 특정 사용자의 등장인물만 모으기 ---
def get_user_people(user_diaries):
    people_set = set()
    for diary in user_diaries:
        if 'people' in diary:
            for p in diary['people']:
                people_set.add(p['name'])
    return sorted(list(people_set))

def get_csrf_token():
    token = session.get('csrf_token')
    if not token:
        token = secrets.token_hex(16)
        session['csrf_token'] = token
    return token

def validate_csrf():
    token = session.get('csrf_token')
    form_token = request.form.get('csrf_token')
    if not token or not form_token or token != form_token:
        abort(400, description='Invalid CSRF token')

def parse_english_date(date_str):
    try:
        date_obj = datetime.strptime(date_str, '%B %d, %Y')
        return date_obj.strftime('%Y-%m-%d')
    except ValueError:
        return date_str

def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def _rgb_to_hex(rgb_tuple):
    return '#{:02x}{:02x}{:02x}'.format(*rgb_tuple)

def invert_hex_color(hex_color):
    try:
        r, g, b = _hex_to_rgb(hex_color)
    except ValueError:
        return '#ffffff'
    return _rgb_to_hex((255 - r, 255 - g, 255 - b))

def blend_with_white(hex_color, factor=0.35):
    try:
        r, g, b = _hex_to_rgb(hex_color)
    except ValueError:
        return '#ffffff'
    new_r = int(r + (255 - r) * factor)
    new_g = int(g + (255 - g) * factor)
    new_b = int(b + (255 - b) * factor)
    return _rgb_to_hex((new_r, new_g, new_b))

def find_entry_index(entries, entry_id):
    for idx, entry in enumerate(entries):
        if entry.get('id') == entry_id:
            return idx
    return None


# ================= 라우팅 (Routes) =================

# 1. [로그인 페이지] 앱 접속 시 첫 화면
@app.route('/', methods=['GET', 'POST'])
def login_page():
    # 이미 로그인 되어있으면 바로 글쓰기 화면으로
    if 'user' in session:
        return redirect(url_for('write_diary'))
    return render_template('login.html', csrf_token=get_csrf_token()) # 새로 만든 login.html

# 2. [기능] 로그인 처리
@app.route('/login', methods=['POST'])
def login():
    validate_csrf()
    users = load_data(USER_FILE, dict)
    username = request.form.get('username')
    password = request.form.get('password')
    
    if username in users and check_password_hash(users[username], password):
        session['user'] = username
        return redirect(url_for('write_diary'))
    else:
        return "<script>alert('아이디 또는 비밀번호가 틀렸습니다.'); location.href='/';</script>"

# 3. [기능] 회원가입 처리
@app.route('/register', methods=['POST'])
def register():
    validate_csrf()
    users = load_data(USER_FILE, dict)
    username = request.form.get('username')
    password = request.form.get('password')
    
    if username in users:
        return "<script>alert('이미 존재하는 이름입니다.'); location.href='/';</script>"
    
    # 사용자 저장
    users[username] = generate_password_hash(password)
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
    return render_template('index.html', user=session['user'], csrf_token=get_csrf_token())

# 6. [기능] 분석 및 저장 (핵심 로직 통합)
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'user' not in session:
        return redirect(url_for('login_page'))

    validate_csrf()

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
        'people': result['people'],
        'id': secrets.token_hex(8)
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
                           people=result['people'],
                           emotion_flow=result.get('emotion_flow', []),
                           emotion_gradient=result.get('emotion_gradient'))

# 7. [히스토리] 내 일기만 보기 + 필터링
@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login_page'))
        
    current_user = session['user']
    
    # 1. 전체 데이터 불러오기
    all_data = load_data(DIARY_FILE, dict)
    
    # 2. [중요] '내 일기'만 꺼내오기 (없으면 빈 리스트)
    my_diaries = all_data.get(current_user, [])
    data_changed = False
    for entry in my_diaries:
        if 'id' not in entry:
            entry['id'] = secrets.token_hex(8)
            data_changed = True
    if data_changed:
        save_data(DIARY_FILE, all_data)
    my_diaries.reverse() # 최신순 정렬
    
    # 3. 필터 조건 받기
    filter_emotion = request.args.get('emotion')
    filter_date = request.args.get('date')
    filter_person = request.args.get('person')

    # 4. 내 일기에서 등장인물 목록 뽑기
    my_people_list = get_user_people(my_diaries)

    filtered_diaries = []
    
    # 5. 필터링 로직 (님 코드 로직 유지)
    for diary in my_diaries:
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
            
        filtered_diaries.append(diary)

    return render_template('history.html', 
                           diaries=filtered_diaries,
                           all_people=my_people_list, # 내 친구 목록만 전달
                           current_emotion=filter_emotion,
                           current_date=filter_date,
                           current_person=filter_person,
                           user=current_user) # 사용자 이름도 전달

@app.route('/view/<entry_id>')
def view_entry(entry_id):
    if 'user' not in session:
        return redirect(url_for('login_page'))

    current_user = session['user']
    all_data = load_data(DIARY_FILE, dict)
    user_diaries = all_data.get(current_user, [])
    data_changed = False
    for entry in user_diaries:
        if 'id' not in entry:
            entry['id'] = secrets.token_hex(8)
            data_changed = True
    if data_changed:
        save_data(DIARY_FILE, all_data)

    reversed_diaries = list(reversed(user_diaries))

    selected_entry = next((entry for entry in reversed_diaries if entry.get('id') == entry_id), None)
    if not selected_entry:
        return redirect(url_for('history'))

    related_entries = [
        entry for entry in reversed_diaries
        if entry.get('emotion') == selected_entry.get('emotion') and entry.get('id') != entry_id
    ]

    theme_color = selected_entry.get('color', '#e6dec8')
    page_bg = blend_with_white(theme_color, factor=0.45)
    text_color = invert_hex_color(page_bg)

    _, viewer_emotion_flow, viewer_gradient = improved_analyzer.analyze_emotion_with_flow(selected_entry.get('text', ''))

    return render_template('viewer.html',
                           user=current_user,
                           selected=selected_entry,
                           related=related_entries[:5],
                           theme_color=theme_color,
                           page_bg=page_bg,
                           text_color=text_color,
                           csrf_token=get_csrf_token(),
                           emotion_flow=viewer_emotion_flow,
                           emotion_gradient=viewer_gradient)

@app.route('/entry/<entry_id>/delete', methods=['POST'])
def delete_entry(entry_id):
    if 'user' not in session:
        return redirect(url_for('login_page'))
    validate_csrf()

    current_user = session['user']
    all_data = load_data(DIARY_FILE, dict)
    user_diaries = all_data.get(current_user, [])
    index = find_entry_index(user_diaries, entry_id)

    if index is None:
        return redirect(url_for('history'))

    user_diaries.pop(index)
    all_data[current_user] = user_diaries
    save_data(DIARY_FILE, all_data)
    return redirect(url_for('history'))

@app.route('/entry/<entry_id>/edit', methods=['GET', 'POST'])
def edit_entry(entry_id):
    if 'user' not in session:
        return redirect(url_for('login_page'))

    current_user = session['user']
    all_data = load_data(DIARY_FILE, dict)
    user_diaries = all_data.get(current_user, [])
    index = find_entry_index(user_diaries, entry_id)

    if index is None:
        return redirect(url_for('history'))

    entry = user_diaries[index]

    if request.method == 'POST':
        validate_csrf()
        raw_date = request.form.get('date')
        diary_text = request.form.get('diary')
        english_date = format_english_date(raw_date)

        result = improved_analyzer.analyze_emotion_and_color(diary_text)

        entry.update({
            'date': english_date,
            'text': diary_text,
            'emotion': result['emotion'],
            'color': result['color_hex'],
            'color_name': result['color_name'],
            'tone': result['tone'],
            'people': result['people']
        })

        user_diaries[index] = entry
        all_data[current_user] = user_diaries
        save_data(DIARY_FILE, all_data)
        return redirect(url_for('view_entry', entry_id=entry_id))

    iso_date = parse_english_date(entry.get('date', ''))
    return render_template('edit.html',
                           entry=entry,
                           entry_id=entry_id,
                           iso_date=iso_date,
                           csrf_token=get_csrf_token(),
                           user=current_user)

if __name__ == '__main__':
    app.run(debug=True, port=5001) # 포트 충돌 방지 5001
