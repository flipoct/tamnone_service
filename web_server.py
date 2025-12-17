# -*- coding: utf-8 -*-
import os
import json
import ast
import time
import tempfile
from gradio_client import Client
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# --- 설정 ---
SPACE_ID = "leewatson/kshs33_emotion_predict"
HF_TOKEN = os.environ.get("HF_TOKEN")
ALPHA = 0.1
Z = 1.0
STEPS = 1
# 대화 기록을 시스템 임시 폴더에 저장하여 자동 재시작 문제 해결
HISTORY_FILE = os.path.join(tempfile.gettempdir(), "tamnone_conversation_history.json")

# --- Flask 앱 및 CORS 설정 ---
app = Flask(__name__)
CORS(app) # 모든 출처에서의 요청을 허용합니다.

# --- 대화 기록 파일 관리 함수 ---
def read_history():
    return []

def write_history(history):
    pass

# --- 분석 로직 (기존 cli_emotion_analyzer.py에서 가져옴) ---
def parse_struct(s):
    if isinstance(s, dict): return s
    try: return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        try: return ast.literal_eval(s)
        except (ValueError, SyntaxError): return s

def call_space(client, text, alpha=0.1, z=1.8, steps=3):
    start_time = time.time()
    try:
        out = client.predict(text=text, alpha=float(alpha), z=float(z), steps=int(steps), api_name="/predict")
        print(out)
        latency = time.time() - start_time
        if out and len(out) > 2:
            return parse_struct(out[2]), latency
        else:
            return f"Error: API 응답 형식이 올바르지 않습니다. 응답: {out}", latency
    except Exception as e:
        latency = time.time() - start_time
        return f"Error: API 호출 중 오류 발생: {str(e)}", latency

# --- Gradio 클라이언트 초기화 ---
print("🚀 API 클라이언트 연결을 초기화합니다...")
try:
    client = Client(SPACE_ID, hf_token=HF_TOKEN) if HF_TOKEN else Client(SPACE_ID)
    # 테스트 호출
    test_result, _ = call_space(client, "테스트: 안녕하세요.", ALPHA, Z, STEPS)
    if isinstance(test_result, dict):
        print("✅ API 클라이언트 연결 및 테스트 성공.")
    else:
        print(f"❌ API 테스트 실패: {test_result}")
except Exception as e:
    print(f"❌ 클라이언트 초기화 실패: {e}")
    client = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ping', methods=['GET'])
def pingpong():
    return jsonify({"return": "200 OK"}), 200

# --- API 엔드포인트 ---
@app.route('/analyze', methods=['POST'])
def analyze_text():
    ALPHA = request.args.get('alpha')
    Z = request.args.get('z')
    STEPS = request.args.get('steps')
    if not client:
        return jsonify({"error": "API 클라이언트가 초기화되지 않았습니다."}), 500

    data = request.get_json()
    speaker = data.get('speaker', '화자').strip()
    user_input = data.get('text', '').strip()

    if not speaker or not user_input:
        return jsonify({"error": "화자 이름과 텍스트를 모두 입력해야 합니다."}), 400

    dialogue_history = read_history()
    
    # 대화 기록에 현재 발화 추가
    new_utterance = f"{speaker}: {user_input}"
    dialogue_history.append(new_utterance)

    # 전체 대화 기록을 모델 입력으로 사용
    full_dialogue_text = '\n'.join(dialogue_history)
    
    print(f"⏳ {len(dialogue_history)}개 발화 분석 요청 중...\n{full_dialogue_text}")

    # 분석 실행
    analysis_result, latency = call_space(client, full_dialogue_text, alpha=ALPHA, z=Z, steps=STEPS)

    if isinstance(analysis_result, str) and analysis_result.startswith("Error:"):
        # 오류 발생 시 파일에 저장하지 않음 (기록 추가를 되돌림)
        return jsonify({"error": analysis_result, "latency": latency})
    
    # 성공 시에만 파일에 기록
    write_history(dialogue_history)
    
    # 성공 시 결과 반환
    return jsonify({"result": analysis_result, "latency": latency})

@app.route('/analyze_snapshot', methods=['POST'])
def analyze_snapshot():
    ALPHA = request.args.get('alpha')
    Z = request.args.get('z')
    STEPS = request.args.get('steps')
    if not client:
        return jsonify({"error": "API 클라이언트가 초기화되지 않았습니다."}), 500

    data = request.get_json() or {}

    # WhisperLiveKit: { lines: [ {speaker, text, start, end, ...}, ... ] }
    lines = data.get('lines', [])
    if not isinstance(lines, list):
        return jsonify({"error": "lines must be a list"}), 400

    # 1) lines -> "대화 기록(list of strings)"로 정규화 (빈 text 제거)
    dialogue_history = []
    for l in lines:
        if not isinstance(l, dict):
            continue
        t = (l.get('text') or '').strip()
        if not t:
            continue
        sp = l.get('speaker', 'Speaker')
        dialogue_history.append(f"{sp}: {t}")

    # 스냅샷이 비어있으면 분석 안 함 (원하면 여기서 history를 비울 수도 있음)
    if not dialogue_history:
        return jsonify({"error": "스냅샷에 유효한 발화가 없습니다."}), 400

    # 2) 서버 history를 "교체" 저장 (append 금지)
    write_history(dialogue_history)

    # 3) 전체 대화 텍스트로 분석
    full_dialogue_text = "\n".join(dialogue_history)
    analysis_result, latency = call_space(client, full_dialogue_text, alpha=ALPHA, z=Z, steps=STEPS)

    if isinstance(analysis_result, str) and analysis_result.startswith("Error:"):
        return jsonify({"error": analysis_result, "latency": latency}), 500

    return jsonify({
        "result": analysis_result,
        "latency": latency,
        "history": dialogue_history
    })


@app.route('/reset', methods=['POST'])
def reset_history():
    write_history([]) # 파일을 비움
    print("🗑️ 대화 기록이 초기화되었습니다.")
    return jsonify({"message": "대화 기록이 초기화되었습니다."})

@app.route('/history', methods=['GET'])
def get_history():
    dialogue_history = read_history()
    return jsonify({"history": dialogue_history})

# --- 서버 실행 ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
