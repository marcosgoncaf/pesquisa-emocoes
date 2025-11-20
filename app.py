import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from av import VideoFrame
import numpy as np
import cv2
from deepface import DeepFace
import pandas as pd
import gspread
from datetime import datetime
import time
import threading
import json
import random
import string
import os
import queue

# =================================================================================
# --- ÁREA DE CREDENCIAIS (CONFIGURADA) ---
# =================================================================================

CREDENCIAIS_JSON = {
  "type": "service_account",
  "project_id": "minha-ferramenta-de-pesquisa",
  "private_key_id": "1bb14eca16499ef1d49076a2109ec5505afdb14d",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCuybQHPpl6BEUE\nbU0N3ZIAF+K7MQju/vmEvkJo2mrM0qnKirMhqdivqf34Q0VTVq6LjfjYEZYuiF+a\ncLsdbbOo9YOn6Oxv1elcJQa0fH7s7BjEYa6TrufmO4w6ZUdU9mB93Gw3oKGl9eAT\nzkM3c31rvDdw07X+5lyLYmlL3J5ufmLN0/4gOUTZ+CdfFORntnN6LQJciIVggGrn\ne56QBfWRLYRCHOA2W1MbiBo5fHb8ClamGQn7ffHi+4kWj/Dh2hzgSK1KjD9n3xou\nTuHTaNo295T3lPnLJWsLBjdvusikxZ+SJjSJOf93MEeeiKeLUDxB6TyRNAN+NyEM\nuVAVCPqLAgMBAAECggEAUFWZfmjprniO5il8bF6Fj7Bqv8GaSwp60WWG7V1SEADV\nFCfqjBLd6mMVvQjiMJsbDz45/Mkcfxej5T5h6fU1SXdKQx7dqMsm/fhwm/zKc4dS\nHNSOogcWQ63j1iUdg3HXyJFKVcjo8vfOvRukrjRRFXbE/oD32Ye0SkX5gmoMkViW\nQeJ9sGN/7CddcfP8RGDS9YIkEAMjOejmX3vCwQ2pYbnzbT5HNho2DLNvt/XDysUV\nsrJ9DYha5m9zcc/dAyRBSFBGOCziLkUujSoznRA0nQ6anPMS1pCL6veWvjzXx75v\nBsGv2/lDXxkO7+ozBRZYvU/Nw8Q72civ18UTTCOr4QKBgQDyOCyPL5WYvUA6fIOb\nNZtV6itMfTdnMHSk1COaYjd8k6xIAbkBLzp0b276x3B2PCIOO+FEZ9bLkeoFg33k\n0TvrTUX1Vu4NzTGOFRmL4vHfUAMMOlc4kczRJbOzxf5hm2AQVbTJr9LGByk0pGCM\nupDLRSbuopxqw4aAUSIE9LNrKwKBgQC4u2q2Vg0PeL8aERmDW/QrrBG53ZKo1ZQH\nt8itB0XvEzmn/ZdXTAaGsTgGi2hkeKl55tTigtnywFVMTcfObkYQl/k3bEn0Kg0u\ny2GzsEYl6J+XHnIqRg2OT8itWgc1syTIYqtZZ2KQZCNkZ83hfw6x9/t0Z+5gxJyz\nej1CePt+IQKBgQDQcjUul+Wmhx8R9fuQA2cFXbsbXRrjq5iGDKYDAwj7JV56rTjX\n4xvr/cXE/QM0TPWFyFRI+Q+pKo35Zrpdww5MpbhRtRiyOeaufkv4zToXUpT83ewX\nm6lmR4rJZ9dNilf2Vrt5Yd0CXEFCsz7/fMKEm6MwdDIl0tQZ8zhSiX8p9wKBgQCn\nPp5FB7D5UeMJsXN8tpJfu3+s20n86qgDOMNIy23oHWq0iWUr1puN9AH6Atp3I7qj\nr3UKwVaabMCMvVporPNn8H7jS3nEwRGATQdeS/emOG3Lvfe0CAobzmWrc/dVCaQC\nezoQTgYxHUfhjg0Z2xMt0onoAHQTTc1kvAWcGbXuoQKBgAPhFs9BKb8IBYS00b7I\n5iqH9tx0BOps7ypfZcbOeWuZGvMWrGFVG1iZH7ymdB8AMLy5lG8rNjOxlJUCafkg\n46m+gmbihHGI5VuNA91QUiO6029RBHEvUVNXFvDw0AlR4TUgQlM7X8iDYVB28X9L\n90p0K3WiqPN1Qkf6DdZ0o8Xx\n-----END PRIVATE KEY-----\n".replace("\\n", "\n"),
  "client_email": "bot-da-planilha@minha-ferramenta-de-pesquisa.iam.gserviceaccount.com",
  "client_id": "100467755227624269135",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/bot-da-planilha%40minha-ferramenta-de-pesquisa.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}

# Configurações do App
NOME_DA_SUA_PLANILHA = "Resultados Pesquisa Emoções"
URL_BASE_DA_SUA_APP = "https://pesquisa-emocoes-jjhae3nwqqs4mslggexsmn.streamlit.app"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

st.set_page_config(page_title="Estudo Sensorial", layout="centered", page_icon="🧠")

# Estilos CSS (App-like)
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; font-family: 'Helvetica Neue', sans-serif; }
    .app-card { background-color: white; padding: 2rem; border-radius: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 20px; text-align: center; }
    .circle-img { border-radius: 50%; object-fit: cover; margin: 0 auto; display: block; box-shadow: 0 8px 16px rgba(0,0,0,0.2); }
    .stButton>button { width: 100%; border-radius: 12px; height: 3em; font-weight: bold; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =================================================================================
# --- FUNÇÕES DE BACKEND ---
# =================================================================================
def generate_study_id(length=8):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

@st.cache_resource
def connect_gsheets():
    try:
        sa = gspread.service_account_from_dict(CREDENCIAIS_JSON)
        sh = sa.open(NOME_DA_SUA_PLANILHA)
        return sh
    except Exception: return None

def analyze_emotion(frame_bytes):
    try:
        nparr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        analysis = DeepFace.analyze(img_path=img, actions=['emotion'], enforce_detection=False)
        if isinstance(analysis, list) and len(analysis) > 0:
            return analysis[0]['dominant_emotion']
        return "não_detectado"
    except: return "erro"

@st.cache_resource
def load_face_cascade():
    if not os.path.exists('haarcascade_frontalface_default.xml'):
        import urllib.request
        urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml", 'haarcascade_frontalface_default.xml')
    return cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class VideoProcessor:
    def __init__(self):
        self.frames_buffer = []
        self.capture_lock = threading.Lock()
        self.recording = False
        self.face_detected = False
        self.cascade = load_face_cascade()
        self.result_queue = queue.Queue()

    def recv(self, frame: VideoFrame) -> VideoFrame:
        frm = frame.to_ndarray(format="bgr24")
        if not self.recording:
            gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(gray, 1.1, 4)
            is_face = len(faces) > 0
            color = (0, 255, 0) if is_face else (0, 0, 255)
            if is_face != self.face_detected:
                self.result_queue.put(is_face)
                self.face_detected = is_face
            h, w, _ = frm.shape
            cv2.ellipse(frm, (w // 2, h // 2), (100, 140), 0, 0, 360, color, 2)
            return VideoFrame.from_ndarray(frm, format="bgr24")
        
        with self.capture_lock:
            if self.recording:
                self.frames_buffer.append(frm)
        return VideoFrame.from_ndarray(np.zeros((1, 1, 3), np.uint8), format="bgr24")

# =================================================================================
# --- APP PRINCIPAL ---
# =================================================================================
params = st.query_params
study_id_from_url = params.get("study_id")

if 'participant_results' not in st.session_state: st.session_state.participant_results = []

if study_id_from_url:
    # --- MODO PARTICIPANTE ---
    if 'study_config' not in st.session_state:
        sh = connect_gsheets()
        config = None
        if sh:
            try:
                ws = sh.worksheet("Estudos")
                cell = ws.find(study_id_from_url)
                if cell: config = json.loads(ws.cell(cell.row, 2).value)
            except: pass
        st.session_state.study_config = config

    config = st.session_state.study_config
    if not config:
        st.error("Estudo não encontrado.")
        st.stop()

    if 'p_stage' not in st.session_state: st.session_state.p_stage = 'check_in'
    if 'current_item' not in st.session_state: st.session_state.current_item = 0
    if 'face_ok' not in st.session_state: st.session_state.face_ok = False

    # Câmera
    css_cam_visible = "<style>div[data-testid='stWebRTC'] {margin: 0 auto; width: 300px; border-radius: 20px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.2);} video { transform: scaleX(-1); }</style>"
    css_cam_hidden = "<style>div[data-testid='stWebRTC'] {display: none; visibility: hidden; height: 0px;}</style>"
    
    st.markdown(css_cam_visible if st.session_state.p_stage == 'check_in' else css_cam_hidden, unsafe_allow_html=True)
    
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    webrtc_ctx = webrtc_streamer(
        key="stream", video_processor_factory=VideoProcessor, rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False}, async_processing=True
    )

    if webrtc_ctx.video_processor:
        try:
            while True:
                st.session_state.face_ok = webrtc_ctx.video_processor.result_queue.get_nowait()
        except queue.Empty: pass

    if st.session_state.p_stage == 'check_in':
        st.markdown(f"<div class='app-card'><h2>👋 Olá!</h2><p>{config.get('welcome_message')}</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='app-card'><h4>📸 Validação Facial</h4><p>Posicione seu rosto no oval.</p></div>", unsafe_allow_html=True)
        
        start_disabled = not st.session_state.face_ok
        if not start_disabled: st.success("Rosto Detectado!")
        else: st.warning("Aguardando rosto...")

        pid = st.text_input("Seu ID/Nome:")
        if st.button("COMEÇAR 🚀", disabled=start_disabled):
            if pid:
                st.session_state.participant_id = pid
                st.session_state.p_stage = 'instruction'
                st.rerun()
            else: st.toast("Preencha seu ID!")

    elif st.session_state.p_stage == 'instruction':
        idx = st.session_state.current_item
        if idx >= len(config['items']):
            st.session_state.p_stage = 'end'
            st.rerun()
        
        item = config['items'][idx]
        duration = config.get('exposure_time', 5)
        
        st.markdown(f"<div class='app-card'><h3>👁️ Tarefa {idx+1}</h3><p>Observe atentamente.</p></div>", unsafe_allow_html=True)
        with st.container():
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                if 'start_time' not in st.session_state:
                    st.session_state.start_time = time.time()
                    if webrtc_ctx.video_processor:
                         webrtc_ctx.video_processor.frames_buffer = []
                         webrtc_ctx.video_processor.recording = True
                
                try:
                    st.markdown(f"""<div style="display: flex; justify-content: center; margin: 20px 0;"><img src="{item['stimulus_url']}" class="circle-img" style="width: 300px; height: 300px;"></div><p style="text-align: center;">{item['caption']}</p>""", unsafe_allow_html=True)
                except: st.error("Erro na imagem")

                elapsed = time.time() - st.session_state.start_time
                if elapsed >= duration:
                    if webrtc_ctx.video_processor:
                        webrtc_ctx.video_processor.recording = False
                        with webrtc_ctx.video_processor.capture_lock:
                            frames = webrtc_ctx.video_processor.frames_buffer.copy()
                        fps = config.get('fps_limit', 3)
                        sel = []
                        if len(frames) > 0:
                            step = max(1, int(len(frames) / (duration * fps)))
                            sel = frames[::step][:int(duration*10)]
                        st.session_state.last_frames = sel
                    del st.session_state['start_time']
                    st.session_state.p_stage = 'questions'
                    st.rerun()
                
                st.progress(min(elapsed / duration, 1.0))
                time.sleep(0.1)
                st.rerun()

    elif st.session_state.p_stage == 'questions':
        item = config['items'][st.session_state.current_item]
        st.markdown("<div class='app-card'><h3>📝 Sua Opinião</h3></div>", unsafe_allow_html=True)
        with st.form("qs"):
            lk, em, wd = None, [], ""
            if 'Nota de Gostar (1-9)' in item['questions']: st.write("**Gostou?**"); lk = st.slider("", 1, 9, 5)
            if 'Lista de Emoções (Múltipla Escolha)' in item['questions']: st.write("**Sentiu?**"); em = st.multiselect("", ['Alegre', 'Calmo', 'Nojo', 'Medo', 'Triste', 'Surpreso', 'Neutro', 'Apetite'])
            if 'Uma Palavra que Define (Campo de Texto)' in item['questions']: st.write("**Defina:**"); wd = st.text_input("")
            if st.form_submit_button("Enviar"):
                st.session_state.participant_results.append({
                    'frames': st.session_state.get('last_frames', []),
                    'ans_liking': lk, 'ans_emotions': em, 'ans_word': wd,
                    'stimulus': item['name'], 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.session_state.current_item += 1
                st.session_state.p_stage = 'instruction'
                st.rerun()

    elif st.session_state.p_stage == 'end':
        st.markdown("<div class='app-card'><h2>🎉 Fim!</h2><p>Salvando...</p></div>", unsafe_allow_html=True)
        if not st.session_state.get('saved', False):
            bar = st.progress(0)
            rows = []
            tot = len(st.session_state.participant_results)
            for i, res in enumerate(st.session_state.participant_results):
                cap = res['frames']
                sel_f = []
                if cap:
                    sel_f.append(cap[0])
                    if len(cap)>2: sel_f.append(cap[len(cap)//2])
                    if len(cap)>1: sel_f.append(cap[-1])
                
                ems = []
                for f in sel_f:
                    try: _, b = cv2.imencode('.jpg', f); ems.append(analyze_emotion(b.tobytes()))
                    except: ems.append("erro")
                while len(ems)<3: ems.append("N/A")
                
                rows.append([
                    st.session_state.participant_id, study_id_from_url, res['stimulus'], res['timestamp'],
                    ems[0], ems[1], ems[2], res['ans_liking'] or "", ", ".join(res['ans_emotions']), res['ans_word']
                ])
                bar.progress((i+1)/tot)
            
            sh = connect_gsheets()
            if sh:
                try:
                    sh.worksheet("Resultados").append_rows(rows)
                    st.session_state.saved = True
                    st.balloons()
                    st.success("Salvo!")
                except: st.error("Erro ao salvar.")
        else: st.success("Pode fechar.")

else:
    # --- MODO ADMIN ---
    st.markdown("<h1 style='text-align: center;'>🎛️ Painel Admin</h1>", unsafe_allow_html=True)
    sh = connect_gsheets()
    if not sh: st.error("Erro de conexão com a planilha.")
    else:
        if 'study_items' not in st.session_state: st.session_state.study_items = []
        
        with st.container():
            st.markdown("<div class='app-card'><h3>📁 Novo Estudo</h3></div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1: s_name = st.text_input("Nome")
            with c2: w_msg = st.text_area("Boas-vindas")
            with st.expander("⚙️ Avançado"):
                et = st.slider("Tempo (s)", 3, 10, 5)
                fps = st.slider("FPS", 1, 10, 3)

        st.markdown("<div class='app-card'><h3>➕ Itens</h3></div>", unsafe_allow_html=True)
        with st.form("add"):
            url = st.text_input("URL Imagem")
            nm = st.text_input("Nome ID")
            cp = st.text_input("Legenda")
            qs = st.multiselect("Perguntas", ['Nota de Gostar (1-9)', 'Lista de Emoções (Múltipla Escolha)', 'Uma Palavra que Define (Campo de Texto)'])
            if st.form_submit_button("Adicionar") and url and nm:
                st.session_state.study_items.append({"name": nm, "stimulus_url": url, "caption": cp, "questions": qs})
                st.success("Ok!")

        if st.session_state.study_items:
            st.markdown(f"### Roteiro ({len(st.session_state.study_items)})")
            for it in st.session_state.study_items: st.info(f"{it['name']}")
            
            if st.button("Limpar"): st.session_state.study_items = []; st.rerun()
            
            if st.button("💾 GERAR LINK", type="primary"):
                sid = generate_study_id()
                cfg = {"study_name": s_name, "welcome_message": w_msg, "items": st.session_state.study_items, "exposure_time": et, "fps_limit": fps}
                try:
                    sh.worksheet("Estudos").append_row([sid, json.dumps(cfg)])
                    st.success("Sucesso!")
                    st.code(f"{URL_BASE_DA_SUA_APP}?study_id={sid}")
                except Exception as e: st.error(f"Erro: {e}")
