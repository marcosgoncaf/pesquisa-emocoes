import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
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
# --- CREDENCIAIS (JÁ INCLUSAS) ---
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
NOME_DA_SUA_PLANILHA = "Resultados Pesquisa Emoções"
URL_BASE_DA_SUA_APP = "https://pesquisa-emocoes-jjhae3nwqqs4mslggexsmn.streamlit.app"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

st.set_page_config(page_title="Estudo Sensorial", layout="wide", initial_sidebar_state="collapsed")

# --- CSS ULTRA CLEAN (Sem rolagem, sem botões feios) ---
st.markdown("""
<style>
    /* Esconde elementos padrão do Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;} /* Esconde sidebar para participante */
    
    /* Esconde o seletor de dispositivo do WebRTC */
    div[class*="stWebcameraselector"] {display: none;}
    div[class*="stSelectbox"] {display: none;} 
    
    /* Layout Centralizado */
    .main-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 90vh;
        text-align: center;
    }
    
    /* Cards Limpos */
    .app-card {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        max-width: 600px;
        margin: auto;
    }
    
    /* Imagem Circular */
    .circle-img {
        width: 300px;
        height: 300px;
        border-radius: 50%;
        object-fit: cover;
        margin: 0 auto 20px auto;
        display: block;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Botão Primário */
    .stButton>button {
        background-color: #2D2D2D;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #404040;
        color: white;
    }

    /* Câmera Centralizada */
    div[data-testid="stWebRTC"] {
        margin: 0 auto;
        border-radius: 15px;
        overflow: hidden;
    }
    video { transform: scaleX(-1); }
</style>
""", unsafe_allow_html=True)

# =================================================================================
# --- BACKEND ---
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
        
        # Modo Validação (Desenha o Oval)
        if not self.recording:
            gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(gray, 1.1, 4)
            is_face = len(faces) > 0
            
            if is_face != self.face_detected:
                self.result_queue.put(is_face)
                self.face_detected = is_face
            
            h, w, _ = frm.shape
            # Desenha Oval Branco Suave
            cv2.ellipse(frm, (w // 2, h // 2), (100, 130), 0, 0, 360, (255, 255, 255), 2)
            
            # Se detectar, contorno fica verde sutil
            if is_face:
                cv2.ellipse(frm, (w // 2, h // 2), (100, 130), 0, 0, 360, (0, 255, 0), 3)
                
            return VideoFrame.from_ndarray(frm, format="bgr24")
        
        # Modo Gravação (Invisível/Silencioso)
        with self.capture_lock:
            if self.recording:
                self.frames_buffer.append(frm)
        return VideoFrame.from_ndarray(np.zeros((1, 1, 3), np.uint8), format="bgr24")

# =================================================================================
# --- LÓGICA PRINCIPAL ---
# =================================================================================
params = st.query_params
study_id_from_url = params.get("study_id")

if study_id_from_url:
    # --- MODO PARTICIPANTE ---
    
    # Carregamento
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
    if not config: st.error("Estudo não encontrado."); st.stop()

    # Estados
    if 'p_stage' not in st.session_state: st.session_state.p_stage = 'check_in'
    if 'current_item' not in st.session_state: st.session_state.current_item = 0
    if 'face_ok' not in st.session_state: st.session_state.face_ok = False
    if 'participant_results' not in st.session_state: st.session_state.participant_results = []

    # CSS Dinâmico da Câmera
    cam_visible = """
    <style>
    div[data-testid="stWebRTC"] {
        width: 320px; 
        height: 240px; 
        margin: 0 auto 20px auto;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    </style>
    """
    cam_hidden = """
    <style>div[data-testid="stWebRTC"] {display: none; visibility: hidden; height: 0px;}</style>
    """
    
    # WebRTC Persistente
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    # Renderiza a câmera (Visível no check-in, oculta depois)
    st.markdown(cam_visible if st.session_state.p_stage == 'check_in' else cam_hidden, unsafe_allow_html=True)
    
    webrtc_ctx = webrtc_streamer(
        key="stream", 
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    # Atualiza status do rosto
    if webrtc_ctx.video_processor:
        try:
            while True:
                st.session_state.face_ok = webrtc_ctx.video_processor.result_queue.get_nowait()
        except queue.Empty: pass

    # Container Principal (Centralizado)
    main = st.container()

    with main:
        # TELA 1: CHECK-IN
        if st.session_state.p_stage == 'check_in':
            st.markdown(f"<div class='app-card'><h2>Bem-vindo</h2><p>{config.get('welcome_message')}</p></div>", unsafe_allow_html=True)
            
            # A câmera já está sendo renderizada acima pelo st.markdown
            
            # Feedback de Rosto
            if st.session_state.face_ok:
                st.success("Rosto identificado")
            else:
                st.info("Posicione seu rosto no centro")

            pid = st.text_input("Identificação (Nome/ID):")
            
            if st.button("Iniciar Estudo", disabled=not st.session_state.face_ok):
                if pid:
                    st.session_state.participant_id = pid
                    st.session_state.p_stage = 'instruction'
                    st.rerun()
                else: st.warning("Preencha sua identificação.")

        # TELA 2: INSTRUÇÃO + ESTÍMULO (LOOP)
        elif st.session_state.p_stage == 'instruction':
            idx = st.session_state.current_item
            if idx >= len(config['items']):
                st.session_state.p_stage = 'end'
                st.rerun()
            
            item = config['items'][idx]
            duration = config.get('exposure_time', 5)
            
            # Inicia timer e gravação na primeira passagem
            if 'start_time' not in st.session_state:
                st.session_state.start_time = time.time()
                if webrtc_ctx.video_processor:
                     webrtc_ctx.video_processor.frames_buffer = []
                     webrtc_ctx.video_processor.recording = True
            
            # Exibe imagem e legenda
            st.markdown(f"""
                <div class='app-card'>
                    <img src="{item['stimulus_url']}" class="circle-img">
                    <p style="font-size: 1.2rem; font-weight: 500;">{item['caption']}</p>
                </div>
            """, unsafe_allow_html=True)

            # Barra de tempo
            elapsed = time.time() - st.session_state.start_time
            progress = min(elapsed / duration, 1.0)
            st.progress(progress)
            
            # Lógica de fim de tempo
            if elapsed >= duration:
                if webrtc_ctx.video_processor:
                    webrtc_ctx.video_processor.recording = False
                    with webrtc_ctx.video_processor.capture_lock:
                        frames = webrtc_ctx.video_processor.frames_buffer.copy()
                    
                    # Amostragem de frames
                    fps = config.get('fps_limit', 3)
                    sel = []
                    if frames:
                        step = max(1, int(len(frames) / (duration * fps)))
                        sel = frames[::step][:int(duration*10)]
                    st.session_state.last_frames = sel
                
                del st.session_state['start_time']
                st.session_state.p_stage = 'questions'
                st.rerun()
            
            time.sleep(0.1)
            st.rerun()

        # TELA 3: PERGUNTAS (CENTRALIZADAS)
        elif st.session_state.p_stage == 'questions':
            item = config['items'][st.session_state.current_item]
            
            st.markdown("<div class='app-card'><h3>Avaliação</h3></div>", unsafe_allow_html=True)
            
            with st.form("qs"):
                lk, em, wd = None, [], ""
                if 'Nota de Gostar (1-9)' in item['questions']: 
                    st.write("O quanto você gostou?")
                    lk = st.slider("", 1, 9, 5)
                
                if 'Lista de Emoções (Múltipla Escolha)' in item['questions']:
                    st.write("O que sentiu?")
                    em = st.multiselect("", ['Alegre', 'Calmo', 'Interessado', 'Nojo', 'Medo', 'Triste', 'Surpreso', 'Neutro'])
                
                if 'Uma Palavra que Define (Campo de Texto)' in item['questions']:
                    st.write("Defina em uma palavra:")
                    wd = st.text_input("")
                
                if st.form_submit_button("Próximo"):
                    st.session_state.participant_results.append({
                        'frames': st.session_state.get('last_frames', []),
                        'ans_liking': lk, 'ans_emotions': em, 'ans_word': wd,
                        'stimulus': item['name'], 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    st.session_state.current_item += 1
                    st.session_state.p_stage = 'instruction'
                    st.rerun()

        # TELA 4: FINAL
        elif st.session_state.p_stage == 'end':
            st.markdown("<div class='app-card'><h2>Obrigado!</h2><p>Salvando respostas...</p></div>", unsafe_allow_html=True)
            
            if not st.session_state.get('saved', False):
                bar = st.progress(0)
                rows = []
                tot = len(st.session_state.participant_results)
                
                for i, res in enumerate(st.session_state.participant_results):
                    # Analisa frames
                    cap = res['frames']
                    # Pega 3 frames representativos
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
                        st.success("Salvo com sucesso!")
                    except: st.error("Erro ao salvar.")
            else:
                st.info("Você pode fechar esta janela.")

else:
    # --- MODO ADMIN ---
    # (Restaura a sidebar para o admin)
    st.markdown("<style>[data-testid='stSidebar'] {display: block;}</style>", unsafe_allow_html=True)
    
    st.title("Painel do Pesquisador")
    sh = connect_gsheets()
    if sh: st.success(f"✅ Conectado: {NOME_DA_SUA_PLANILHA}")
    
    if 'study_items' not in st.session_state: st.session_state.study_items = []

    with st.sidebar:
        st.header("Novo Estudo")
        s_name = st.text_input("Nome do Estudo")
        w_msg = st.text_area("Boas-vindas")
        
        with st.expander("⚙️ Configurações Técnicas"):
            et = st.slider("Tempo de Exposição (s)", 3, 10, 5)
            fps = st.slider("Taxa de Captura (FPS)", 1, 10, 3)

        st.subheader("Adicionar Tarefa")
        with st.form("add"):
            url = st.text_input("URL Imagem")
            nm = st.text_input("Nome ID")
            cp = st.text_input("Legenda")
            qs = st.multiselect("Perguntas", ['Nota de Gostar (1-9)', 'Lista de Emoções (Múltipla Escolha)', 'Uma Palavra que Define (Campo de Texto)'])
            if st.form_submit_button("Adicionar") and url and nm:
                st.session_state.study_items.append({"name": nm, "stimulus_url": url, "caption": cp, "questions": qs})
                st.success("Adicionado")

    if st.session_state.study_items:
        st.markdown(f"### Roteiro ({len(st.session_state.study_items)} itens)")
        for it in st.session_state.study_items: st.info(f"{it['name']} ({it['caption']})")
        
        c1, c2 = st.columns(2)
        with c1: 
            if st.button("Limpar"): st.session_state.study_items = []; st.rerun()
        with c2:
            if st.button("💾 GERAR LINK", type="primary"):
                sid = generate_study_id()
                cfg = {"study_name": s_name, "welcome_message": w_msg, "items": st.session_state.study_items, "exposure_time": et, "fps_limit": fps}
                sh.worksheet("Estudos").append_row([sid, json.dumps(cfg)])
                st.success("Salvo!")
                st.code(f"{URL_BASE_DA_SUA_APP}?study_id={sid}")
