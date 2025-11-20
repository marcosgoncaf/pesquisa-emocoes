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

st.set_page_config(page_title="Estudo", layout="wide", initial_sidebar_state="collapsed")

# =================================================================================
# --- BACKEND & LÓGICA ---
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
        
        # --- MODO FACE CHECK (Com visual) ---
        if not self.recording:
            gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(gray, 1.1, 4)
            is_face = len(faces) > 0
            
            if is_face != self.face_detected:
                self.result_queue.put(is_face)
                self.face_detected = is_face
            
            h, w, _ = frm.shape
            # Desenha Oval (Verde se OK, Cinza se não)
            color = (0, 200, 0) if is_face else (200, 200, 200)
            thickness = 3 if is_face else 2
            
            # Guia Visual
            cv2.ellipse(frm, (w // 2, h // 2), (90, 120), 0, 0, 360, color, thickness)
            
            # Texto de instrução na câmera
            text = "ROSTO OK" if is_face else "POSICIONE O ROSTO"
            cv2.putText(frm, text, (w//2 - 60, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
            return VideoFrame.from_ndarray(frm, format="bgr24")
        
        # --- MODO GRAVAÇÃO (Silencioso/Buffer) ---
        with self.capture_lock:
            if self.recording:
                self.frames_buffer.append(frm)
        # Retorna frame preto para economizar banda durante o teste
        return VideoFrame.from_ndarray(np.zeros((1, 1, 3), np.uint8), format="bgr24")

# =================================================================================
# --- INTERFACE PRINCIPAL ---
# =================================================================================
params = st.query_params
study_id_from_url = params.get("study_id")

# --- CSS GERAL (APP-LIKE) ---
st.markdown("""
<style>
    /* Remove elementos padrão */
    #MainMenu, footer, header, [data-testid="stSidebar"] {display: none !important;}
    
    /* Fundo e Fontes */
    .stApp { background-color: #F0F2F6; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
    
    /* Cards Estilizados */
    .app-card {
        background: white;
        padding: 2rem;
        border-radius: 24px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    /* Imagem do Estudo */
    .study-img {
        width: 100%;
        max-width: 400px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Imagem Circular (Estímulo) */
    .circle-stimulus {
        width: 280px;
        height: 280px;
        border-radius: 50%;
        object-fit: cover;
        border: 4px solid white;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        margin: 0 auto;
        display: block;
    }

    /* Botão Personalizado */
    .stButton>button {
        background: linear-gradient(45deg, #4F46E5, #3B82F6);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 16px 32px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4);
        transition: transform 0.1s;
    }
    .stButton>button:active { transform: scale(0.98); }
    .stButton>button:disabled { background: #cbd5e1; color: #64748b; box-shadow: none; }

    /* Esconde controles WebRTC */
    div[class*="stWebcameraselector"] {display: none;}
</style>
""", unsafe_allow_html=True)

# Variáveis de Estado
if 'participant_results' not in st.session_state: st.session_state.participant_results = []
if 'p_stage' not in st.session_state: st.session_state.p_stage = 'check_in'
if 'current_item' not in st.session_state: st.session_state.current_item = 0
if 'face_ok' not in st.session_state: st.session_state.face_ok = False

if study_id_from_url:
    # #############################################################################
    # MODO PARTICIPANTE
    # #############################################################################
    
    # Carrega Configuração
    if 'study_config' not in st.session_state:
        sh = connect_gsheets()
        if sh:
            try:
                ws = sh.worksheet("Estudos")
                cell = ws.find(study_id_from_url)
                if cell: st.session_state.study_config = json.loads(ws.cell(cell.row, 2).value)
            except: pass
    
    config = st.session_state.get('study_config')
    if not config:
        st.error("Estudo não encontrado.")
        st.stop()

    # CSS Dinâmico para Câmera (Esconde durante o teste para não distrair)
    if st.session_state.p_stage == 'check_in':
        st.markdown("""<style>
            div[data-testid="stWebRTC"] {
                margin: 0 auto; 
                width: 300px; 
                border-radius: 20px; 
                overflow: hidden; 
                border: 2px solid #e5e7eb;
            }
        </style>""", unsafe_allow_html=True)
    else:
        # Câmera oculta mas ativa (height 0)
        st.markdown("""<style>div[data-testid="stWebRTC"] {height: 0px; visibility: hidden; margin: 0;}</style>""", unsafe_allow_html=True)

    # Componente WebRTC (Sempre ativo no fundo)
    webrtc_ctx = webrtc_streamer(
        key="stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    # Loop de verificação do rosto (Face Check)
    if webrtc_ctx.video_processor:
        try:
            while True:
                st.session_state.face_ok = webrtc_ctx.video_processor.result_queue.get_nowait()
        except queue.Empty: pass

    # --- CONTAINER PRINCIPAL ---
    col_spacer1, col_main, col_spacer2 = st.columns([1, 2, 1])
    
    with col_main:
        
        # 1. CHECK-IN E FACE ID
        if st.session_state.p_stage == 'check_in':
            st.markdown(f"""
            <div class='app-card'>
                <h1>Olá! 👋</h1>
                <p style='color: #666; margin-bottom: 20px;'>{config.get('welcome_message')}</p>
                <p style='font-weight: bold; color: #333;'>Posicione seu rosto no oval abaixo para liberar o acesso.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # A câmera aparece aqui automaticamente pelo Streamlit
            
            if st.session_state.face_ok:
                st.success("Face identificada! Pode começar.")
            else:
                st.info("Aguardando posicionamento...")

            pid = st.text_input("Digite seu Nome ou ID:", placeholder="Ex: Ana Silva")
            
            if st.button("COMEÇAR AVALIAÇÃO", disabled=not st.session_state.face_ok):
                if pid:
                    st.session_state.participant_id = pid
                    st.session_state.p_stage = 'instruction'
                    st.rerun()
                else:
                    st.toast("⚠️ Por favor, digite seu nome.")

        # 2. ESTÍMULO E TIMER
        elif st.session_state.p_stage == 'instruction':
            idx = st.session_state.current_item
            if idx >= len(config['items']):
                st.session_state.p_stage = 'end'
                st.rerun()
            
            item = config['items'][idx]
            duration = config.get('exposure_time', 5)
            
            # Inicializa Timer
            if 'start_time' not in st.session_state:
                st.session_state.start_time = time.time()
                # Inicia gravação silenciosa
                if webrtc_ctx.video_processor:
                     webrtc_ctx.video_processor.frames_buffer = []
                     webrtc_ctx.video_processor.recording = True
            
            # Layout Visual
            st.markdown(f"<h3 style='text-align: center; color: #444;'>Observe por {duration} segundos...</h3>", unsafe_allow_html=True)
            
            # Imagem Circular
            try:
                st.markdown(f"""
                    <div style="margin: 30px 0;">
                        <img src="{item['stimulus_url']}" class="circle-stimulus">
                    </div>
                    <p style="text-align: center; font-size: 1.1rem; color: #555; font-weight: 500;">{item['caption']}</p>
                """, unsafe_allow_html=True)
            except: st.error("Erro ao carregar imagem")

            # Barra de Progresso Circular (Simulada com barra normal estilizada)
            elapsed = time.time() - st.session_state.start_time
            if elapsed < duration:
                st.progress(min(elapsed / duration, 1.0))
                time.sleep(0.1)
                st.rerun()
            else:
                # Fim do tempo: Para gravação e salva frames
                if webrtc_ctx.video_processor:
                    webrtc_ctx.video_processor.recording = False
                    with webrtc_ctx.video_processor.capture_lock:
                        frames = webrtc_ctx.video_processor.frames_buffer.copy()
                    
                    # Amostragem inteligente (FPS)
                    fps = config.get('fps_limit', 3)
                    total_frames = len(frames)
                    target_frames = int(duration * fps)
                    
                    if total_frames > 0:
                        step = max(1, int(total_frames / target_frames))
                        st.session_state.last_frames = frames[::step]
                    else:
                        st.session_state.last_frames = []
                        
                del st.session_state['start_time']
                st.session_state.p_stage = 'questions'
                st.rerun()

        # 3. QUESTIONÁRIO
        elif st.session_state.p_stage == 'questions':
            item = config['items'][st.session_state.current_item]
            
            st.markdown("<div class='app-card'>", unsafe_allow_html=True)
            st.markdown("<h3>📝 Sua Avaliação</h3>", unsafe_allow_html=True)
            
            with st.form("qs"):
                ans_liking, ans_emotions, ans_word = None, [], ""
                
                if 'Nota de Gostar (1-9)' in item['questions']:
                    st.write("**Quanto você gostou?**")
                    ans_liking = st.slider("", 1, 9, 5)
                
                if 'Lista de Emoções (Múltipla Escolha)' in item['questions']:
                    st.write("**Quais emoções você sentiu?**")
                    ans_emotions = st.multiselect("", ['Alegre', 'Calmo', 'Interessado', 'Nojo', 'Medo', 'Triste', 'Surpreso', 'Neutro', 'Apetite'])
                
                if 'Uma Palavra que Define (Campo de Texto)' in item['questions']:
                    st.write("**Defina em uma palavra:**")
                    ans_word = st.text_input("")
                
                st.write("")
                if st.form_submit_button("PRÓXIMO"):
                    st.session_state.participant_results.append({
                        'frames': st.session_state.get('last_frames', []),
                        'ans_liking': ans_liking, 'ans_emotions': ans_emotions, 'ans_word': ans_word,
                        'stimulus': item['name'], 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    st.session_state.current_item += 1
                    st.session_state.p_stage = 'instruction'
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        # 4. TELA FINAL
        elif st.session_state.p_stage == 'end':
            st.markdown("<div class='app-card'><h2>🎉 Obrigado!</h2><p>Estamos salvando suas respostas...</p></div>", unsafe_allow_html=True)
            
            if not st.session_state.get('saved', False):
                bar = st.progress(0)
                rows = []
                tot = len(st.session_state.participant_results)
                
                for i, res in enumerate(st.session_state.participant_results):
                    # Processamento de Emoção
                    frames_to_anal = []
                    if res['frames']:
                        # Pega inicio, meio e fim para análise rápida
                        frames_to_anal.append(res['frames'][0])
                        if len(res['frames']) > 2: frames_to_anal.append(res['frames'][len(res['frames'])//2])
                        if len(res['frames']) > 1: frames_to_anal.append(res['frames'][-1])
                    
                    ems = []
                    for f in frames_to_anal:
                        try: _, b = cv2.imencode('.jpg', f); ems.append(analyze_emotion(b.tobytes()))
                        except: ems.append("erro")
                    
                    while len(ems) < 3: ems.append("N/A")
                    
                    rows.append([
                        st.session_state.participant_id, study_id_from_url, res['stimulus'], res['timestamp'],
                        ems[0], ems[1], ems[2], 
                        res['ans_liking'] or "", ", ".join(res['ans_emotions']), res['ans_word']
                    ])
                    bar.progress((i+1)/tot)
                
                sh = connect_gsheets()
                if sh:
                    try:
                        sh.worksheet("Resultados").append_rows(rows)
                        st.session_state.saved = True
                        st.success("✅ Dados salvos com sucesso!")
                        st.balloons()
                    except: st.error("Erro de conexão.")
            else:
                st.info("Você já pode fechar esta página.")

else:
    # #############################################################################
    # MODO ADMIN (PESQUISADOR)
    # #############################################################################
    # Restaura CSS do Admin
    st.markdown("<style>[data-testid='stSidebar'] {display: block !important;} .stApp {background-color: white;}</style>", unsafe_allow_html=True)

    st.title("🎛️ Painel do Pesquisador")
    
    if connect_gsheets(): st.success(f"Conectado: {NOME_DA_SUA_PLANILHA}")
    else: st.error("Erro de conexão com a planilha.")
    
    if 'study_items' not in st.session_state: st.session_state.study_items = []

    # Layout Admin
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("Configuração do Estudo")
        s_name = st.text_input("Nome do Estudo")
        w_msg = st.text_area("Mensagem de Boas-vindas")
        
        with st.expander("⚙️ Configurações Avançadas"):
            et = st.slider("Tempo de Exibição (segundos)", 3, 10, 5)
            fps = st.slider("Qualidade da Captura (FPS)", 1, 10, 3)

        st.subheader("Adicionar Estímulo")
        with st.form("add"):
            url = st.text_input("Link da Imagem (URL)")
            nm = st.text_input("Nome Interno (ID)")
            cp = st.text_input("Legenda para o Usuário")
            qs = st.multiselect("Perguntas", ['Nota de Gostar (1-9)', 'Lista de Emoções (Múltipla Escolha)', 'Uma Palavra que Define (Campo de Texto)'])
            if st.form_submit_button("Adicionar") and url and nm:
                st.session_state.study_items.append({"name": nm, "stimulus_url": url, "caption": cp, "questions": qs})
                st.success("Item Adicionado")

    with c2:
        st.subheader(f"Roteiro ({len(st.session_state.study_items)} itens)")
        for i, it in enumerate(st.session_state.study_items):
            st.info(f"{i+1}. {it['name']}")
        
        if st.button("Limpar Lista"):
            st.session_state.study_items = []
            st.rerun()

        if st.button("💾 SALVAR E GERAR LINK", type="primary"):
            if s_name and st.session_state.study_items:
                sid = generate_study_id()
                cfg = {"study_name": s_name, "welcome_message": w_msg, "items": st.session_state.study_items, "exposure_time": et, "fps_limit": fps}
                sh = connect_gsheets()
                try:
                    sh.worksheet("Estudos").append_row([sid, json.dumps(cfg)])
                    st.success("Link Criado:")
                    st.code(f"{URL_BASE_DA_SUA_APP}?study_id={sid}")
                except: st.error("Erro ao salvar.")
            else:
                st.warning("Preencha os dados obrigatórios.")
