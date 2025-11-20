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

# =================================================================================
# --- ÁREA DE CREDENCIAIS (CORRIGIDA) ---
# =================================================================================

CREDENCIAIS_JSON = {
  "type": "service_account",
  "project_id": "minha-ferramenta-de-pesquisa",
  "private_key_id": "1bb14eca16499ef1d49076a2109ec5505afdb14d",
  # A CORREÇÃO ESTÁ NESTA LINHA ABAIXO (.replace)
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCuybQHPpl6BEUE\nbU0N3ZIAF+K7MQju/vmEvkJo2mrM0qnKirMhqdivqf34Q0VTVq6LjfjYEZYuiF+a\ncLsdbbOo9YOn6Oxv1elcJQa0fH7s7BjEYa6TrufmO4w6ZUdU9mB93Gw3oKGl9eAT\nzkM3c31rvDdw07X+5lyLYmlL3J5ufmLN0/4gOUTZ+CdfFORntnN6LQJciIVggGrn\ne56QBfWRLYRCHOA2W1MbiBo5fHb8ClamGQn7ffHi+4kWj/Dh2hzgSK1KjD9n3xou\nTuHTaNo295T3lPnLJWsLBjdvusikxZ+SJjSJOf93MEeeiKeLUDxB6TyRNAN+NyEM\nuVAVCPqLAgMBAAECggEAUFWZfmjprniO5il8bF6Fj7Bqv8GaSwp60WWG7V1SEADV\nFCfqjBLd6mMVvQjiMJsbDz45/Mkcfxej5T5h6fU1SXdKQx7dqMsm/fhwm/zKc4dS\nHNSOogcWQ63j1iUdg3HXyJFKVcjo8vfOvRukrjRRFXbE/oD32Ye0SkX5gmoMkViW\nQeJ9sGN/7CddcfP8RGDS9YIkEAMjOejmX3vCwQ2pYbnzbT5HNho2DLNvt/XDysUV\nsrJ9DYha5m9zcc/dAyRBSFBGOCziLkUujSoznRA0nQ6anPMS1pCL6veWvjzXx75v\nBsGv2/lDXxkO7+ozBRZYvU/Nw8Q72civ18UTTCOr4QKBgQDyOCyPL5WYvUA6fIOb\nNZtV6itMfTdnMHSk1COaYjd8k6xIAbkBLzp0b276x3B2PCIOO+FEZ9bLkeoFg33k\n0TvrTUX1Vu4NzTGOFRmL4vHfUAMMOlc4kczRJbOzxf5hm2AQVbTJr9LGByk0pGCM\nupDLRSbuopxqw4aAUSIE9LNrKwKBgQC4u2q2Vg0PeL8aERmDW/QrrBG53ZKo1ZQH\nt8itB0XvEzmn/ZdXTAaGsTgGi2hkeKl55tTigtnywFVMTcfObkYQl/k3bEn0Kg0u\ny2GzsEYl6J+XHnIqRg2OT8itWgc1syTIYqtZZ2KQZCNkZ83hfw6x9/t0Z+5gxJyz\nej1CePt+IQKBgQDQcjUul+Wmhx8R9fuQA2cFXbsbXRrjq5iGDKYDAwj7JV56rTjX\n4xvr/cXE/QM0TPWFyFRI+Q+pKo35Zrpdww5MpbhRtRiyOeaufkv4zToXUpT83ewX\nm6lmR4rJZ9dNilf2Vrt5Yd0CXEFCsz7/fMKEm6MwdDIl0tQZ8zhSiX8p9wKBgQCn\nPp5FB7D5UeMJsXN8tpJfu3+s20n86qgDOMNIy23oHWq0iWUr1puN9AH6Atp3I7qj\nr3UKwVaabMCMvVporPNn8H7jS3nEwRGATQdeS/emOG3Lvfe0CAobzmWrc/dVCaQC\nezoQTgYxHUfhjg0Z2xMt0onoAHQTTc1kvAWcGbXuoQKBgAPhFs9BKb8IBYS00b7I\n5iqH9tx0BOps7ypfZcbOeWuZGvMWrGFVG1iZH7ymdB8AMLy5lG8rNjOxlJUCafkg\n46m+gmbihHGI5VuNA91QUiO6029RBHEvUVNXFvDw0AlR4TUgQlM7X8iDYVB28X9L\n90p0K3WiqPN1Qkf6DdZ0o8Xx\n-----END PRIVATE KEY-----\n".replace("\\n", "\n"),
  "client_email": "bot-da-planilha@minha-ferramenta-de-pesquisa.iam.gserviceaccount.com",
  "client_id": "100467755227624269135",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/bot-da-planilha%40minha-ferramenta-de-pesquisa.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}

# Nome da sua planilha no Google Sheets
NOME_DA_SUA_PLANILHA = "Resultados Pesquisa Emoções"

# Link final do seu aplicativo (Atualize após o deploy se mudar)
URL_BASE_DA_SUA_APP = "https://pesquisa-emocoes-jjhae3nwqqs4mslggexsmn.streamlit.app"

# =================================================================================
# --- CONFIGURAÇÕES GLOBAIS ---
# =================================================================================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # Otimização para evitar erros no Cloud
st.set_page_config(page_title="Plataforma de Pesquisa", layout="centered")

# =================================================================================
# --- FUNÇÕES DE APOIO ---
# =================================================================================
def generate_study_id(length=8):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

@st.cache_resource
def connect_gsheets():
    """Conecta usando as credenciais embutidas no código."""
    try:
        # Usa diretamente a variável CREDENCIAIS_JSON que você preencheu acima
        sa = gspread.service_account_from_dict(CREDENCIAIS_JSON)
        sh = sa.open(NOME_DA_SUA_PLANILHA)
        return sh
    except Exception as e:
        st.error(f"Erro ao conectar na planilha. Verifique se colou o JSON corretamente e se o nome da planilha está certo. Detalhes: {e}")
        st.stop()

def analyze_emotion(frame_bytes):
    """Analisa a imagem. Retorna a emoção ou 'Erro'."""
    try:
        nparr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        analysis = DeepFace.analyze(img_path=img, actions=['emotion'], enforce_detection=False)
        if isinstance(analysis, list) and len(analysis) > 0:
            return analysis[0]['dominant_emotion']
        return "rosto_nao_detectado"
    except Exception as e:
        return "erro_analise"

class VideoProcessor:
    def __init__(self):
        self.frames_buffer = []
        self.capture_lock = threading.Lock()
        self.recording = False 

    def recv(self, frame: VideoFrame) -> VideoFrame:
        frm = frame.to_ndarray(format="bgr24")
        with self.capture_lock:
            if self.recording:
                self.frames_buffer.append(frm)
        return VideoFrame.from_ndarray(np.zeros((1, 1, 3), np.uint8), format="bgr24")

# =================================================================================
# --- APP ---
# =================================================================================
params = st.query_params
study_id_from_url = params.get("study_id")

if study_id_from_url:
    # --- MODO PARTICIPANTE ---
    
    def load_study_config(_study_id):
        sh = connect_gsheets()
        if not sh: return None
        try:
            worksheet = sh.worksheet("Estudos")
            cell = worksheet.find(_study_id)
            if cell: return json.loads(worksheet.cell(cell.row, 2).value)
        except: pass
        return None

    study_config = load_study_config(study_id_from_url)

    if study_config:
        if 'participant_stage' not in st.session_state: st.session_state.participant_stage = 'welcome'
        if 'participant_results' not in st.session_state: st.session_state.participant_results = []
        if 'current_item' not in st.session_state: st.session_state.current_item = 0
        if 'last_captured_frames' not in st.session_state: st.session_state.last_captured_frames = []

        st.markdown("<style>div[data-testid='stWebRTC']{display: none;}</style>", unsafe_allow_html=True)
        
        rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        webrtc_ctx = webrtc_streamer(
            key="webcam", 
            video_processor_factory=VideoProcessor,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )

        # TELA 1: BOAS VINDAS
        if st.session_state.participant_stage == 'welcome':
            st.title("Bem-vindo(a)")
            st.write(study_config.get('welcome_message', ''))
            st.session_state.participant_id = st.text_input("Seu ID:")
            if st.button("Iniciar"):
                if st.session_state.participant_id:
                    st.session_state.participant_stage = 'test'
                    st.rerun()
                else: st.warning("Preencha seu ID.")

        # TELA 2: TESTE
        elif st.session_state.participant_stage == 'test':
            idx = st.session_state.current_item
            if idx >= len(study_config['items']):
                st.session_state.participant_stage = 'end'
                st.rerun()
            
            item = study_config['items'][idx]
            
            if webrtc_ctx.video_processor:
                with webrtc_ctx.video_processor.capture_lock:
                    webrtc_ctx.video_processor.frames_buffer = []

            st.header(f"Tarefa {idx + 1}")
            try: st.image(item['stimulus_url'], caption=item['caption'], use_column_width=True)
            except: st.image("https://via.placeholder.com/400", caption="Erro imagem")
            
            if st.button("OK, observei", key=f"btn_{idx}"):
                if not webrtc_ctx.state.playing or not webrtc_ctx.video_processor:
                    st.error("🚨 Erro na câmera. Recarregue a página.")
                    st.stop()

                webrtc_ctx.video_processor.recording = True
                with st.spinner("Registrando..."):
                    time.sleep(4)
                webrtc_ctx.video_processor.recording = False
                
                frames = []
                with webrtc_ctx.video_processor.capture_lock:
                    frames = webrtc_ctx.video_processor.frames_buffer.copy()
                
                if len(frames) == 0:
                    st.error("⚠️ Nenhuma imagem capturada. Tente novamente.")
                else:
                    selected = []
                    if len(frames) > 0: selected.append(frames[0])
                    if len(frames) > 2:
                        selected.append(frames[len(frames)//2])
                        selected.append(frames[-1])
                    elif len(frames) == 2: selected.append(frames[1])
                    
                    st.session_state.last_captured_frames = selected
                    st.success("Captura OK!")
                    time.sleep(0.5)
                    st.session_state.participant_stage = 'questionnaire'
                    st.rerun()

        # TELA 3: QUESTIONÁRIO
        elif st.session_state.participant_stage == 'questionnaire':
            idx = st.session_state.current_item
            item = study_config['items'][idx]
            st.header("Responda")
            ans_liking, ans_emotions, ans_word = None, [], ""
            
            if 'Nota de Gostar (1-9)' in item['questions']: ans_liking = st.slider("Quanto você gostou?", 1, 9, 5)
            if 'Lista de Emoções (Múltipla Escolha)' in item['questions']: ans_emotions = st.multiselect("O que sentiu?", ['Alegre', 'Calmo', 'Interessado', 'Nojo', 'Medo', 'Triste', 'Surpreso', 'Neutro'])
            if 'Uma Palavra que Define (Campo de Texto)' in item['questions']: ans_word = st.text_input("Uma palavra que define:")
            
            if st.button("Próximo"):
                st.session_state.participant_results.append({
                    'frames': st.session_state.last_captured_frames,
                    'ans_liking': ans_liking, 'ans_emotions': ans_emotions, 'ans_word': ans_word,
                    'stimulus': item['name'], 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.session_state.current_item += 1
                st.session_state.participant_stage = 'test'
                st.rerun()

        # TELA 4: FIM
        elif st.session_state.participant_stage == 'end':
            st.title("Finalizado")
            if not st.session_state.get('data_saved', False):
                with st.spinner("Salvando..."):
                    rows = []
                    for r in st.session_state.participant_results:
                        ems = []
                        for f in r['frames']:
                            try:
                                _, buffer = cv2.imencode('.jpg', f)
                                ems.append(analyze_emotion(buffer.tobytes()))
                            except: ems.append("Erro")
                        
                        while len(ems) < 3: ems.append("N/A")
                        
                        rows.append([
                            st.session_state.participant_id, study_id_from_url, r['stimulus'], r['timestamp'],
                            ems[0], ems[1], ems[2], 
                            r['ans_liking'] or "", ", ".join(r['ans_emotions']), r['ans_word']
                        ])
                    
                    sh = connect_gsheets()
                    if sh:
                        try:
                            sh.worksheet("Resultados").append_rows(rows)
                            st.success("Dados salvos!")
                            st.session_state.data_saved = True
                        except Exception as e: st.error(f"Erro: {e}")
            st.balloons()
            st.write("Obrigado!")
    else:
        st.error("Estudo inválido.")

else:
    # --- MODO ADMIN ---
    st.title("Painel do Pesquisador")
    
    sh = connect_gsheets()
    if sh: st.success(f"✅ Conectado a: {NOME_DA_SUA_PLANILHA}")
    
    if 'study_items' not in st.session_state: st.session_state.study_items = []

    with st.sidebar:
        st.header("Novo Estudo")
        s_name = st.text_input("Nome do Estudo")
        w_msg = st.text_area("Boas-vindas")
        st.subheader("Adicionar Tarefa")
        with st.form("add"):
            url = st.text_input("URL Imagem")
            name = st.text_input("Nome ID")
            cap = st.text_input("Legenda")
            qs = st.multiselect("Perguntas", ['Nota de Gostar (1-9)', 'Lista de Emoções (Múltipla Escolha)', 'Uma Palavra que Define (Campo de Texto)'])
            if st.form_submit_button("Adicionar") and url and name:
                st.session_state.study_items.append({"name": name, "stimulus_url": url, "caption": cap, "questions": qs})
                st.success("Adicionado")
    
    st.write(f"Itens: {len(st.session_state.study_items)}")
    if st.button("Salvar e Gerar Link", type="primary"):
        if s_name and st.session_state.study_items:
            sh = connect_gsheets()
            if sh:
                sid = generate_study_id()
                cfg = {"study_name": s_name, "welcome_message": w_msg, "items": st.session_state.study_items}
                sh.worksheet("Estudos").append_row([sid, json.dumps(cfg)])
                st.success("Estudo Salvo!")
                st.code(f"{URL_BASE_DA_SUA_APP}?study_id={sid}")
