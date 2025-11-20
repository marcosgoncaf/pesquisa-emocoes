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
# --- CONFIGURAÇÕES GLOBAIS ---
# =================================================================================
# IMPORTANTE: Para evitar erros de memória com DeepFace no Cloud
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

NOME_DA_SUA_PLANILHA = "Resultados Pesquisa Emoções"
# Atualize com seu link real
URL_BASE_DA_SUA_APP = "https://pesquisa-emocoes-jjhae3nwqqs4mslggexsmn.streamlit.app"

st.set_page_config(page_title="Plataforma de Pesquisa", layout="centered")

# =================================================================================
# --- FUNÇÕES DE APOIO ---
# =================================================================================
def generate_study_id(length=8):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

@st.cache_resource
def connect_gsheets():
    creds = None
    try:
        if "gcp_service_account" in st.secrets:
            creds = st.secrets["gcp_service_account"]
    except Exception:
        pass
    
    if creds is None:
        if 'json_creds_content' in st.session_state and st.session_state.json_creds_content:
            try:
                creds = json.loads(st.session_state.json_creds_content)
            except json.JSONDecodeError:
                st.error("JSON inválido.")
                st.stop()
        else:
            return None

    try:
        sa = gspread.service_account_from_dict(creds)
        sh = sa.open(NOME_DA_SUA_PLANILHA)
        return sh
    except Exception as e:
        st.error(f"Erro planilha: {e}")
        st.stop()

def analyze_emotion(frame_bytes):
    """Analisa a imagem. Retorna a emoção ou 'Erro'."""
    try:
        # Converte bytes para imagem OpenCV
        nparr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # DeepFace analysis
        analysis = DeepFace.analyze(img_path=img, actions=['emotion'], enforce_detection=False)
        if isinstance(analysis, list) and len(analysis) > 0:
            return analysis[0]['dominant_emotion']
        return "rosto_nao_detectado"
    except Exception as e:
        print(f"Erro DeepFace: {e}")
        return "erro_analise"

class VideoProcessor:
    def __init__(self):
        self.frames_buffer = []
        self.capture_lock = threading.Lock()
        # Flag interna para controlar captura sem depender só do session_state na thread
        self.recording = False 

    def recv(self, frame: VideoFrame) -> VideoFrame:
        # Converte frame
        frm = frame.to_ndarray(format="bgr24")
        
        # Lógica de captura
        with self.capture_lock:
            if self.recording:
                self.frames_buffer.append(frm)
        
        # Retorna frame preto (invisível)
        return VideoFrame.from_ndarray(np.zeros((1, 1, 3), np.uint8), format="bgr24")

# =================================================================================
# --- APP ---
# =================================================================================
params = st.query_params
study_id_from_url = params.get("study_id")

if study_id_from_url:
    # --- MODO PARTICIPANTE ---
    
    # Função de carregamento simples
    def load_study_config(_study_id):
        sh = connect_gsheets()
        if not sh: return None
        try:
            worksheet = sh.worksheet("Estudos")
            cell = worksheet.find(_study_id)
            if cell: return json.loads(worksheet.cell(cell.row, 2).value)
        except: pass
        return None

    # Aviso se faltar credencial no modo teste
    if 'json_creds_content' not in st.session_state and 'gcp_service_account' not in st.secrets:
         st.warning("⚠️ Modo Teste: Carregue o JSON na tela inicial primeiro.")
         st.stop()
    
    study_config = load_study_config(study_id_from_url)

    if study_config:
        # Inicializa estados
        if 'participant_stage' not in st.session_state: st.session_state.participant_stage = 'welcome'
        if 'participant_results' not in st.session_state: st.session_state.participant_results = []
        if 'current_item' not in st.session_state: st.session_state.current_item = 0
        if 'last_captured_frames' not in st.session_state: st.session_state.last_captured_frames = []

        # CSS para esconder visualização
        st.markdown("<style>div[data-testid='stWebRTC']{display: none;}</style>", unsafe_allow_html=True)
        
        # Configuração WebRTC mais estável
        rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        
        webrtc_ctx = webrtc_streamer(
            key="webcam", 
            video_processor_factory=VideoProcessor,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )

        # --- FLUXO ---
        if st.session_state.participant_stage == 'welcome':
            st.title("Bem-vindo(a)")
            st.write(study_config.get('welcome_message', ''))
            st.session_state.participant_id = st.text_input("Seu ID:")
            if st.button("Iniciar"):
                if st.session_state.participant_id:
                    st.session_state.participant_stage = 'test'
                    st.rerun()
                else: st.warning("Preencha seu ID.")

        elif st.session_state.participant_stage == 'test':
            idx = st.session_state.current_item
            if idx >= len(study_config['items']):
                st.session_state.participant_stage = 'end'
                st.rerun()
            
            item = study_config['items'][idx]
            
            # Limpa buffer anterior
            if webrtc_ctx.video_processor:
                with webrtc_ctx.video_processor.capture_lock:
                    webrtc_ctx.video_processor.frames_buffer = []

            st.header(f"Tarefa {idx + 1}")
            try: st.image(item['stimulus_url'], caption=item['caption'], use_column_width=True)
            except: st.image("https://via.placeholder.com/400", caption="Erro imagem")
            
            # Botão de Captura
            if st.button("OK, observei", key=f"btn_{idx}"):
                
                # Verifica se a câmera está ativa antes de tentar capturar
                if not webrtc_ctx.state.playing or not webrtc_ctx.video_processor:
                    st.error("🚨 Erro na câmera. Por favor, recarregue a página e permita o uso da webcam.")
                    st.stop()

                # Ativa gravação no processador
                webrtc_ctx.video_processor.recording = True
                
                with st.spinner("Registrando reação... (Mantenha a câmera aberta)"):
                    time.sleep(4) # Captura por 4 segundos
                
                # Para gravação
                webrtc_ctx.video_processor.recording = False
                
                # Recupera frames
                frames = []
                with webrtc_ctx.video_processor.capture_lock:
                    frames = webrtc_ctx.video_processor.frames_buffer.copy()
                
                # VALIDAÇÃO IMPORTANTE: Se não capturou nada, não avança!
                if len(frames) == 0:
                    st.error("⚠️ Nenhuma imagem capturada. Verifique se sua webcam está funcionando e tente clicar em 'OK' novamente.")
                else:
                    # Seleciona 3 frames
                    selected = []
                    if len(frames) > 0: selected.append(frames[0])
                    if len(frames) > 2:
                        selected.append(frames[len(frames)//2])
                        selected.append(frames[-1])
                    elif len(frames) == 2: selected.append(frames[1])
                    
                    st.session_state.last_captured_frames = selected
                    st.success(f"Captura concluída! ({len(frames)} frames processados)")
                    time.sleep(1)
                    st.session_state.participant_stage = 'questionnaire'
                    st.rerun()

        elif st.session_state.participant_stage == 'questionnaire':
            idx = st.session_state.current_item
            item = study_config['items'][idx]
            st.header("Responda")
            
            ans_liking, ans_emotions, ans_word = None, [], ""

            if 'Nota de Gostar (1-9)' in item['questions']:
                ans_liking = st.slider("Quanto você gostou?", 1, 9, 5)
            if 'Lista de Emoções (Múltipla Escolha)' in item['questions']:
                ans_emotions = st.multiselect("O que sentiu?", ['Alegre', 'Calmo', 'Interessado', 'Nojo', 'Medo', 'Triste', 'Surpreso', 'Neutro'])
            if 'Uma Palavra que Define (Campo de Texto)' in item['questions']:
                ans_word = st.text_input("Uma palavra que define:")

            if st.button("Próximo"):
                st.session_state.participant_results.append({
                    'frames': st.session_state.last_captured_frames,
                    'ans_liking': ans_liking, 'ans_emotions': ans_emotions, 'ans_word': ans_word,
                    'stimulus': item['name'], 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.session_state.current_item += 1
                st.session_state.participant_stage = 'test'
                st.rerun()

        elif st.session_state.participant_stage == 'end':
            st.title("Finalizado")
            if not st.session_state.get('data_saved', False):
                with st.spinner("Salvando dados... (Isso pode levar alguns segundos)"):
                    rows = []
                    for r in st.session_state.participant_results:
                        # Processa as emoções aqui no final
                        # Converte cada frame salvo para bytes JPG antes de enviar para análise
                        ems = []
                        for f in r['frames']:
                            try:
                                _, buffer = cv2.imencode('.jpg', f)
                                ems.append(analyze_emotion(buffer.tobytes()))
                            except:
                                ems.append("Erro_Img")
                        
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
                        except Exception as e: st.error(f"Erro ao salvar: {e}")
            st.balloons()
            st.write("Obrigado!")
    else:
        st.error("Estudo inválido.")

else:
    # ADMIN
    st.title("Painel do Pesquisador")
    with st.sidebar:
        json_file = st.file_uploader("Upload JSON (Colab)", type="json")
        if json_file:
            st.session_state.json_creds_content = json_file.getvalue().decode("utf-8")
            st.success("JSON OK")

    if 'study_items' not in st.session_state: st.session_state.study_items = []

    st.subheader("Novo Estudo")
    s_name = st.text_input("Nome do Estudo")
    w_msg = st.text_area("Boas-vindas")
    
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
                
                # Link inteligente
                base = URL_BASE_DA_SUA_APP
                if "COLE_" in base: 
                     st.warning("Configure a URL base no código.")
                     full_url = f"?study_id={sid}"
                else:
                     full_url = f"{base}?study_id={sid}"

                st.success("Estudo Salvo!")
                st.code(full_url)
