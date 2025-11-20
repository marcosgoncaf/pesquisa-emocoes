import streamlit as st
from streamlit_webrtc import webrtc_streamer
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

# =================================================================================
# --- CONFIGURAÇÕES GLOBAIS ---
# =================================================================================
# ⬇️ IMPORTANTE: O nome deve ser IDÊNTICO ao da sua planilha no Google Sheets
NOME_DA_SUA_PLANILHA = "Resultados Pesquisa Emoções"

# ⬇️ ATUALIZE ISTO DEPOIS DO DEPLOY: A URL oficial do seu site no Streamlit
URL_BASE_DA_SUA_APP = "https://seu-link-final-aqui.streamlit.app"

st.set_page_config(page_title="Plataforma de Pesquisa", layout="centered")

# =================================================================================
# --- FUNÇÕES DE APOIO (BACKEND) ---
# =================================================================================
def generate_study_id(length=8):
    """Gera um ID aleatório para o estudo."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

@st.cache_resource
def connect_gsheets():
    """Conecta ao Google Sheets buscando os Secrets do Streamlit."""
    try:
        # Tenta pegar as credenciais dos segredos do Streamlit Cloud
        if "gcp_service_account" in st.secrets:
            creds = st.secrets["gcp_service_account"]
            sa = gspread.service_account_from_dict(creds)
            sh = sa.open(NOME_DA_SUA_PLANILHA)
            return sh
        else:
            st.error("Segredos (Secrets) não configurados. Configure o 'gcp_service_account' no painel do Streamlit.")
            return None
    except Exception as e:
        st.error(f"Erro ao conectar na planilha '{NOME_DA_SUA_PLANILHA}'. Verifique o nome e as permissões. Erro: {e}")
        return None

def analyze_emotion(frame):
    """Analisa a emoção de um frame usando DeepFace."""
    try:
        analysis = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)
        if isinstance(analysis, list) and len(analysis) > 0:
            return analysis[0]['dominant_emotion']
        return "rosto_nao_detectado"
    except Exception:
        return "erro_na_analise"

class VideoProcessor:
    """Processador de vídeo para captura em segundo plano."""
    def __init__(self):
        self.frames_buffer = []
        self.capture_lock = threading.Lock()

    def recv(self, frame: VideoFrame) -> VideoFrame:
        frm = frame.to_ndarray(format="bgr24")
        with self.capture_lock:
            # Se o gatilho de captura estiver ativo, armazena o frame
            if st.session_state.get('start_capture', False):
                self.frames_buffer.append(frm)
        # Retorna um frame preto 1x1 para o frontend (câmera invisível)
        return VideoFrame.from_ndarray(np.zeros((1, 1, 3), np.uint8), format="bgr24")

# =================================================================================
# --- LÓGICA DE ROTEAMENTO (Admin vs. Participante) ---
# =================================================================================

params = st.query_params
study_id_from_url = params.get("study_id")

if study_id_from_url:
    # ############################
    # ### MODO PARTICIPANTE ###
    # ############################

    @st.cache_data(ttl=300)
    def load_study_config(_study_id):
        sh = connect_gsheets()
        if not sh: return None
        try:
            worksheet = sh.worksheet("Estudos")
            cell = worksheet.find(_study_id)
            if cell:
                config_json = worksheet.cell(cell.row, 2).value
                return json.loads(config_json)
        except Exception as e:
            st.error(f"Erro ao carregar o estudo: {e}")
        return None

    study_config = load_study_config(study_id_from_url)

    if not study_config:
        st.error("Estudo não encontrado ou link inválido.")
    else:
        # Inicialização da memória do participante
        if 'participant_stage' not in st.session_state:
            st.session_state.participant_stage = 'welcome'
        if 'participant_results' not in st.session_state:
            st.session_state.participant_results = []
        if 'current_item' not in st.session_state:
            st.session_state.current_item = 0
            
        # Esconde a webcam visualmente
        st.markdown("<style>div[data-testid='stWebRTC']{display: none;}</style>", unsafe_allow_html=True)
        
        # Inicia o componente da webcam
        webrtc_ctx = webrtc_streamer(
            key="webcam", 
            video_processor_factory=VideoProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )

        # --- TELA 1: BOAS VINDAS ---
        if st.session_state.participant_stage == 'welcome':
            st.title("Bem-vindo(a) à Pesquisa")
            st.write(study_config.get('welcome_message', 'Sua participação é muito importante.'))
            st.session_state.participant_id = st.text_input("Por favor, insira um ID para você (ex: suas iniciais ou um apelido):", key="pid_welcome")
            
            if st.button("Iniciar"):
                if st.session_state.participant_id:
                    st.session_state.participant_stage = 'test'
                    st.rerun()
                else:
                    st.warning("Por favor, insira um ID para continuar.")

        # --- TELA 2: APRESENTAÇÃO DO ESTÍMULO E CAPTURA ---
        elif st.session_state.participant_stage == 'test':
            current_idx = st.session_state.current_item
            
            if current_idx >= len(study_config['items']):
                st.session_state.participant_stage = 'end'
                st.rerun()
            
            item = study_config['items'][current_idx]
            
            # Limpa o buffer da câmera
            if webrtc_ctx.video_processor:
                with webrtc_ctx.video_processor.capture_lock:
                    webrtc_ctx.video_processor.frames_buffer.clear()
            
            st.header(f"Tarefa {current_idx + 1} de {len(study_config['items'])}")
            
            try:
                st.image(item['stimulus_url'], caption=item['caption'], use_column_width=True)
            except:
                st.image("https://via.placeholder.com/400?text=Erro+ao+Carregar+Imagem", caption="Erro ao carregar imagem")
                st.error(f"Não foi possível carregar a imagem: {item['stimulus_url']}")

            if st.button("OK, observei.", key=f"ok_{current_idx}"):
                st.session_state.start_capture = True
                with st.spinner("Registrando reação... Aguarde 5 segundos."):
                    time.sleep(5)
                st.session_state.start_capture = False
                
                # Coleta frames
                captured_frames = []
                if webrtc_ctx.video_processor:
                    with webrtc_ctx.video_processor.capture_lock:
                        captured_frames = webrtc_ctx.video_processor.frames_buffer.copy()
                
                # Seleciona 3 frames
                final_frames = []
                if len(captured_frames) > 0: final_frames.append(captured_frames[0])
                if len(captured_frames) > 2:
                    final_frames.append(captured_frames[len(captured_frames) // 2])
                    final_frames.append(captured_frames[-1])
                elif len(captured_frames) == 2: final_frames.append(captured_frames[1])
                
                st.session_state.last_captured_frames = final_frames
                st.success("Captura concluída!")
                time.sleep(0.5)
                st.session_state.participant_stage = 'questionnaire'
                st.rerun()

        # --- TELA 3: QUESTIONÁRIO ---
        elif st.session_state.participant_stage == 'questionnaire':
            current_idx = st.session_state.current_item
            item = study_config['items'][current_idx]
            
            st.header("Questionário")
            
            ans_liking = None
            ans_emotions = []
            ans_word = ""

            # Renderiza perguntas condicionalmente
            if 'Nota de Gostar (1-9)' in item['questions']:
                ans_liking = st.slider("Quanto você gostou?", 1, 9, 5, key=f"like_{current_idx}")
            
            if 'Lista de Emoções (Múltipla Escolha)' in item['questions']:
                ans_emotions = st.multiselect("Quais emoções você sentiu?", ['Alegre', 'Calmo', 'Interessado', 'Nojo', 'Medo', 'Triste', 'Surpreso', 'Neutro', 'Curioso', 'Decepcionado'], key=f"cata_{current_idx}")
            
            if 'Uma Palavra que Define (Campo de Texto)' in item['questions']:
                ans_word = st.text_input("Se pudesse definir sua sensação em uma palavra, qual seria?", key=f"word_{current_idx}")

            if st.button("Próxima Tarefa", key=f"next_{current_idx}"):
                # Armazena resultados
                result = {
                    'participant_id': st.session_state.participant_id,
                    'study_id': study_id_from_url,
                    'stimulus_name': item['name'],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'answers': {
                        'liking_score': ans_liking,
                        'cata_emotions': ans_emotions,
                        'defining_word': ans_word
                    },
                    'frames': st.session_state.last_captured_frames
                }
                st.session_state.participant_results.append(result)
                st.session_state.current_item += 1
                st.session_state.participant_stage = 'test'
                st.rerun()

        # --- TELA 4: FINALIZAÇÃO E SALVAMENTO ---
        elif st.session_state.participant_stage == 'end':
            st.title("Pesquisa Concluída!")
            
            # Verifica se já salvou para evitar duplicação
            if not st.session_state.get('data_saved', False):
                with st.spinner("Analisando e salvando seus dados..."):
                    final_data_to_save = []
                    
                    for result in st.session_state.participant_results:
                        # Analisa emoções dos frames
                        emotions = [analyze_emotion(cv2.imencode('.jpg', f)[1].tobytes()) for f in result['frames']]
                        # Garante 3 colunas de emoção
                        while len(emotions) < 3: emotions.append("N/A")
                        
                        # Monta o dicionário da linha
                        row_dict = {
                            'id_participante': result['participant_id'],
                            'id_estudo': result['study_id'],
                            'estimulo': result['stimulus_name'],
                            'timestamp': result['timestamp'],
                            'emocao_1': emotions[0],
                            'emocao_2': emotions[1],
                            'emocao_3': emotions[2],
                            'nota_gostar': result['answers'].get('liking_score', ''),
                            'emocoes_declaradas': ", ".join(result['answers'].get('cata_emotions', [])),
                            'palavra_definidora': result['answers'].get('defining_word', '')
                        }
                        final_data_to_save.append(row_dict)
                    
                    try:
                        sh = connect_gsheets()
                        if sh:
                            worksheet = sh.worksheet("Resultados")
                            
                            # Garante a ordem das colunas
                            ordered_columns = [
                                'id_participante', 'id_estudo', 'estimulo', 'timestamp', 'emocao_1',
                                'emocao_2', 'emocao_3', 'nota_gostar', 'emocoes_declaradas', 'palavra_definidora'
                            ]
                            
                            # Cria DataFrame
                            df_to_append = pd.DataFrame(final_data_to_save)
                            
                            # Garante que todas as colunas existam no DF
                            for col in ordered_columns:
                                if col not in df_to_append.columns:
                                    df_to_append[col] = ""
                            
                            # Reordena
                            df_to_append = df_to_append[ordered_columns]

                            worksheet.append_rows(df_to_append.values.tolist(), value_input_option='USER_ENTERED')
                            st.success("Seus dados foram salvos com segurança!")
                            st.session_state.data_saved = True
                        else:
                            st.error("Erro ao conectar na planilha para salvar.")
                    except Exception as e:
                        st.error(f"Não foi possível salvar os dados na planilha. Erro: {e}")
            
            st.balloons()
            st.write("Muito obrigado pela sua participação!")
        else:
            st.balloons()
            st.write("Muito obrigado pela sua participação!")

else:
    # ############################
    # ### MODO PESQUISADOR (ADMIN) ###
    # ############################
    st.title("Plataforma de Pesquisa - Modo Pesquisador")
    
    # Verifica conexão com Planilha
    sh = connect_gsheets()
    if sh:
        st.success("✅ Conectado ao Google Sheets com sucesso via Segredos!")
    
    if 'study_items' not in st.session_state: st.session_state.study_items = []

    with st.sidebar:
        st.header("Configuração")
        st.markdown("---")
        st.header("Montar Novo Estudo")
        study_name = st.text_input("Nome do Estudo")
        welcome_message = st.text_area("Mensagem de Boas-vindas", "Bem-vindo(a) à nossa pesquisa.")
        
        st.subheader("Adicionar Nova Tarefa")
        with st.form("new_task_form", clear_on_submit=True):
            stimulus_url = st.text_input("URL da Imagem/Estímulo", placeholder="https://imgur.com/...")
            stimulus_name = st.text_input("Nome do Estímulo (ex: Produto A)")
            caption = st.text_input("Legenda/Instrução para a tarefa")
            questions = st.multiselect(
                "Perguntas para esta tarefa",
                ['Nota de Gostar (1-9)', 'Lista de Emoções (Múltipla Escolha)', 'Uma Palavra que Define (Campo de Texto)']
            )
            submitted = st.form_submit_button("Adicionar Tarefa ao Estudo")
            if submitted:
                if stimulus_url and stimulus_name:
                    new_item = { "name": stimulus_name, "stimulus_url": stimulus_url, "caption": caption, "questions": questions }
                    st.session_state.study_items.append(new_item)
                    st.success(f"Tarefa '{stimulus_name}' adicionada!")
                else:
                    st.warning("Preencha a URL e o Nome do Estímulo.")

    st.subheader("Roteiro do Estudo Atual")
    if not st.session_state.study_items:
        st.info("Adicione tarefas ao estudo usando o painel à esquerda.")
    else:
        for i, item in enumerate(st.session_state.study_items):
            st.write(f"{i+1}. **{item['name']}** - Legenda: {item['caption']}")
        if st.button("Limpar Roteiro"):
            st.session_state.study_items = []
            st.rerun()
    
    st.markdown("---")
    if st.button("Salvar Estudo e Gerar Link", type="primary"):
        if study_name and st.session_state.study_items:
            with st.spinner("Salvando..."):
                study_id = generate_study_id()
                study_config = { "study_name": study_name, "welcome_message": welcome_message, "items": st.session_state.study_items }
                config_json = json.dumps(study_config)
                
                try:
                    sh = connect_gsheets()
                    worksheet = sh.worksheet("Estudos")
                    worksheet.append_row([study_id, config_json])
                    
                    full_url = f"{URL_BASE_DA_SUA_APP}?study_id={study_id}"

                    st.success("Estudo salvo!")
                    st.subheader("Link para os Participantes:")
                    st.code(full_url)
                except Exception as e:
                    st.error(f"Falha ao salvar. Erro: {e}")
        else:
            st.error("Dê um nome ao estudo e adicione pelo menos uma tarefa.")
