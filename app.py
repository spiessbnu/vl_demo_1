import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
import logging
import json
import re

# --- Configuração da Página e Logger ---
st.set_page_config(page_title="Tutor de Matemática", page_icon="🤖", layout="centered")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# --- INICIALIZAÇÃO DO SESSION STATE ---
if "app_state" not in st.session_state:
    st.session_state.app_state = "COLETA_INFO"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "log_messages" not in st.session_state:
    st.session_state.log_messages = []
if "aluno_ano" not in st.session_state:
    st.session_state.aluno_ano = None
if "aluno_nome" not in st.session_state:
    st.session_state.aluno_nome = ""
if "unidade_tematica_atual" not in st.session_state:
    st.session_state.unidade_tematica_atual = ""
if "topico_selecionado_idx" not in st.session_state:
    st.session_state.topico_selecionado_idx = None

# --- Funções Auxiliares (Logging, LaTeX, etc.) ---
def log_to_terminal(message):
    st.session_state.log_messages.append(str(message))
    logging.info(message)

def corrigir_latex_inline(texto: str) -> str:
    pattern = r'(?<!\$)\\(frac|rac)\{([^\}]+)\}\{([^\}]+)\}(?!\$)'
    def normalizar_e_delimitar(match):
        numerador = match.group(2)
        denominador = match.group(3)
        return f"$\\frac{{{numerador}}}{{{denominador}}}$"
    return re.sub(pattern, normalizar_e_delimitar, texto)

# --- Funções de Carregamento de Dados e RAG ---
@st.cache_data
def carregar_dados():
    log_to_terminal("Iniciando carregamento dos dados...")
    try:
        df = pd.read_parquet("dados_curriculares_enriquecidos.parquet")
        matriz_embeddings = np.array(df['embedding'].tolist())
        return df, matriz_embeddings
    except FileNotFoundError:
        st.error("Arquivo 'dados_curriculares_enriquecidos.parquet' não encontrado.")
        return None, None

def gerar_embedding_query(texto, client):
    log_to_terminal(f"Gerando embedding para a query: '{texto}'")
    try:
        response = client.embeddings.create(input=[texto], model="text-embedding-3-large")
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Erro ao gerar embedding: {e}")
        return None

def buscar_conteudo_inicial(query, df, matriz_embeddings):
    # Esta função faz a busca inicial para definir a "playlist"
    if not query: return None
    log_to_terminal("Buscando conteúdo inicial...")
    embedding = gerar_embedding_query(query, client)
    if embedding is None: return None
    
    scores = cosine_similarity([embedding], matriz_embeddings)[0]
    df['similaridade'] = scores
    
    # Retorna o índice do item com maior similaridade
    top_hit_idx = df['similaridade'].idxmax()
    log_to_terminal(f"Melhor resultado inicial encontrado no índice {top_hit_idx} com score {df.loc[top_hit_idx, 'similaridade']:.4f}")
    return top_hit_idx

# --- NOVA FUNÇÃO PARA EXTRAIR DADOS DO TEXTO ---
def extrair_dados_iniciais(texto_usuario, client):
    log_to_terminal("Extraindo dados da primeira mensagem do usuário...")
    prompt_extracao = f"""
    Extraia o nome do aluno, o ano (como um número inteiro) e o assunto principal da frase abaixo.
    Responda APENAS com um objeto JSON com as chaves "nome", "ano" e "assunto".
    Frase: "{texto_usuario}"
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_extracao}],
            response_format={"type": "json_object"}
        )
        dados = json.loads(response.choices[0].message.content)
        log_to_terminal(f"Dados extraídos: {dados}")
        return dados
    except Exception as e:
        log_to_terminal(f"Erro ao extrair dados iniciais: {e}")
        return None

# --- Inicialização ---
try:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except (KeyError, FileNotFoundError):
    st.error("Chave da API da OpenAI não encontrada. Configure o arquivo .streamlit/secrets.toml")
    st.stop()

df, matriz_embeddings = carregar_dados()
if df is None: st.stop()

# --- Interface Principal e Máquina de Estados ---
st.title("🤖 Tutor Inteligente de Matemática")

# ESTADO 1: COLETA DE INFORMAÇÕES
if st.session_state.app_state == "COLETA_INFO":
    st.info("👋 Olá! Para começarmos, diga seu nome, o ano que você está cursando e o assunto que gostaria de estudar hoje.")
    st.caption("Exemplo: 'Meu nome é Ana, sou do 8º ano e quero aprender sobre o teorema de Pitágoras.'")

    if prompt := st.chat_input("Diga seu nome, ano e assunto..."):
        dados_iniciais = extrair_dados_iniciais(prompt, client)
        if dados_iniciais:
            st.session_state.aluno_nome = dados_iniciais.get("nome", "estudante")
            st.session_state.aluno_ano = dados_iniciais.get("ano", 7) # Padrão para 7º ano se não encontrar
            assunto = dados_iniciais.get("assunto", prompt)

            top_hit_idx = buscar_conteudo_inicial(assunto, df, matriz_embeddings)
            if top_hit_idx is not None:
                st.session_state.unidade_tematica_atual = df.loc[top_hit_idx, "Unidade Temática"]
                st.session_state.topico_selecionado_idx = top_hit_idx
                st.session_state.app_state = "SELECAO_TOPICO"
                st.rerun()
            else:
                st.error("Não consegui identificar um tópico. Pode tentar de novo?")
        else:
            st.error("Desculpe, não consegui entender sua mensagem. Por favor, tente o formato do exemplo.")

# ESTADO 2: SELEÇÃO DE TÓPICO (A "PLAYLIST")
elif st.session_state.app_state == "SELECAO_TOPICO":
    st.markdown(f"### Olá, {st.session_state.aluno_nome}!")
    st.markdown(f"Legal! Vamos falar sobre **{st.session_state.unidade_tematica_atual}**. Encontrei estes tópicos relacionados no currículo do seu ano e de anos anteriores. O que mais se parece com o que você procura está marcado.")

    # Filtra o DataFrame pela unidade temática e ordena
    playlist_df = df[df["Unidade Temática"] == st.session_state.unidade_tematica_atual].sort_values("Ordem")

    for idx, row in playlist_df.iterrows():
        col1, col2, col3 = st.columns([5, 3, 2])
        with col1:
            st.markdown(f"**{row['Objetos do conhecimento']}**")
            st.caption(f"Ano: {row['Ano']} | Ordem: {row['Ordem']}")
        with col2:
            if idx == st.session_state.topico_selecionado_idx:
                st.success("📍 Assunto mais próximo")
        with col3:
            if st.button("Estudar este tópico", key=f"topic_{idx}"):
                # Ao clicar, inicia o chat sobre este tópico
                st.session_state.topico_selecionado_idx = idx
                st.session_state.app_state = "CHAT"
                st.session_state.messages = [
                    {"role": "assistant", "content": json.dumps({
                        "response": [f"Ok! Vamos focar em **{df.loc[idx, 'Objetos do conhecimento']}**. O que especificamente você gostaria de saber? Me peça uma explicação, exemplos ou exercícios!"]
                    })}
                ]
                st.rerun()
    st.divider()

# ESTADO 3: CHAT (LÓGICA ANTERIOR ADAPTADA)
elif st.session_state.app_state == "CHAT":
    # A lógica de renderização e de chat que já tínhamos
    def renderizar_mensagem(message):
        # ... (código completo na seção abaixo) ...
    def criar_query_contextualizada(historico_mensagens, topico_atual):
        # ... (código completo na seção abaixo) ...

    # Renderiza o histórico de mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            renderizar_mensagem(message)
    
    # Lógica de input e resposta do chat
    if prompt := st.chat_input("Peça uma explicação, exemplos ou exercícios!"):
        # ... (lógica completa na seção abaixo) ...
        pass # Placeholder

# --- Bloco final para o Estado de CHAT (para manter a legibilidade) ---
# Cole a lógica da função de renderização e o loop de chat aqui

def renderizar_mensagem(message):
    if message["role"] == "user":
        st.markdown(message["content"])
        return
    try:
        data = json.loads(message["content"])
        for line in data.get("response", []):
            st.markdown(corrigir_latex_inline(line), unsafe_allow_html=True)
    except (json.JSONDecodeError, TypeError):
        st.markdown(corrigir_latex_inline(message["content"]), unsafe_allow_html=True)

def criar_query_contextualizada(historico_mensagens: list, topico_atual: str) -> str:
    log_to_terminal("Criando query contextualizada para a busca...")
    mensagens_relevantes = historico_mensagens[-4:]
    contexto_str = " ".join([f"{msg['role']}: {msg['content']}" for msg in mensagens_relevantes])
    query_final = f"Contexto do tópico: {topico_atual}. Conversa recente: {contexto_str}"
    return query_final

if st.session_state.app_state == "CHAT":
    if prompt := st.chat_input("Peça uma explicação, exemplos ou exercícios!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                st.session_state.log_messages = []
                log_to_terminal("--- NOVA QUERY (CHAT) ---")

                topico_atual_texto = df.loc[st.session_state.topico_selecionado_idx, 'texto_completo']
                query_para_rag = criar_query_contextualizada(st.session_state.messages, topico_atual_texto)
                log_to_terminal(f"Query Contextualizada para RAG: '{query_para_rag}'")

                # A busca agora é sempre focada no tópico já selecionado, mas a query ajuda a refinar.
                # Para simplificar, vamos usar o texto do tópico principal como contexto fixo nesta fase.
                contexto_curricular = topico_atual_texto
                log_to_terminal(f"\n--- Usando contexto do tópico selecionado (Índice: {st.session_state.topico_selecionado_idx}) ---")

                system_prompt = f"""
                Você é um tutor de matemática. Sua resposta DEVE ser um objeto JSON válido com uma chave "response" contendo uma lista de strings.
                Use Markdown e LaTeX (com $...$) para formatar.

                REGRAS DE FORMATAÇÃO:
                1. Blocos matemáticos ($$) devem estar em seu próprio item na lista.
                2. Fórmulas inline ($) podem estar no meio de um parágrafo.
                3. Use espaços antes e depois de blocos matemáticos.
                4. Números mistos: `$1\\frac{{1}}{{4}}$`.

                O aluno está estudando o tópico: "{df.loc[st.session_state.topico_selecionado_idx, 'Objetos do conhecimento']}".
                Use o CONTEXTO CURRICULAR abaixo para responder a pergunta dele.
                CONTEXTO CURRICULAR: {contexto_curricular}
                """
                mensagens_para_api = [{"role": "system", "content": system_prompt}] + st.session_state.messages
                
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini", messages=mensagens_para_api,
                        response_format={"type": "json_object"}
                    )
                    resposta_json_str = response.choices[0].message.content
                    log_to_terminal("\n--- RESPOSTA BRUTA DA API (JSON) ---")
                    log_to_terminal(resposta_json_str)
                    st.session_state.messages.append({"role": "assistant", "content": resposta_json_str})
                except Exception as e:
                    st.error(f"Ocorreu um erro com a API da OpenAI: {e}")
                    log_to_terminal(f"ERRO na API de Chat: {e}")
        st.rerun()

# Bloco final para exibir o terminal de debug
with st.sidebar:
    st.header("Debug")
    with st.expander("🔌 Terminal de Debug", expanded=True):
        log_container = st.container(height=300)
        log_text = "\n".join(st.session_state.log_messages)
        log_container.text(log_text)
