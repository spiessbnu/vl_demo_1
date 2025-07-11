import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re

# --- Configuração da Página e Logger ---
st.set_page_config(
    page_title="VL demo 1 - Matemática",
    page_icon="🤖",
    layout="centered"
)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# --- INICIALIZAÇÃO DO SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "log_messages" not in st.session_state:
    st.session_state.log_messages = []
if "aluno_ano" not in st.session_state:
    st.session_state.aluno_ano = 5

def log_to_terminal(message):
    st.session_state.log_messages.append(message)
    logging.info(message)

def corrigir_notacao_latex(texto: str) -> str:
    padroes = [
        r"(?<!\$)\\frac\{[^\}]+\}\{[^\}]+\}",
        r"(?<!\$)\\sqrt\{[^\}]+\}",
        r"(?<!\$)\\sum_\{[^\}]+\}\^\{[^\}]+\}",
        r"(?<!\$)[a-zA-Z]\^[0-9]+",
    ]
    def adicionar_cifroes(match):
        return f"${match.group(0)}$"
    for padrao in padroes:
        texto = re.sub(padrao, adicionar_cifroes, texto)
    return texto

@st.cache_data
def carregar_dados():
    log_to_terminal("Iniciando carregamento dos dados...")
    try:
        df = pd.read_parquet("dados_curriculares_enriquecidos.parquet")
        matriz_embeddings = np.array(df['embedding'].tolist())
        log_to_terminal("Dados carregados com sucesso!")
        return df, matriz_embeddings
    except FileNotFoundError:
        st.error("Arquivo 'dados_curriculares_enriquecidos.parquet' não encontrado.")
        log_to_terminal("ERRO: Arquivo de dados não encontrado.")
        return None, None

def gerar_embedding_query(texto, client):
    log_to_terminal(f"Gerando embedding para a query: '{texto[:30]}...'")
    try:
        response = client.embeddings.create(input=[texto], model="text-embedding-3-large")
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Erro ao gerar embedding: {e}")
        log_to_terminal(f"ERRO na API de Embeddings: {e}")
        return None

def buscar_conteudo_relevante(query_embedding, df, matriz_embeddings, ano_aluno, top_k=5):
    if query_embedding is None:
        return pd.DataFrame()
    log_to_terminal("Calculando similaridade de cosseno...")
    scores = cosine_similarity([query_embedding], matriz_embeddings)[0]
    df['similaridade'] = scores
    log_to_terminal(f"Ranqueando resultados. Prioridade para o {ano_aluno}º ano.")
    df_ranqueado = df.sort_values(
        by=['Ano', 'similaridade'],
        ascending=[True, False],
        key=lambda col: col if col.name != 'Ano' else col != ano_aluno
    )
    resultados = df_ranqueado.head(top_k)
    log_to_terminal("Top 5 resultados (Índice | Ano | Score):")
    for i, row in resultados.iterrows():
        log_to_terminal(f"- {i} | {row['Ano']}º ano | {row['similaridade']:.4f}")
    return resultados.iloc[[0]]

try:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except (KeyError, FileNotFoundError):
    st.error("Chave da API da OpenAI não encontrada. Por favor, configure o arquivo .streamlit/secrets.toml")
    st.stop()

df, matriz_embeddings = carregar_dados()
if df is None:
    st.stop()

st.title("🤖 VL demo 1 - Matemática")
st.caption("Um assistente baseado no currículo de SC para te ajudar a estudar.")

with st.sidebar:
    st.header("Configurações")
    anos_disponiveis = sorted(df['Ano'].unique())
    ano_selecionado = st.selectbox(
        "Qual ano você está cursando?",
        options=anos_disponiveis,
        index=anos_disponiveis.index(st.session_state.aluno_ano)
    )
    if ano_selecionado != st.session_state.aluno_ano:
        st.session_state.aluno_ano = ano_selecionado
        st.rerun()
    st.divider()
    with st.expander("🔌 Terminal de Debug", expanded=True):
        log_container = st.container(height=300)
        log_text = "\n".join(st.session_state.log_messages)
        log_container.text(log_text)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("O que vamos estudar hoje?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            st.session_state.log_messages = []
            log_to_terminal("--- NOVA QUERY RECEBIDA ---")
            query_embedding = gerar_embedding_query(prompt, client)
            df_contexto = buscar_conteudo_relevante(
                query_embedding, df.copy(), matriz_embeddings, st.session_state.aluno_ano
            )
            if not df_contexto.empty:
                contexto_row = df_contexto.iloc[0]
                contexto_curricular = contexto_row['texto_completo']

                # --- LOG DETALHADO DO CONTEXTO (NOVA SEÇÃO) ---
                log_to_terminal("\n--- CONTEXTO SELECIONADO PARA O LLM ---")
                log_to_terminal(f"Índice: {contexto_row.name}")
                log_to_terminal(f"Ano: {contexto_row['Ano']}")
                log_to_terminal(f"Unidade: {contexto_row['Unidade Temática']}")
                log_to_terminal(f"Conteúdo: {contexto_row['texto_completo']}")
                log_to_terminal(f"Score: {contexto_row['similaridade']:.4f}")
                log_to_terminal("---------------------------------------")
                log_to_terminal(contexto_curricular)
                log_to_terminal("---------------------------------------\n")
                # --- FIM DA NOVA SEÇÃO ---

                system_prompt = f"""
                Você é um tutor de matemática amigável, paciente e didático.
                Sua missão é ajudar um aluno do {st.session_state.aluno_ano}º ano.
                Use o seguinte CONTEXTO CURRICULAR para basear sua resposta. Não invente informações.
                Seja claro, use exemplos simples e sempre responda em português do Brasil.

                IMPORTANTE: Sempre que você escrever notação matemática, como frações, raízes ou equações, coloque-a entre cifrões ($).
                Por exemplo, para a fração 3/4, escreva: $\\frac{{3}}{{4}}$. Para uma equação, escreva: $x^2 + y^2 = z^2$.

                CONTEXTO CURRICULAR:
                {contexto_curricular}
                """
                mensagens_para_api = [{"role": "system", "content": system_prompt}]
                for msg in st.session_state.messages:
                    mensagens_para_api.append(msg)
                log_to_terminal("Enviando requisição para a API 'gpt-4o-mini'...")
                try:
                    stream = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=mensagens_para_api,
                        stream=True,
                    )
                    resposta_completa = ""
                    placeholder = st.empty()
                    for chunk in stream:
                        resposta_completa += (chunk.choices[0].delta.content or "")
                        placeholder.markdown(resposta_completa + "▌")
                    resposta_corrigida = corrigir_notacao_latex(resposta_completa)
                    placeholder.markdown(resposta_corrigida)
                    st.session_state.messages.append({"role": "assistant", "content": resposta_corrigida})
                    log_to_terminal("Resposta da API recebida e exibida.")
                    if resposta_completa != resposta_corrigida:
                        log_to_terminal("Notação LaTeX corrigida na resposta da API.")
                except Exception as e:
                    st.error(f"Ocorreu um erro com a API da OpenAI: {e}")
                    log_to_terminal(f"ERRO na API de Chat: {e}")
            else:
                st.write("Não consegui encontrar um conteúdo diretamente relacionado no currículo. Você pode tentar reformular a pergunta?")
                log_to_terminal("Nenhum contexto relevante encontrado.")
    st.rerun()
