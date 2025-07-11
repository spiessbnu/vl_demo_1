import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
import logging

# --- Configuração da Página e Logger ---
st.set_page_config(
    page_title="Tutor de Matemática",
    page_icon="🤖",
    layout="centered"
)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# --- INICIALIZAÇÃO DO SESSION STATE (MOVIDO PARA CIMA) ---
# Este bloco agora executa antes de qualquer outra coisa, garantindo que as chaves existam.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "log_messages" not in st.session_state:
    st.session_state.log_messages = []
if "aluno_ano" not in st.session_state:
    st.session_state.aluno_ano = 5 # Definimos um padrão inicial

def log_to_terminal(message):
    """Função para adicionar logs ao nosso terminal na UI."""
    st.session_state.log_messages.append(message)
    logging.info(message)

# --- Carregamento de Dados (com Cache) ---
@st.cache_data
def carregar_dados():
    """Carrega o DataFrame e extrai a matriz de embeddings.
    Usa o cache do Streamlit para executar esta função apenas uma vez."""
    log_to_terminal("Iniciando carregamento dos dados...")
    try:
        df = pd.read_parquet("dados_curriculares_enriquecidos.parquet")
        matriz_embeddings = np.array(df['embedding'].tolist())
        log_to_terminal("Dados carregados com sucesso!")
        return df, matriz_embeddings
    except FileNotFoundError:
        st.error("Arquivo 'dados_curriculares_enriquecidos.parquet' não encontrado. Verifique o caminho.")
        log_to_terminal("ERRO: Arquivo de dados não encontrado.")
        return None, None

# --- Funções do Núcleo de IA (RAG) ---
def gerar_embedding_query(texto, client):
    """Gera o embedding para a query do usuário."""
    log_to_terminal(f"Gerando embedding para a query: '{texto[:30]}...'")
    try:
        response = client.embeddings.create(input=[texto], model="text-embedding-3-large")
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Erro ao gerar embedding: {e}")
        log_to_terminal(f"ERRO na API de Embeddings: {e}")
        return None

def buscar_conteudo_relevante(query_embedding, df, matriz_embeddings, ano_aluno, top_k=5):
    """Busca, calcula similaridade e ranqueia o conteúdo."""
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

# --- Inicialização da Aplicação ---
# Verifica se a chave da API foi configurada nos segredos do Streamlit
try:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except (KeyError, FileNotFoundError):
    st.error("Chave da API da OpenAI não encontrada. Por favor, configure o arquivo .streamlit/secrets.toml")
    st.stop()

# Carrega os dados após a inicialização do session state
df, matriz_embeddings = carregar_dados()
if df is None:
    st.stop() # Interrompe a execução se os dados não puderem ser carregados

# --- Interface do Usuário (UI) ---
st.title("🤖 Tutor Inteligente de Matemática")
st.caption("Um assistente baseado no currículo de SC para te ajudar a estudar.")

with st.sidebar:
    st.header("Configurações")
    anos_disponiveis = sorted(df['Ano'].unique())
    
    # O valor do selectbox agora usa o que já está no session_state como padrão
    ano_selecionado = st.selectbox(
        "Qual ano você está cursando?",
        options=anos_disponiveis,
        index=anos_disponiveis.index(st.session_state.aluno_ano) # Garante que o índice corresponda ao ano
    )
    # Atualiza o session_state se o usuário mudar a seleção
    if ano_selecionado != st.session_state.aluno_ano:
        st.session_state.aluno_ano = ano_selecionado
        st.rerun() # Opcional: recarregar se a mudança de ano deve limpar o chat
        
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
                contexto_curricular = df_contexto.iloc[0]['texto_completo']
                fonte = f"Fonte: Currículo do {df_contexto.iloc[0]['Ano']}º ano - {df_contexto.iloc[0]['Objetos do conhecimento']}"
                log_to_terminal("Contexto selecionado para a API.")
                
                system_prompt = f"""
                Você é um tutor de matemática amigável, paciente e didático.
                Sua missão é ajudar um aluno do {st.session_state.aluno_ano}º ano.
                Use o seguinte CONTEXTO CURRICULAR para basear sua resposta. Não invente informações.
                Seja claro, use exemplos simples e sempre responda em português do Brasil.

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
                    response = st.write_stream(stream)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    log_to_terminal("Resposta da API recebida e exibida.")
                except Exception as e:
                    st.error(f"Ocorreu um erro com a API da OpenAI: {e}")
                    log_to_terminal(f"ERRO na API de Chat: {e}")
            else:
                st.write("Não consegui encontrar um conteúdo diretamente relacionado no currículo. Você pode tentar reformular a pergunta?")
                log_to_terminal("Nenhum contexto relevante encontrado.")
    st.rerun()
