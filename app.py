import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re

# --- Configuração da Página e Logger ---
st.set_page_config(
    page_title="Tutor de Matemática",
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

# --- FUNÇÃO DE CORREÇÃO DE LATEX (VERSÃO 3.0 - CIRÚRGICA) ---
def corrigir_notacao_latex(texto: str) -> str:
    """Aplica uma série de correções para garantir a renderização LaTeX."""

    # Etapa 1: Normalizar delimitadores inconsistentes (ex: $$...$$$)
    texto = texto.replace('$$$', '$$')

    # Etapa 2: Garantir espaço antes e depois de blocos $$
    # Isso ajuda a evitar textos colados como "palavra$$\frac{a}{b}$$outra"
    texto = re.sub(r'([^\s])(\$\$)', r'\1 \2', texto) # Adiciona espaço antes de $$
    texto = re.sub(r'(\$\$[^\$]+\$\$)([^\s])', r'\1 \2', texto) # Adiciona espaço depois de $$

    # Etapa 3: Corrigir frações que não foram delimitadas
    # Padrão: Encontra '\frac{...}{...}' que NÃO é precedido por um '$'
    pattern_frac = r'(?<!\$)\\frac\{[^\}]+\}\{[^\}]+\}'
    texto = re.sub(pattern_frac, lambda match: f"${match.group(0)}$", texto)

    # Etapa 4 (Opcional, mas útil): Corrigir outros comandos comuns sem delimitador
    pattern_cmd = r'(?<![\$a-zA-Z])(\\(?:sqrt|cdot|times|div|pi|alpha|beta)\{[^\}]+\})'
    texto = re.sub(pattern_cmd, lambda match: f"${match.group(0)}$", texto)

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
        return None, None

def gerar_embedding_query(texto, client):
    log_to_terminal(f"Gerando embedding para a query: '{texto[:30]}...'")
    try:
        response = client.embeddings.create(input=[texto], model="text-embedding-3-large")
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Erro ao gerar embedding: {e}")
        return None

def buscar_conteudo_relevante(query_embedding, df, matriz_embeddings, ano_aluno, top_k=5):
    if query_embedding is None: return pd.DataFrame()
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
    st.error("Chave da API da OpenAI não encontrada. Configure o arquivo .streamlit/secrets.toml")
    st.stop()

df, matriz_embeddings = carregar_dados()
if df is None: st.stop()

st.title("🤖 Tutor Inteligente de Matemática")
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
        # A renderização do histórico usa st.markdown, que lida bem com strings que já estão corretas
        st.markdown(message["content"], unsafe_allow_html=True)

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

                log_to_terminal("\n--- CONTEXTO SELECIONADO PARA O LLM ---")
                log_to_terminal(f"Índice: {contexto_row.name}, Ano: {contexto_row['Ano']}, Score: {contexto_row['similaridade']:.4f}")
                log_to_terminal("---------------------------------------\n" + contexto_curricular + "\n---------------------------------------\n")

                # --- PROMPT DO SISTEMA (VERSÃO 3.0) ---
                system_prompt = f"""
                Você é um tutor de matemática amigável e didático para um aluno do {st.session_state.aluno_ano}º ano.
                Baseie sua resposta no CONTEXTO CURRICULAR fornecido.
                
                REGRAS RÍGIDAS DE FORMATAÇÃO:
                1.  Blocos matemáticos (equações, frações grandes) devem estar em linhas separadas e entre dois cifrões ($$).
                2.  Fórmulas ou variáveis pequenas no meio do texto devem estar entre um cifrão ($).
                3.  NUNCA misture os formatos.
                4.  SEMPRE adicione um espaço antes e depois de qualquer bloco matemático ($ ou $$) para não colar com o texto. Exemplo CORRETO: `O resultado é $$\frac{{a}}{{b}}$$.` Exemplo ERRADO: `O resultado é$$\frac{{a}}{{b}}$$.`

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
                    
                    # --- LÓGICA DE RENDERIZAÇÃO SEGURA ---
                    # Para evitar o problema de "escape" do Python, escrevemos o stream
                    # em um placeholder, mas a versão final corrigida é salva no histórico
                    # e exibida corretamente no st.markdown() após o loop.
                    resposta_completa = st.write_stream(stream)
                    
                    # Usa a nova função cirúrgica para limpar e formatar a resposta
                    resposta_corrigida = corrigir_notacao_latex(resposta_completa)
                    
                    st.session_state.messages.append({"role": "assistant", "content": resposta_corrigida})
                    log_to_terminal("Resposta da API recebida.")
                    if resposta_completa != resposta_corrigida:
                        log_to_terminal("Notação LaTeX foi corrigida programaticamente.")

                except Exception as e:
                    st.error(f"Ocorreu um erro com a API da OpenAI: {e}")
                    log_to_terminal(f"ERRO na API de Chat: {e}")
            else:
                st.write("Não consegui encontrar um conteúdo diretamente relacionado no currículo. Você pode tentar reformular a pergunta?")
                log_to_terminal("Nenhum contexto relevante encontrado.")
    
    # Recarrega a página para exibir a mensagem do assistente que foi adicionada ao histórico
    st.rerun()
