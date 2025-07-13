# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
import logging
import json
import re

# --- 1. CONFIGURAÇÃO INICIAL E SESSION STATE ---
st.set_page_config(page_title="Tutor de Matemática", page_icon="🤖", layout="centered")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

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
if "df_com_similaridade" not in st.session_state:
    st.session_state.df_com_similaridade = None
if "topico_selecionado_idx" not in st.session_state:
    st.session_state.topico_selecionado_idx = None
if "initial_action_taken" not in st.session_state:
    st.session_state.initial_action_taken = False


# --- 2. DEFINIÇÃO DE TODAS AS FUNÇÕES AUXILIARES ---
def log_to_terminal(message):
    st.session_state.log_messages.append(str(message))
    logging.info(message)

def corrigir_formatacao_matematica(texto: str) -> str:
    """
    Aplica correções de formatação LaTeX em uma string para garantir a renderização correta.
    """
    texto = re.sub(r'extsqrt\(([^)]+)\)', r'$\\sqrt{\1}$', texto)
    texto = re.sub(r'(?<!\$)\b([A-Z])\(([^)]+)\)\b(?!\$)', r'$\1(\2)$', texto)
    pattern_frac = r'(?<!\$)\\(frac|rac)\{([^\}]+)\}\{([^\}]+)\}(?!\$)'
    def normalizar_e_delimitar(match):
        numerador = match.group(2)
        denominador = match.group(3)
        return f"$\\frac{{{numerador}}}{{{denominador}}}$"
    texto = re.sub(pattern_frac, normalizar_e_delimitar, texto)
    return texto

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

def calcular_similaridades_iniciais(query, df, matriz_embeddings, client):
    if not query: return None
    log_to_terminal("Calculando similaridades para a busca inicial...")
    embedding = gerar_embedding_query(query, client)
    if embedding is None: return None
    scores = cosine_similarity([embedding], matriz_embeddings)[0]
    df_com_scores = df.copy()
    df_com_scores['similaridade'] = scores
    log_to_terminal("Cálculo de similaridade inicial concluído.")
    return df_com_scores

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

def renderizar_mensagem(message):
    if message["role"] == "user":
        st.markdown(message["content"])
        return
    try:
        data = json.loads(message["content"])
        for line in data.get("response", []):
            st.markdown(corrigir_formatacao_matematica(line), unsafe_allow_html=True)
    except (json.JSONDecodeError, TypeError):
        st.markdown(corrigir_formatacao_matematica(message["content"]), unsafe_allow_html=True)

def criar_query_contextualizada(historico_mensagens: list, topico_atual: str) -> str:
    log_to_terminal("Criando query contextualizada para a busca...")
    mensagens_relevantes = historico_mensagens[-4:]
    contexto_str = " ".join([f"{msg['role']}: {msg['content']}" for msg in mensagens_relevantes])
    query_final = f"Contexto do tópico: {topico_atual}. Conversa recente: {contexto_str}"
    return query_final


# --- 3. INICIALIZAÇÃO DE OBJETOS GLOBAIS ---
try:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except (KeyError, FileNotFoundError):
    st.error("Chave da API da OpenAI não encontrada. Configure o arquivo .streamlit/secrets.toml")
    st.stop()

df, matriz_embeddings = carregar_dados()
if df is None: st.stop()


# --- 4. LÓGICA PRINCIPAL DA APLICAÇÃO (MÁQUINA DE ESTADOS) ---
st.title("🤖 Tutor Inteligente de Matemática")

# ESTADO 1: COLETA DE INFORMAÇÕES
if st.session_state.app_state == "COLETA_INFO":
    st.info("👋 Olá! Para começarmos, diga seu nome, o ano que você está cursando e o assunto que gostaria de estudar hoje.")
    st.caption("Exemplo: 'Meu nome é Ana, sou do 8º ano e quero aprender sobre o teorema de Pitágoras.'")

    if prompt := st.chat_input("Diga seu nome, ano e assunto..."):
        dados_iniciais = extrair_dados_iniciais(prompt, client)
        if dados_iniciais:
            st.session_state.aluno_nome = dados_iniciais.get("nome", "estudante")
            st.session_state.aluno_ano = dados_iniciais.get("ano", 8)
            assunto = dados_iniciais.get("assunto", prompt)
            df_com_similaridade = calcular_similaridades_iniciais(assunto, df, matriz_embeddings, client)
            if df_com_similaridade is not None:
                st.session_state.df_com_similaridade = df_com_similaridade
                st.session_state.app_state = "SELECAO_TOPICO"
                st.rerun()

# ESTADO 2: SELEÇÃO DE TÓPICO
elif st.session_state.app_state == "SELECAO_TOPICO":
    st.markdown(f"### Olá, {st.session_state.aluno_nome}!")
    st.markdown(f"Com base no que você pediu, encontrei estes tópicos do **{st.session_state.aluno_ano}º ano**, ordenados por relevância. Qual deles você gostaria de estudar?")
    
    df_filtrado = st.session_state.df_com_similaridade
    playlist_df = df_filtrado[df_filtrado["Ano"] == st.session_state.aluno_ano].sort_values("similaridade", ascending=False)
    
    top_hit_idx = playlist_df.index[0] if not playlist_df.empty else None

    if top_hit_idx is None:
        st.warning("Não encontrei nenhum tópico correspondente no currículo do seu ano. Tente uma busca diferente na tela inicial.")
        if st.button("Voltar ao início"):
            st.session_state.app_state = "COLETA_INFO"
            st.rerun()
    else:
        for idx, row in playlist_df.head(5).iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{row['Objetos do conhecimento']}**")
                    st.caption(f"Unidade: {row['Unidade Temática']} | Similaridade: {row['similaridade']:.2f}")
                    if idx == top_hit_idx:
                        st.success("✨ Mais relevante")
                with col2:
                    if st.button("Estudar este", key=f"topic_{idx}"):
                        st.session_state.topico_selecionado_idx = idx
                        st.session_state.app_state = "CHAT"
                        st.session_state.initial_action_taken = False
                        st.session_state.messages = [
                            {"role": "assistant", "content": json.dumps({
                                "response": [f"Ok! Vamos focar em **{df.loc[idx, 'Objetos do conhecimento']}**. O que você gostaria de saber? Me peça uma explicação, exemplos ou exercícios!"]
                            })}
                        ]
                        st.rerun()

# ESTADO 3: CHAT
elif st.session_state.app_state == "CHAT":
    if st.button("⬅️ Mudar de Tópico"):
        st.session_state.messages = []
        st.session_state.app_state = "SELECAO_TOPICO"
        st.session_state.df_com_similaridade = None
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            renderizar_mensagem(message)
    
    prompt_gerado = None
    if not st.session_state.get("initial_action_taken", False):
        st.write("Escolha uma ação:")
        col1, col2, col3 = st.columns(3)
        if col1.button("Explique o tópico", use_container_width=True):
            prompt_gerado = "Por favor, me dê uma explicação detalhada sobre este tópico."
        if col2.button("Me dê um exemplo", use_container_width=True):
            prompt_gerado = "Pode me dar um exemplo prático sobre isso?"
        if col3.button("Quero exercícios", use_container_width=True):
            prompt_gerado = "Gostaria de alguns exercícios para praticar."
        if prompt_gerado:
            st.session_state.initial_action_taken = True
    else:
        prompt_gerado = st.chat_input("Faça outra pergunta ou peça mais exercícios!")

    if prompt_gerado:
        st.session_state.messages.append({"role": "user", "content": prompt_gerado})
        with st.chat_message("user"):
            st.markdown(prompt_gerado)
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                st.session_state.log_messages = []
                log_to_terminal("--- NOVA QUERY (CHAT) ---")
                
                topico_selecionado = df.loc[st.session_state.topico_selecionado_idx]
                nome_topico = topico_selecionado['Objetos do conhecimento']
                contexto_curricular = topico_selecionado['texto_completo']

                prompt_template = """
                Você é um tutor de matemática especialista. Sua tarefa é fornecer explicações claras e precisas.

                **REGRAS DE FORMATAÇÃO ESTRITAS:**
                1.  **JSON OBRIGATÁRIO:** Sua resposta DEVE ser um objeto JSON válido com uma chave "response" contendo uma lista de strings.
                2.  **LATEX PARA TUDO:** SEMPRE use a sintaxe LaTeX para TODA e QUALQUER notação matemática. Envolva as fórmulas com delimitadores.
                    - Use $ ... $ para matemática em linha (no meio de uma frase).
                    - Use $$ ... $$ para equações de destaque (em sua própria linha).
                3.  **COMANDOS PADRÃO:** Utilize apenas comandos LaTeX padrão. NUNCA invente comandos como 'extsqrt'.

                **MINI-DICIONÁRIO DE LATEX:**
                - Raiz quadrada: \\sqrt{{...}} (Ex: $\\sqrt{{b^2 - 4ac}}$)
                - Fração: \\frac{{numerador}}{{denominador}} (Ex: $\\frac{{1}}{{2}}$)
                - Expoente: ^ (Ex: $x^2$)
                - Subscrito: _ (Ex: $x_1$)
                - Pontos de coordenadas: $A(x_1, y_1)$

                **EXEMPLO OBRIGATÓRIO:**
                - **RUIM:** A distância d é extsqrt((x2-x1)^2).
                - **BOM:** A distância $d$ é calculada com a fórmula $\\sqrt{{(x_2 - x_1)}^2}$.

                O aluno está estudando o tópico: "{topic_name}".
                Use o CONTEXTO CURRICULAR abaixo para responder, seguindo ESTRITAMENTE todas as regras acima.
                CONTEXTO CURRICULAR: {curriculum_context}
                """
                
                system_prompt = prompt_template.format(
                    topic_name=nome_topico,
                    curriculum_context=contexto_curricular
                )
                
                mensagens_para_api = [{"role": "system", "content": system_prompt}] + st.session_state.messages
                
                log_to_terminal("Enviando requisição para API (modo JSON)...")
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini", messages=mensagens_para_api,
                        response_format={"type": "json_object"}
                    )
                    resposta_json_str = response.choices[0].message.content
                    st.session_state.messages.append({"role": "assistant", "content": resposta_json_str})
                    log_to_terminal("\n--- RESPOSTA BRUTA DA API (JSON) ---")
                    log_to_terminal(resposta_json_str)
                except Exception as e:
                    st.error(f"Ocorreu um erro com a API da OpenAI: {e}")
                    log_to_terminal(f"ERRO na API de Chat: {e}")
        st.rerun()

# --- 5. SIDEBAR (SEMPRE VISÍVEL) ---
with st.sidebar:
    st.header("Debug")
    with st.expander("🔌 Terminal de Debug", expanded=True):
        log_container = st.container(height=300)
        log_text = "\n".join(st.session_state.log_messages)
        log_container.text(log_text)
