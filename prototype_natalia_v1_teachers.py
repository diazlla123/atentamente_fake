import streamlit as st

# MUST BE FIRST
st.set_page_config(page_title="Study bot", page_icon="📖")

import os
import sys
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.output_parsers.json import SimpleJsonOutputParser
from langsmith import Client
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from streamlit_feedback import streamlit_feedback
import streamlit.components.v1 as components
from functools import partial
from llm_config_espanol import LLMConfig

# === Load environment variables from Streamlit secrets ===
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
os.environ["LANGSMITH_ENDPOINT"] = st.secrets["LANGCHAIN_ENDPOINT"]

# === Load Custom CSS ===
def load_custom_css(path="style.css"):
    with open(path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_custom_css()

# === Parse config file path ===
input_args = sys.argv[1:]
config_file = input_args[0] if input_args else st.secrets.get("CONFIG_FILE", "config_natalia_v0.1_teachers.toml")
print(f"Configuring app using {config_file}...\n")

# === Load prompt config from TOML ===
llm_prompts = LLMConfig(config_file)

# === Debug switch ===
DEBUG = False

# === LangSmith Client ===
smith_client = Client()

# === Initialize session state variables ===
if 'run_id' not in st.session_state: 
    st.session_state['run_id'] = None
if 'agentState' not in st.session_state: 
    st.session_state['agentState'] = "start"
if 'consent' not in st.session_state: 
    st.session_state['consent'] = False
if 'exp_data' not in st.session_state: 
    st.session_state['exp_data'] = True
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = "gpt-4o"
if 'intro_shown' not in st.session_state:
    st.session_state.intro_shown = False
if 'final_response' not in st.session_state:
    st.session_state.final_response = None
if 'waiting_for_listo' not in st.session_state:
    st.session_state['waiting_for_listo'] = True

# === Load OpenAI API key ===
openai_api_key = st.secrets.get("OPENAI_API_KEY")
if not openai_api_key:
    openai_api_key = st.sidebar.text_input("🔑 Ingresa tu OpenAI API Key", type="password")

if not openai_api_key:
    st.info("Por favor ingresa una OpenAI API Key para continuar.")
    st.stop()

# === Setup LangChain Memory ===
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)

# === Adjust prompt for gpt-4o ===
if st.session_state.llm_model == "gpt-4o":
    prompt_datacollection = llm_prompts.questions_prompt_template

# === Set up LLM instance ===
chat = ChatOpenAI(temperature=0.3, model=st.session_state.llm_model, openai_api_key=openai_api_key)

# === ConversationChain setup ===
prompt_updated = PromptTemplate(input_variables=["history", "input"], template=prompt_datacollection)
conversation = ConversationChain(prompt=prompt_updated, llm=chat, verbose=True, memory=memory)

# === UI and Conversation flow ===
if st.session_state['consent']:
    entry_messages = st.expander("Collecting your story", expanded=st.session_state['exp_data'])

    # Show initial AI message if chat just started
    if not st.session_state.intro_shown:
        msgs.add_ai_message(llm_prompts.questions_intro)
        with entry_messages:
            st.chat_message("ai").markdown(
                f"<span style='color:black'>{llm_prompts.questions_intro}</span>",
                unsafe_allow_html=True)
        st.session_state.intro_shown = True

    prompt = st.chat_input()

    if prompt:
        with entry_messages:
            st.chat_message("human").write(prompt)

            if st.session_state['waiting_for_listo']:
                if prompt.strip().lower() == "listo":
                    st.session_state['waiting_for_listo'] = False
                    response = conversation.invoke(input=prompt)
                    st.chat_message("ai").markdown(
                        f"<span style='color:black'>{response['response']}</span>",
                        unsafe_allow_html=True
                )
                else:
                    st.warning("🔒 Para comenzar, por favor escribe la palabra **\"listo\"**.")
            else:
                response = conversation.invoke(input=prompt)
                if "FINISHED" in response['response']:
                    refinement_prompt = PromptTemplate(
                        input_variables=["text"],
                        template="""
    Toma el siguiente texto y reescríbelo de forma clara, directa y estructurada para que la persona pueda ver reflejada su propia experiencia. 
    La finalidad de esta sección es despertar dentro de la persona una introspección sobre su situación y el cómo es que la maneja.
    La micronarrativa debe estar basada en los inputs de la persona, siendo objetiva y concisa, con una postura neutra.
    Haz este párrafo tener una longitud mínima de 3 oraciones y máxima de un párrafo.

    {text}
    """
                    )
                    refined_chain = refinement_prompt | chat
                    refined = refined_chain.invoke({"text": response['response']})

                    st.divider()
                    st.chat_message("ai").write(llm_prompts.questions_outro)
                    st.session_state.agentState = "summarise"
                    st.session_state.final_response = refined.content
                    st.rerun()
                else:
                    st.chat_message("ai").write(response['response'])

    # Show editable final summary if collected
    if st.session_state.agentState == "summarise" and st.session_state.final_response:
        st.subheader("Tu historia en tus propias palabras")
        st.markdown("Aquí tienes la versión final de tu narrativa. Si quieres mejorarla o adaptarla, puedes hacerlo a continuación:")

        new_text = st.text_area("✍️ Edita tu micronarrativa si lo deseas", value=st.session_state.final_response, height=250, label_visibility="visible")

        if st.button("✅ Guardar versión final"):
            st.success("Tu narrativa ha sido actualizada.")
            st.session_state.final_response = new_text

        with st.container():
            st.markdown("### ✨ ¿Quieres mejorar tu narrativa con ayuda de la IA?")
            with st.expander("🔧 Haz clic aquí para adaptar tu texto con la IA", expanded=True):
                st.chat_message("ai").write("¿Qué podríamos mejorar o cambiar en tu narrativa?")
                adaptation_input = st.chat_input("Escribe cómo quieres mejorarla...")
                if adaptation_input:
                    st.chat_message("human").write(adaptation_input)
                    adaptation_prompt = PromptTemplate(input_variables=["input", "scenario"], template=llm_prompts.extraction_adaptation_prompt_template)
                    json_parser = SimpleJsonOutputParser()
                    chain = adaptation_prompt | chat | json_parser
                    with st.spinner('Generando versión mejorada...'):
                        improved = chain.invoke({"scenario": st.session_state.final_response, "input": adaptation_input})
                    st.markdown(f"Versión adaptada sugerida: \n {improved['new_scenario']}")
                    if st.button("✅ Usar versión sugerida"):
                        st.session_state.final_response = improved['new_scenario']
                        st.success("Narrativa actualizada con la sugerencia de IA.")
            # ✅ Mostrar botón de encuesta solo si el usuario llegó a la narrativa final
            st.markdown("---")
            st.markdown("### 📝 ¿Nos ayudas con tu opinión?")
            st.markdown(
    """
    <a href="https://forms.gle/pxBtvu8WPRAort7b7" target="_blank">
        <button style="background-color:#ec6041; color:white; padding:0.75rem 1.5rem; border:none; border-radius:12px; font-size:16px; cursor:pointer;">
            Ir a encuesta de retroalimentación
        </button>
    </a>
    """,
    unsafe_allow_html=True
)
else:
    consent_message = st.container()
    with consent_message:
        st.markdown(llm_prompts.intro_and_consent)
        st.button("He leído, entiendo", key="consent_button", on_click=lambda: st.session_state.update({"consent": True}))
