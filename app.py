import os
import spacy
from wordcloud import WordCloud
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from spacy import displacy
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
dotenv_path = r'env'
load_dotenv(dotenv_path)

# Access environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
# function for generating wordcloud
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""


def get_wordcloud(text: str):
    wordcloud = WordCloud(
        width=800, height=600, background_color='black', min_font_size=10).generate(text)
    wordcloud.to_file('wordcloud.png')
    return 'wordcloud.png'


def extract_key_findings(text):
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    llm = ChatGroq(groq_api_key=GROQ_API_KEY,
                   model='Llama3-8b-8192', temperature=0.3)
    system = "First Generate Key Insights in maximum of 5 bullet points? Don't add your staff."
    human = "{text}"
    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", human)])

    chain = prompt | llm
    key_insights_summary = chain.invoke({"text": text}).content
    print(key_insights_summary)
    formatted_summary = "<ul>" + "<li>" + \
        "</li><li>".join(key_insights_summary.split("\n")[2:]) + "</li></ul>"
    return formatted_summary.replace("• ", "")


def ner(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    html = displacy.render(doc, style='ent', jupyter=False)
    html = html.replace("<br><br>", "<br>")
    st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)


st.set_page_config(layout="wide")
st.title('Llama3 powered text analytics app:')
with st.expander('About App'):
    st.markdown('This App is built with Llama3-70b, Stremlit and Spacy.')

# text to analyze goes here
text_input = st.text_area('Enter your text to Analyze:')
if text_input is not None:
    if st.button('Analyze'):
        st.markdown('**Input Text**')
        st.info(text_input)
        col_1, col_2, = st.columns([1, 2])
        with col_1:
            st.markdown('**Key Findings**')
            st.write(extract_key_findings(text_input), unsafe_allow_html=True)
        with col_2:
            st.markdown('**Words Summary**')
            st.image(get_wordcloud(text_input))

        st.markdown('**Named Entity Recognition**')
        ner(text_input)
else:
    st.error('please enter a message in text area???')
