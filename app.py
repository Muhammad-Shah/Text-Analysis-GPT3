import os
import spacy
from wordcloud import WordCloud
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from spacy import displacy
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
dotenv_path = r'D:\Transcripts_and_News_Articles_Generation\env'
load_dotenv(dotenv_path)

# Access environment variables
GROQ_API = os.getenv('GROQ_API')

# function for generating wordcloud


def get_wordcloud(text: str):
    wordcloud = WordCloud(
        width=800, height=800, background_color='black', min_font_size=10).generate(text)
    wordcloud.to_file('wordcloud.png')
    return wordcloud


def extract_key_findings(text):
    print('testing----------------??')
    llm = ChatGroq(groq_api_key=GROQ_API, model='Llama3-8b-8192')
    system = "First Generate Key Insights in maximum of 5 bullet points? Second Generate summary from the below Text? Don't add your staff."
    human = "{text}"
    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", human)])

    chain = prompt | llm
    key_insights_summary = chain.invoke({"text": text}).content
    return key_insights_summary


def ner(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    html = displacy.render(doc, style='ent', jupyter=True)
    return html


st.set_page_config(layout="wide")
st.title('GPT3 powered text analytics app:')
with st.expander('About App'):
    st.markdown('This App is built with Llama3-70b, Stremlit and Spacy.')

# text to analyze goes here
text_input = st.text_input('Enter your text to Analyze:')
if text_input is not None:
    if st.button('Analyze'):
        st.markdown('**Input Text**')
        st.info(text_input)
        col_1, col_2, = st.columns([1, 2])
        with col_1:
            st.markdown('**Key Findings**')
            st.success(extract_key_findings(text_input))
        with col_1:
            st.markdown('**Words Summary**')
            st.image(get_wordcloud(text_input))

        st.markdown('**Named Entity Recognition**')
        ner(text_input)
