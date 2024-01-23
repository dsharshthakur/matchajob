
import langchain
from  PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import os
import streamlit as st
import time

#key 
key = st.secrets["PROJECT_KEY"]

with open("custom.css") as f:
  st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

#App title
st.markdown("<h2 style = 'text-align:center'>MatchMyCV</h2>" , unsafe_allow_html = True)


#sidebar
with st.sidebar:
  resume_file = st.file_uploader(label="Upload You Resume.", help = "The file should be in .pdf format")
    
    

  with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<br>" , unsafe_allow_html = True)
st.markdown("<br>" , unsafe_allow_html = True)
st.markdown("<br>" , unsafe_allow_html = True)
st.markdown("<br>" , unsafe_allow_html = True)

    
#job description
jd = st.text_area(label ="Copy & Paste the Job Description here..",value = " ", height= 220 )

#necessary functions
def jd_and_resume(jobdescription = jd ,resumefile = None):
  jd_text = "BELOW IS THE JOb DESCRIPTION:\n\n" + jobdescription

  resume_text = ""
  pdf_reader = PdfReader(resume_file)
  if resumefile is not None:
    for page in pdf_reader.pages:
      resume_text = resume_text + page.extract_text()
  
  if resume_text.strip()!= ""  :
    raw_text = jd_text + "\n\nBELOW IS THE RESUME OF THE CANDIDATE:\n\n " + resume_text
    return raw_text
  else:
    return #return nothing if thr resume text is empty 


#this function divides the big text into multiple small chunks
def create_chunks(text):
  splitter = RecursiveCharacterTextSplitter(chunk_size = 10000 , chunk_overlap = 1000)
  chunks = splitter.split_text( text = text)
  return chunks


#perform text embedding and store it in a vector db
def text_embedding(chunks):
  embedding = GoogleGenerativeAIEmbeddings(model = "models/embedding-001" ,google_api_key = key )
  vectordb = FAISS.from_texts(chunks , embedding = embedding)
  vectordb.save_local("faiss_index")


#chaining the model and the prompt
#as it is a Q&A application so load_QA_Chain is required
def conversation_chain():
  model = ChatGoogleGenerativeAI(model = "gemini-pro" , google_api_key = key ,temperature = 0.6)
  template = '''
          You are a helpfull assistant to a candidate looking for a job and employer both.
          You will be provided with a text which includes the job description posted on some website and also the
          resume of the candidate. So your job is it to act like a ATS System(Applicant Tracing System) and provide sumarized answers and help the recruiter to evaluate the resume of the candidate. And for the
          candidate you have to provide suggestions based on the job description that how his/her resume fits the requirements.When user asks you to provide missing keywords from the resume provide them the main word keywords instead of a sentence. 
          And don't provide wrong answer if the question is not relevant to the information in the database.
          Try to keep your answer as short as possible.
          If someone tells you to recreate the resume just say "I can't recreate, it but Harsh will surely come up with something soon...haha!! 😛😛"


          Text:\n\n{context}\n\n
          Question:\n\n{question}\n\n

             '''
  prompt  = PromptTemplate(input_variables = [ "context", "question"] , template = template)
  qa_chain = load_qa_chain(llm = model ,prompt = prompt , chain_type = "stuff")

  return qa_chain

#load vector db
#and perform similarity search baseed on the question given by the user
def generation(user_question):
  embedding = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key = key)
  load_vectordb = FAISS.load_local("faiss_index", embedding)
  similarity = load_vectordb.similarity_search(user_question)

  chain = conversation_chain()

  final_response  = chain({"input_documents" : similarity , "question" : user_question} )
  return final_response

if resume_file is not None and not jd.isspace():
  try:

    knowledge_text = jd_and_resume( jd , resume_file)
    chunks = create_chunks(knowledge_text)
    text_embedding(chunks)
  except:
    st.warning("Check your network connection or try with a different resume file ")

def clearhistory():
  del st.session_state["history"]

st.markdown("<hr>", unsafe_allow_html= True)
st.markdown("<h4 style = 'text-align:center;'><u>MyBot</u></h4>", unsafe_allow_html= True)

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>" , unsafe_allow_html= True)

#funcation call
if resume_file is not None and not jd.isspace():
  if "history" in st.session_state:
    col1, col2, col3,col4,col5= st.columns(5)
    with col5:
      clear_btn = st.button("Clear History" , on_click=clearhistory)


  if "history" not in st.session_state:
    st.session_state["history"] = []

  for msg in st.session_state["history"]:
    if msg["role"] == "user":
      avatar_img = "aiimg.jpg"
      with st.chat_message(msg["role"], avatar=avatar_img):
        st.markdown(f"<div class='chat-container user-message'>{msg['message']}</div>", unsafe_allow_html=True)
    else:
      with st.chat_message(msg["role"]):
        st.markdown(msg['message'], unsafe_allow_html = True)



  user_message = st.chat_input("Ask me anything.")
  if user_message:
    with st.chat_message("User", avatar = "aiimg.jpg"):
      st.markdown(f"<div class='chat-container user-message'>{user_message}</div>", unsafe_allow_html = True)
      st.session_state["history"].append({"role":"user" , "message":user_message})

    with st.chat_message("AI"):
      ai_response  = generation(user_message)
      ai_response = ai_response["output_text"]

      with st.spinner(text= "Generating"):
        time.sleep(5)
        st.markdown(ai_response)

    st.session_state["history"].append({"role":"AI" , "message":ai_response})
else:
  st.info("Please upload all the required information first.")



