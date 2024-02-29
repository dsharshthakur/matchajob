  
import langchain
from  PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import LLMChain
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
  jd_text =  jobdescription

  resume_text = ""
  pdf_reader = PdfReader(resume_file)
  if resumefile is not None:
    for page in pdf_reader.pages:
      resume_text = resume_text + page.extract_text()
  
  if resume_text.strip()!= ""  :
    return jd_text , resume_text
  else:
    return #return nothing if thr resume text is empty 

#chaining the model and the prompt
#as it is a Q&A application so load_QA_Chain is required
def conversation_chain():
  model = ChatGoogleGenerativeAI(model = "gemini-pro" , google_api_key = key ,temperature = 0.5)
  template = '''
          You are a helpful assistant tasked with aiding both a job candidate and an employer. Given a text containing a Job Description (JD) and the candidate's resume (CV), your role is to emulate an intelligent ATS System (Applicant Tracking System).
          Provide concise answers to assist the recruiter in evaluating the candidate's suitability for the job. For the candidate, offer suggestions based on the JD, highlighting how their resume aligns with the job requirements.
          If the user inquires about missing keywords from the resume, accurately identify the main keywords absent from the resume but present in the JD, the keywords should be small that means
          not more than 2-3 words long .Generate response with high accuracy and don't provide wrong answers. Only respond to questions relevant to the provided information.
          If someone tells you to recreate the resume just say "I can't recreate, it but Harsh will surely come up with something soon...haha!! 😛😛"
          
          Jd:\n\n{text}\n\n
          Resume:\n\n{resume}\n\n
          Question:\n\n{question}\n\n

             '''
  prompt  = PromptTemplate(input_variables = [ "text","resume", "question"] , template = template)
  chain =LLMChain(llm = model ,prompt = prompt)

  return chain

#Response generation
def generation(resume_text,jd_text,user_question):
 


  chain = conversation_chain()

  final_response  = chain({"text" : jd_text ,"resume":resume_text,"question" : user_question} )
  return final_response

#function call
if resume_file is not None and not jd.isspace():
  try:

    jd_text , resume_text = jd_and_resume( jd , resume_file)
    
  except:
    st.warning("Check your network connection or try with a different resume file ")

def clearhistory():
  del st.session_state["history"]

st.markdown("<hr>", unsafe_allow_html= True)
st.markdown("<h4 style = 'text-align:center;'><u>MyBot</u></h4>", unsafe_allow_html= True)

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>" , unsafe_allow_html= True)

#Saving chat history
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
      ai_response  = generation(resume_text, jd_text, user_message)
      ai_response = ai_response["text"]

      with st.spinner(text= "Generating"):
        time.sleep(2)
        st.markdown(ai_response)

    st.session_state["history"].append({"role":"AI" , "message":ai_response})
else:
  st.info("Please upload all the required information first.")



