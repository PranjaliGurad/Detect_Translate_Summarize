from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from langchain.chains import LLMChain, SequentialChain



spanish_email = """Estimado Jorge:

Espero se encuentre bien. Por medio de la presente, queremos indicarle que cuenta con una advertencia formal por retardo.

Usted llegó más de 15 minutos después de su hora de entrada en las siguientes fechas:

15 de mayo de 2022
23 de mayo de 2022
3 de junio de 2022
Con estos tres retardos, usted ahora cuenta con su primera advertencia formal por retardo. Le recordamos que tres advertencias formales por retardo podrían resultar en la terminación inmediata de su relación laboral con nosotros.

Le rogamos tomar las medidas necesarias para evitar que esto vuelva a suceder. Recuerde que contamos con el programa de empleado puntual del mes, donde el empleado que logre adherirse a su horario establecido con mayor puntualidad recibirá una tarjeta de regalo para su restaurante favorito.

Sin más por el momento, le agradezco por su atención.

Saludos,

Rodrigo Olivera Robles"""



def translate_and_summarize(email):
    
    llm = Ollama(model="llama2")
    
    # detect language....
    template1 = "Return the language of this email which is written in:\n{email}\n Only return the lanuage it was written "
    prompt1 = ChatPromptTemplate.from_template(template1)
    chain1 = LLMChain(llm = llm, prompt = prompt1, output_key = "language")
    
    # translate
    template2 = "Translate the email from(language) to English:\n"+email
    prompt2 = ChatPromptTemplate.from_template(template2)
    chain2 = LLMChain(llm = llm, prompt = prompt2, output_key = "Translated_email")
    
    # summary
    template3 = "Create a short summary of this email:\n{Translated_email}"
    prompt3 = ChatPromptTemplate.from_template(template3)
    chain3 = LLMChain(llm = llm, prompt = prompt3, output_key = "Summary")
    
    seq_chain = SequentialChain(chains =[chain1, chain2, chain3],
                                input_variables=["email"],
                                output_variables=["language", "Translated_email","Summary"],
                                verbose= True)
    
    return seq_chain(email)



result = translate_and_summarize(spanish_email)
print(result.key())
print(result["language"])              # gives you the language of documents
print(result["Translated_email"])      # translated the given document in to specififc document
print(result["Summary"])               # gives summary of tranmslated document