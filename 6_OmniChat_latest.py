import streamlit as st
import os
import io 
import base64
import time
import re
import weaviate
from youtube_transcript_api import YouTubeTranscriptApi
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
import numpy as np

from dotenv import load_dotenv
load_dotenv()


st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 0.8rem;
                    padding-bottom: 0rem;
                    # padding-left: 2rem;
                    # padding-right:2rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)


# Function to extract the video ID from the YouTube URL
def get_youtube_id(url):
    
    regex = (
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
    match = re.match(regex, url)
    return match.group(6) if match else None


def add_documents_with_retry(retriever, docs, retries=3):
    for attempt in range(retries):
        try:
            retriever.add_documents(docs)
            return  # Exit if successful
        except weaviate.exceptions.UnexpectedStatusCodeError as e:
            if attempt < retries - 1:  # If not the last attempt
                time.sleep(2)  # Wait before retrying
            else:
                print(f"Failed to add documents after {retries} attempts: {e}")



@st.cache_resource
def load_all_models():

    # Loading the LLM model
    model = ChatGroq(model="llama3-8b-8192", api_key=os.getenv('GROQ_API_KEY'))

    # Embedding Model
    embedding_model = HuggingFaceInferenceAPIEmbeddings(api_key=os.getenv('Hugg_Face_TOKEN'), model_name="BAAI/bge-base-en-v1.5")

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""])

    WEAVIATE_URL ="https://mraljvvcqhqzgnfv0hmsa.c0.us-west3.gcp.weaviate.cloud"

    client = weaviate.Client(
        url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(os.getenv('WEAVIATE_API_KEY')),
        additional_headers={
            "X-HuggingFace-Api-Key": os.getenv('Hugg_Face_TOKEN')
        },
    )

    # Deleting the existing schema
    client.schema.delete_class("RAG")

    # Defining the schema
    schema = {
        "classes": [
            {
                "class": "RAG",
                "description": "Documents for RAG",
                "vectorizer": "text2vec-huggingface",
                "moduleConfig": {"text2vec-huggingface": {"model": "sentence-transformers/all-MiniLM-L6-v2", "type": "text"}},
                "properties": [
                    {
                        "dataType": ["text"],
                        "description": "The content of the paragraph",
                        "moduleConfig": {
                            "text2vec-huggingface": {
                                "skip": False,
                                "vectorizePropertyName": False,
                            }
                        },
                        "name": "content",
                    },
                ],
            },
        ]
    }

    # Creating the schema
    client.schema.create(schema)

    # Defining the retriever
    retriever = WeaviateHybridSearchRetriever(
        alpha = 0.5,               # defaults to 0.5, which is equal weighting between keyword and semantic search
        client = client,           # keyword arguments to pass to the Weaviate client
        index_name = "RAG",  # The name of the index to use
        text_key = "content",         # The name of the text key to use
        attributes = [], # The attributes to return in the results
        create_schema_if_missing=True,
    ) 

    return model,embedding_model,text_splitter,retriever



def download_trasncript(text):
    """
    Creates a text file with the given summary text and offers download.
    """
    # Generate unique filename
    filename = f"summary_{int(time.time())}.txt"

    # Create the file and write the content
    with open(filename, "w") as file:
        file.write(text)

    # Set content_type and headers
    content_type = "text/plain"
    headers = {
        "Content-Disposition": f"attachment; filename={filename}",
        "Content-type": content_type,
    }

    # Use st.download_button to offer download
    st.download_button(
        "Download Video transcript📝",
        data=text,
        file_name=filename,
        mime=content_type,
        use_container_width=True,
    )



def process_pdf(pdf_file, text_splitter,retriever):
    """
    Processes a PDF file by extracting its text content and splitting it into documents.

    This function reads the PDF file, extracts the text from all its pages, and then uses a text splitter to divide the text into documents. The documents are then returned.

    Parameters:
    - pdf_file: The PDF file to be processed.
    - embedding_model: The embedding model to be used for document creation.
    - text_splitter: The text splitter to be used for dividing the text into documents.

    Returns:
    - A list of documents created from the PDF file's text content.
    """
    # Create a BytesIO object
    file_object = io.BytesIO(pdf_file.read())  
    reader = PdfReader(file_object)

    # Extract text from all pages
    pdf_text = ""
    for page in reader.pages:
        pdf_text += page.extract_text()

    # Creating the documents
    docs = text_splitter.create_documents(pdf_text.split("\n\n"))

    # Adding the docs to retriever
    add_documents_with_retry(retriever, docs)



def get_website_content(url:str,text_splitter,retriever):
    """
    This function retrieves the content of a website, processes it, and returns a list of documents.

    Parameters:
    - url: The URL of the website to be processed.
    - text_splitter: The text splitter to be used for dividing the text into documents.

    Returns:
    - A list of documents created from the website's content.
    """
    # Send a GET request to the webpage
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract the main content (text)
        # This site has the main text within <font> tags
        content = soup.find_all("font")

        # Combine and clean the extracted text
        full_text = "\n".join([element.get_text() for element in content])

        # Create documents from the full text
        docs = text_splitter.create_documents([full_text])
        
        # Adding the docs to retriever
        add_documents_with_retry(retriever, docs)


    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")




def process_youtube(youtube_id, text_splitter,retriever):
    """
    Processes a YouTube video by extracting its transcript and splitting it into documents.

    This function uses the YouTubeTranscriptApi to get the transcript of the YouTube video with the given id. It then creates a single string from all the captions in the transcript. Finally, it uses a text splitter to divide the string into documents, which are then returned.

    Parameters:
    - youtube_id: The id of the YouTube video to be processed.
    - embedding_model: The embedding model to be used for document creation.
    - text_splitter: The text splitter to be used for dividing the text into documents.

    Returns:
    - A list of documents created from the YouTube video's transcript.
    """
    result = YouTubeTranscriptApi.get_transcript(youtube_id)
    yt_captions = ""
    for item in iter(result):
        yt_captions = yt_captions + item['text'] + ""

    # Creating the chunks
    docs = text_splitter.create_documents([yt_captions])

    # Adding the docs to retriever
    add_documents_with_retry(retriever, docs)
    



def chat_with_utube():

    # Calling the function to load the Whisper model, LLM and embedding model
    model,embedding_model,text_splitter,retriever = load_all_models()
    
    pinecone_obj = None

    col1, col2 = st.columns(spec=(2,1), gap="large")
    with col1:
        st.markdown(
            "<h1 style='text-align: left; font-size: 48px;'>OmniChat 🤖</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size: 20px; text-align: left;'>Welcome to our advanced Media Query Module, a versatile tool for interacting with both YouTube videos and PDF documents. Upload any YouTube video URL or PDF file, and our AI will generate searchable transcripts and text extractions. These are stored in our vector database, linked to a Large Language Model (LLM), enabling detailed content queries. Retrieve insights effortlessly with our cutting-edge AI technology.</p>",
            unsafe_allow_html=True,
        )
    
        # Getting the input type from the user
        pdf_file = None
        youtube_link = None
        website_link = None
        with st.container(border=True):
            input_type = st.selectbox("Select the input type", ["Youtube", "PDF"])

            if input_type == "PDF":
                pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
                if not pdf_file:
                    st.error("Please upload a PDF file.")
                else:
                    process_pdf(pdf_file,text_splitter,retriever)
                    message = st.chat_message("assistant")
                    message.write("PDF is Processed Successfully, you can start with your query")
                    query_text = st.text_input("Enter your query : ")
                    if query_text:
                        vdb_context_text = retriever.invoke(query_text,score=True)[0].page_content

                        # Calling the function to get the answer from the LLM
                        response = model.invoke(f"Given the query '{query_text}', and after reviewing the information retrieved from the vector database: {vdb_context_text}, please provide a concise and informative answer.")
                        if response is not None:
                            with st.container(border=True):
                                st.markdown(f"<p style='font-size: 20px;'>{response.content}</p>",unsafe_allow_html=True)


            elif input_type == "Youtube":
                youtube_link = st.text_input("Enter the Youtube video URL")
                if not youtube_link:
                    st.error("Please enter the YouTube video URL.")
                else:
                    youtube_id = get_youtube_id(youtube_link)
                    process_youtube(youtube_id,text_splitter,retriever)
                    message = st.chat_message("assistant")
                    message.write("Video is Processed Succesfully, you can start with your query")
                    query_text = st.text_input("Enter your query : ")
                    if query_text:
                        vdb_context_text = retriever.invoke(query_text,score=True)[0].page_content

                        # Calling the function to get the answer from the LLM
                        response = model.invoke(f"Given the query '{query_text}', and after reviewing the information retrieved from the vector database: {vdb_context_text}, please provide a concise and informative answer.")
                        if response is not None:
                            with st.container(border=True):
                                st.markdown(f"<p style='font-size: 20px;'>{response.content}</p>",unsafe_allow_html=True)


                    with col2:
                        st.title("Uploaded Youtube Video")
                        st.success("Video Uploaded Succesfully!")
                        st.markdown(f"""
                                <iframe width="460" height="450" src="https://www.youtube.com/embed/{youtube_id}" 
                                frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                                allowfullscreen></iframe>
                            """, unsafe_allow_html=True)


         

chat_with_utube()