from functions import *
from datetime import datetime
import streamlit as st
import io
from PyPDF2 import PdfReader

@st.cache(suppress_st_warning=True)
def process_uploaded_file(resources):
    start = datetime.now()
    try:
        # Create a byte stream from the uploaded file
        resource_bytes = io.BytesIO(resources.read())
        # Create a PdfFileReader object
        pdf_reader = PdfReader(resource_bytes)
        # Get the number of pages in the PDF
        num_pages = len(pdf_reader.pages)
        # Extract the text from each page and add it to the 'text' variable
        text = ""
        for page_num in range(num_pages):
            text += pdf_reader.pages[page_num].extract_text()
        
        text = preprocess_data(text)
        entities = extract_entities(text)
        linked_entities = link_entities(entities)
        st.sidebar.write("Extracted Entities :")
        st.sidebar.write(linked_entities)
        now = datetime.now()
        st.sidebar.success("File uploaded! Extraction completed in {} seconds".format((now - start).seconds))
        print(text)

        return text

    except Exception as e: 
        print(e)
        st.warning("An error occured while reading the file")
        return

def main():
    st.set_page_config(page_title="Company Query Engine", page_icon=":guardsman:", layout="wide")
    st.title("Company Query Engine")
    st.sidebar.title("Actions")
    st.markdown("A streamlit application that uses GPT-3 to answer queries about a company based on the company's resources.")

    # sidebar: Upload company resources
    resources = st.sidebar.file_uploader("Upload company resources", type=["txt", "pdf"])

    # sidebar: Extract structured data
    if resources:
        text = process_uploaded_file(resources)
        st.text(text)

    else:
        st.warning("Please upload company resources first")
        return

    query = st.text_input("Ask a question")
    if st.button('Submit'):
        answer = fine_tune_gpt3(query, text)
        st.success(answer)

if __name__ == "__main__":
    main()
