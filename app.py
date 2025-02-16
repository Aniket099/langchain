import streamlit as st
from sagemaker_setup import extract_text_from_pdf, summarize_large_text  # Import functions

st.title("📄 AI-Powered Large PDF Summarization")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        extracted_text = extract_text_from_pdf(uploaded_file)
    
    st.subheader("Extracted Text Preview")
    st.text_area("Extracted Text:", extracted_text[:1000] + "...", height=150, disabled=True)

    if st.button("Summarize"):
        with st.spinner("Summarizing large PDF..."):
            summary = summarize_large_text(extracted_text)
        st.success("Summary:")
        st.write(summary)
