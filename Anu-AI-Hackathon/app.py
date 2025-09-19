import streamlit as st

from vector_db_reader import get_qa_chain

st.title("Query for Data Quality Issues")

chain = get_qa_chain()

question = st.text_input("Ask a question about data quality report")

if question:
    with st.spinner("Searching and thinking..."):
        result = chain({"query": question})
        st.markdown("### 📄 Answer")
        st.write(result["result"])

        st.markdown("### 🔍 Source Chunks")

        if result and "source_documents" in result:
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.code(doc.page_content.strip())
