import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import os
import numpy as np
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Helper Functions ---
def load_chroma_collection():
    """Initialize ChromaDB client and load collection."""
    store_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "vector_store"))
    if not os.path.exists(store_dir):
        st.error("ChromaDB store not found. Please run store_to_vector_db.py first.")
        return None

    try:
        openai_key = os.getenv("OPENAI_API_KEY")
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_key,
            model_name="text-embedding-3-small"
        )
        client = chromadb.PersistentClient(path=store_dir)
        collection = client.get_collection(name="employee_issues", embedding_function=openai_ef)
        return collection
    except Exception as e:
        st.error(f"Could not load ChromaDB collection: {e}")
        return None

def query_chroma(user_query, collection):
    """Query ChromaDB collection and return the best match's resolution."""
    try:
        results = collection.query(
            query_texts=[user_query],
            n_results=1  # only top result for CSV processing
        )
        if results and results.get("metadatas"):
            meta = results["metadatas"][0][0]
            return meta.get("resolution", "No resolution found")
        return "No resolution found"
    except Exception as e:
        return f"Error querying: {e}"

# --- Streamlit UI ---
st.set_page_config(page_title="Employee Issue Resolver", page_icon="🧠")
st.title("🧠 Data Quality-Resolution Assistant")
st.write("Enter an issue description manually OR upload a CSV to generate a resolved report.")

# Load ChromaDB Collection
collection = load_chroma_collection()

if collection:
    # --- Section 1: Manual Query ---
    st.subheader("🔍 Manual Query")
    user_query = st.text_input("Describe the issue (e.g., 'negative salary value'):")

    if st.button("Find Resolution"):
        if user_query.strip():
            res = query_chroma(user_query, collection)
            st.success(f"**Resolution:** {res}")
        else:
            st.warning("Please enter a query.")

    st.markdown("---")

    # --- Section 2: CSV Upload ---
    st.subheader("📂 Upload CSV for Bulk Resolution")
    uploaded_file = st.file_uploader("Upload a CSV with anomalies", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("📊 Preview of Uploaded File:", df.head())

            # Select which column contains the anomaly descriptions
            text_columns = [col for col in df.columns if df[col].dtype == "object"]
            if not text_columns:
                st.error("No text columns found in CSV to use for matching.")
            else:
                selected_col = st.selectbox("Select the column containing issue descriptions:", text_columns)

                if st.button("Generate Resolved Report"):
                    df["Resolution"] = df[selected_col].apply(lambda x: query_chroma(str(x), collection))
                    st.success("✅ Resolution report generated!")

                    st.write(df.head())

                    # Create downloadable CSV
                    output = BytesIO()
                    df.to_csv(output, index=False)
                    output.seek(0)
                    st.download_button(
                        label="⬇️ Download Resolved Report",
                        data=output,
                        file_name="resolved_report.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Failed to process CSV: {e}")
else:
    st.stop()
