import pandas as pd
import httpx
import re
import os
import certifi
import argparse

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.retrievers import MergerRetriever

from vector_db_loader import laod_csv_to_db, view_vector_info

client = httpx.Client(verify=False)

analyzer_llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model = "azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key="sk-zxaFsmvfSDtF1B4UIdXWRw", 
    http_client = client
)

# Embedding model
embedding_model = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key="sk-zxaFsmvfSDtF1B4UIdXWRw",
    http_client=client
)

# LLM for sentiment analysis
sentiment_llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-gpt-35-turbo",  # lighter model for quick tasks
    api_key="sk-O74aRghWm10g17BifcKjgA",
    http_client=client
)


def read_csv_file(csv_path, max_rows=10):
    df = pd.read_csv(csv_path)
    return df.head(max_rows).to_csv(index=False)

def beautify(text):
    df = pd.DataFrame(text)
    print(df)

def run_agent(data_sample_path, report_path):
    
    data_sample = read_csv_file(data_sample_path)
    dq_report = read_csv_file(report_path)

    prompt_template = """
    You are a data quality expert.

    Below is a sample of a dataset (CSV format):
    {data_sample}

    Analyze the issues and suggest:
    1. The key data quality problems
    2. Probable causes
    3. Suggested high-level fixes for each issue
    4. Step wise detailed fixes for each issue
    5. Convert the anamalies into a CSV report format and as answer return only the csv content
    """

    #And here is a data quality report describing the issues:
    #{dq_report}

    prompt = PromptTemplate(
        input_variables=["data_sample"],
        template=prompt_template,
    )
  
    #print(f"From LLM --- {llm.invoke("Hi")}")
    chain = prompt | analyzer_llm

    response = chain.invoke({"data_sample": data_sample, "dq_report": dq_report})

    print(f"Result is : {response.content}")

    #beautify(response)
    return response.content


def write_to_csv(text):

    #csv_content = re.search(r"csv(.*?)", data, re.DOTALL).group(1).strip()

    start = text.find("csv") + len("csv")
    end = text.find("```", start)
    csv_content = text[start:end].strip()

    csv_file_name = "data_quality_report.csv"

    #print(f"Writing the info -- {csv_content}")
    with open(csv_file_name, "w", encoding="utf-8") as file:
        file.write(csv_content)

    print("CSV file written successfully!")

    return csv_content, csv_file_name


def save_to_vector(anamolies_data_csv, user_query):
    persist_dir = "./chroma_index"
    os.makedirs(persist_dir, exist_ok=True)

    os.environ["SSL_CERT_FILE"] = r"C:\\Program Files\\Python312\\Lib\\site-packages\\certifi\\cacert.pem"
    os.environ["TIKTOKEN_CACHE_DIR"] = "./token"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(anamolies_data_csv)

    doc_persist_dir = os.path.join(persist_dir, "data_quality_issues")
    os.makedirs(doc_persist_dir, exist_ok=True)

    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    os.environ["SSL_CERT_FILE"] = certifi.where()

    vectordb = Chroma.from_texts(chunks, embedding_model, persist_directory=doc_persist_dir)
    vectordb.persist()

    #Read from vector

    retrievers = [vectordb.as_retriever()]
    retriever = MergerRetriever(retrievers=retrievers)
    rag_chain = RetrievalQA.from_chain_type(llm=analyzer_llm, retriever=retriever, return_source_documents=False)

    #user_query = "What is the resolution for age not valid issue?"

    result = rag_chain.invoke(user_query)
    answer = result.get("result") if isinstance(result, dict) else None

    print(f"Answer is -- {answer}")





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--user_query", help="Path to CSV")

    args = parser.parse_args()

    query = args.user_query

    data_file = "employees_with_anomalies_100.csv"
    dq_report_file = "employee_data_quality_report_100.csv"

    anomalies_data = run_agent(data_file, dq_report_file)

    anomalies_data_csv, csv_file_name= write_to_csv(anomalies_data)

    save_to_vector(anomalies_data_csv, query)

    #laod_csv_to_db(csv_file_name, "sk-zxaFsmvfSDtF1B4UIdXWRw")

    #view_vector_info()

   


    

