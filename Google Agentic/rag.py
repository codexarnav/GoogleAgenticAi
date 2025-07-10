from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from prophet import Prophet
import pandas as pd
from datetime import datetime, timedelta
from dateutil.parser import parse
import json
import os

from prophet.plot import plot_plotly
import plotly.io as pio



from langchain.schema import Document

def load_json_as_documents(directory):
    documents = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        doc = Document(page_content=content, metadata={"source": file_path})
                        documents.append(doc)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return documents

json_docs = load_json_as_documents("data")


pdf_docs = DirectoryLoader(
    "data",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader
).load()

all_docs = json_docs + pdf_docs


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(all_docs)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(texts, embeddings)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.1,
    api_key='AIzaSyBlVAGaEchXF46DbcNFE66X6HbbcN4oSXI' 
)


retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={'k': 5})


def forecast_series(data, periods=6):
    df = pd.DataFrame(data)
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = pd.to_numeric(df['y'])

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)
    return model, forecast


def extract_net_worth_series(json_obj):
    now = datetime.today()
    values = []
    try:
        net_worth = int(json_obj["netWorthResponse"]["totalNetWorthValue"]["units"])
        for i in range(6):
            date = (now - timedelta(days=30 * i)).strftime('%Y-%m-%d')
            simulated_value = net_worth - i * 15000
            values.append({"ds": date, "y": simulated_value})
        values.reverse()
        return values
    except:
        return []

def extract_epf_series(json_obj):
    series = []
    try:
        est_details = json_obj["uanAccounts"][0]["rawDetails"]["est_details"]
        for est in est_details:
            doj = datetime.strptime(est["doj_epf"], "%d-%m-%Y").strftime("%Y-%m-%d")
            balance = int(est["pf_balance"]["net_balance"])
            series.append({"ds": doj, "y": balance})
        return sorted(series, key=lambda x: x["ds"])
    except:
        return []

def extract_mutual_fund_series(json_obj):
    series = []
    try:
        for txn in json_obj["transactions"]:
            date = parse(txn["transactionDate"]).strftime("%Y-%m-%d")
            price = float(txn["purchasePrice"]["units"])
            series.append({"ds": date, "y": price})
        return sorted(series, key=lambda x: x["ds"])
    except:
        return []


all_json_data = []
for root, dirs, files in os.walk("data"):
    for file in files:
        if file.endswith(".json"):
            try:
                with open(os.path.join(root, file), 'r') as f:
                    all_json_data.append(json.load(f))
            except:
                pass


forecast_summary = ""
models_for_plotting = []  

for json_obj in all_json_data:
    net_worth_ts = extract_net_worth_series(json_obj)
    epf_ts = extract_epf_series(json_obj)
    mf_ts = extract_mutual_fund_series(json_obj)

    if net_worth_ts:
        model, forecast = forecast_series(net_worth_ts)
        forecast_summary += "\nNet Worth Forecast:\n" + forecast.tail(3).to_string(index=False)
        models_for_plotting.append(("Net Worth", model, forecast))

    if epf_ts:
        model, forecast = forecast_series(epf_ts)
        forecast_summary += "\n\nEPF Forecast:\n" + forecast.tail(3).to_string(index=False)
        models_for_plotting.append(("EPF", model, forecast))

    if mf_ts:
        model, forecast = forecast_series(mf_ts)
        forecast_summary += "\n\nMutual Fund NAV Forecast:\n" + forecast.tail(3).to_string(index=False)
        models_for_plotting.append(("Mutual Fund NAV", model, forecast))


query = input("Enter your query: ")
context_docs = retriever.get_relevant_documents(query)

prompt_template = PromptTemplate(
    template="""You are a financial analyst. Use the following context and forecast summary to answer the question:

Context: {context}
Forecast Summary: {forecast_summary}
Question: {query}""",
    input_variables=["context", "forecast_summary", "query"]
)

prompt = prompt_template.format(
    context="\n\n".join([doc.page_content for doc in context_docs]),
    forecast_summary=forecast_summary,
    query=query
)

response = llm.invoke(prompt)
print(response.content)


import plotly.io as pio
for name, model, forecast in models_for_plotting:
    fig = plot_plotly(model, forecast)
    fig.update_layout(title=f"{name} Forecast")
    fig.show()