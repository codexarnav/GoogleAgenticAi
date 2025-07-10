import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
import numpy as np
from prophet import Prophet
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
from dateutil.parser import parse
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="Financial Intelligence Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .alert-warning {
        background-color: #fff3cd;
        border-left-color: #ffc107;
        color: #856404;
    }
    .alert-danger {
        background-color: #f8d7da;
        border-left-color: #dc3545;
        color: #721c24;
    }
    .alert-success {
        background-color: #d4edda;
        border-left-color: #28a745;
        color: #155724;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'financial_data' not in st.session_state:
    st.session_state.financial_data = None
if 'forecasts' not in st.session_state:
    st.session_state.forecasts = {}
if 'anomalies' not in st.session_state:
    st.session_state.anomalies = []

class FinancialAnalyzer:
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.retriever = None
        
    def load_financial_data(self, data_directory="data"):
        """Load and process financial data from JSON and PDF files"""
        try:
            # Load JSON files
            json_data = []
            for root, dirs, files in os.walk(data_directory):
                for file in files:
                    if file.endswith(".json"):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                json_data.append(json.load(f))
                        except Exception as e:
                            st.warning(f"Error loading {file}: {e}")
            
            # Load documents for RAG
            json_docs = self.load_json_as_documents(data_directory)
            pdf_docs = DirectoryLoader(
                data_directory,
                glob="**/*.pdf",
                loader_cls=PyMuPDFLoader
            ).load()
            
            all_docs = json_docs + pdf_docs
            
            if all_docs:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                texts = text_splitter.split_documents(all_docs)
                
                self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                self.vector_store = Chroma.from_documents(texts, self.embeddings)
                self.retriever = self.vector_store.as_retriever(search_type='mmr', search_kwargs={'k': 5})
                
                # Initialize LLM
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=0.1,
                    api_key='AIzaSyBlVAGaEchXF46DbcNFE66X6HbbcN4oSXI'
                )
            
            return json_data
            
        except Exception as e:
            st.error(f"Error loading financial data: {e}")
            return []
    
    def load_json_as_documents(self, directory):
        """Load JSON files as documents for RAG"""
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
                        st.warning(f"Error loading {file_path}: {e}")
        return documents
    
    def detect_anomalies(self, financial_data):
        """Detect financial anomalies and risks"""
        anomalies = []
        
        for data in financial_data:
            try:
                # EMI to Income Ratio Check
                if 'netWorthResponse' in data:
                    net_worth = data['netWorthResponse'].get('totalNetWorthValue', {}).get('units', 0)
                    monthly_income = float(net_worth) / 12 if net_worth else 50000  # Estimate
                    
                    # Simulate EMI calculation
                    emi_amount = monthly_income * 0.55  # Simulated high EMI
                    emi_ratio = (emi_amount / monthly_income) * 100
                    
                    if emi_ratio > 50:
                        anomalies.append({
                            'type': 'danger',
                            'message': f"⚠️ Your EMI is {emi_ratio:.1f}% of your income — high risk.",
                            'category': 'EMI Risk',
                            'severity': 'High'
                        })
                
                # Credit Utilization Check
                if 'creditCards' in data:
                    total_limit = sum([card.get('limit', 0) for card in data['creditCards']])
                    total_used = sum([card.get('used', 0) for card in data['creditCards']])
                    
                    if total_limit > 0:
                        utilization = (total_used / total_limit) * 100
                        if utilization > 40:
                            anomalies.append({
                                'type': 'warning',
                                'message': f"⚠️ Credit utilization is {utilization:.1f}% — above recommended 40%.",
                                'category': 'Credit Risk',
                                'severity': 'Medium'
                            })
                
                # Investment Performance Check
                if 'transactions' in data:
                    # Simulate underperformance
                    anomalies.append({
                        'type': 'warning',
                        'message': "⚠️ SIP X underperformed NIFTY50 by 2% last year.",
                        'category': 'Investment Performance',
                        'severity': 'Medium'
                    })
                
                # Emergency Fund Check
                if 'netWorthResponse' in data:
                    net_worth = float(data['netWorthResponse'].get('totalNetWorthValue', {}).get('units', 0))
                    monthly_expense = net_worth * 0.05  # Estimated monthly expense
                    emergency_fund = net_worth * 0.1  # Estimated emergency fund
                    
                    if emergency_fund < (monthly_expense * 6):
                        anomalies.append({
                            'type': 'warning',
                            'message': "⚠️ Emergency fund is below 6 months of expenses.",
                            'category': 'Emergency Fund',
                            'severity': 'Medium'
                        })
                
            except Exception as e:
                st.warning(f"Error in anomaly detection: {e}")
        
        return anomalies
    
    def generate_forecasts(self, financial_data):
        """Generate financial forecasts"""
        forecasts = {}
        
        for data in financial_data:
            try:
                # Net Worth Forecast
                net_worth_ts = self.extract_net_worth_series(data)
                if net_worth_ts and len(net_worth_ts) > 1:
                    model, forecast = self.forecast_series(net_worth_ts)
                    forecasts['net_worth'] = {
                        'model': model,
                        'forecast': forecast,
                        'current': net_worth_ts[-1]['y'],
                        'projected': forecast['yhat'].iloc[-1]
                    }
                
                # EPF Forecast
                epf_ts = self.extract_epf_series(data)
                if epf_ts and len(epf_ts) > 1:
                    model, forecast = self.forecast_series(epf_ts)
                    forecasts['epf'] = {
                        'model': model,
                        'forecast': forecast,
                        'current': epf_ts[-1]['y'],
                        'projected': forecast['yhat'].iloc[-1]
                    }
                
                # Mutual Fund Forecast
                mf_ts = self.extract_mutual_fund_series(data)
                if mf_ts and len(mf_ts) > 1:
                    model, forecast = self.forecast_series(mf_ts)
                    forecasts['mutual_funds'] = {
                        'model': model,
                        'forecast': forecast,
                        'current': mf_ts[-1]['y'],
                        'projected': forecast['yhat'].iloc[-1]
                    }
                    
            except Exception as e:
                st.warning(f"Error generating forecasts: {e}")
        
        return forecasts
    
    def extract_net_worth_series(self, json_obj):
        """Extract net worth time series data"""
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
    
    def extract_epf_series(self, json_obj):
        """Extract EPF time series data"""
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
    
    def extract_mutual_fund_series(self, json_obj):
        """Extract mutual fund time series data"""
        series = []
        try:
            for txn in json_obj["transactions"]:
                date = parse(txn["transactionDate"]).strftime("%Y-%m-%d")
                price = float(txn["purchasePrice"]["units"])
                series.append({"ds": date, "y": price})
            return sorted(series, key=lambda x: x["ds"])
        except:
            return []
    
    def forecast_series(self, data, periods=6):
        """Generate forecast using Prophet"""
        df = pd.DataFrame(data)
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'])
        
        model = Prophet()
        model.fit(df)
        
        future = model.make_future_dataframe(periods=periods, freq='M')
        forecast = model.predict(future)
        return model, forecast
    
    def query_rag(self, query):
        """Query the RAG system"""
        if not self.retriever or not self.llm:
            return "RAG system not initialized. Please load financial data first."
        
        try:
            context_docs = self.retriever.get_relevant_documents(query)
            
            prompt_template = PromptTemplate(
                template="""You are a financial analyst. Use the following context to answer the question:

Context: {context}
Question: {query}

Provide a comprehensive analysis based on the available data.""",
                input_variables=["context", "query"]
            )
            
            prompt = prompt_template.format(
                context="\n\n".join([doc.page_content for doc in context_docs]),
                query=query
            )
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            return f"Error querying RAG system: {e}"

# Initialize analyzer
@st.cache_resource
def get_analyzer():
    return FinancialAnalyzer()

def main():
    st.markdown('<h1 class="main-header">💰 Financial Intelligence Dashboard</h1>', unsafe_allow_html=True)
    
    analyzer = get_analyzer()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Dashboard Controls")
        
        # Data loading
        if st.button("🔄 Load Financial Data", type="primary"):
            with st.spinner("Loading financial data..."):
                financial_data = analyzer.load_financial_data()
                if financial_data:
                    st.session_state.financial_data = financial_data
                    st.session_state.anomalies = analyzer.detect_anomalies(financial_data)
                    st.session_state.forecasts = analyzer.generate_forecasts(financial_data)
                    st.success("Data loaded successfully!")
                else:
                    st.error("No financial data found. Please check your data directory.")
        
        # Navigation
        st.subheader("🧭 Navigation")
        page = st.selectbox(
            "Choose a section:",
            ["🏠 Home & Alerts", "📈 Vision Board", "🤖 AI Assistant", "📋 Detailed Analysis"]
        )
    
    # Main content based on selected page
    if page == "🏠 Home & Alerts":
        show_home_page()
    elif page == "📈 Vision Board":
        show_vision_board()
    elif page == "🤖 AI Assistant":
        show_ai_assistant(analyzer)
    elif page == "📋 Detailed Analysis":
        show_detailed_analysis()

def show_home_page():
    """Display home page with anomaly alerts"""
    st.header("🏠 Home Dashboard")
    
    # Anomaly Radar Section
    st.subheader("🔍 Anomaly Radar")
    
    if st.session_state.anomalies:
        for anomaly in st.session_state.anomalies:
            alert_type = anomaly['type']
            css_class = f"alert-{alert_type}"
            
            st.markdown(f"""
            <div class="alert-box {css_class}">
                <strong>{anomaly['category']}</strong> - {anomaly['severity']} Risk<br>
                {anomaly['message']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No anomalies detected. Load your financial data to see risk alerts.")
    
    # Quick metrics
    if st.session_state.financial_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Data Sources", len(st.session_state.financial_data))
        
        with col2:
            st.metric("⚠️ Active Alerts", len(st.session_state.anomalies))
        
        with col3:
            forecast_count = len(st.session_state.forecasts)
            st.metric("📈 Forecasts", forecast_count)
        
        with col4:
            st.metric("🔍 Risk Level", "Medium" if st.session_state.anomalies else "Low")

def show_vision_board():
    """Display vision board with financial projections"""
    st.header("📈 Vision Board - Financial Future")
    
    # What-if scenario controls
    st.subheader("🎛️ What-If Scenario Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Investment Parameters")
        current_age = st.slider("Current Age", 20, 65, 30)
        retirement_age = st.slider("Target Retirement Age", 50, 75, 60)
        monthly_sip = st.slider("Monthly SIP Amount (₹)", 1000, 50000, 10000, step=1000)
        expected_return = st.slider("Expected Annual Return (%)", 8, 15, 12)
        
    with col2:
        st.subheader("💰 Financial Goals")
        target_corpus = st.slider("Target Retirement Corpus (₹ Crores)", 1, 10, 5)
        risk_appetite = st.select_slider(
            "Risk Appetite",
            options=["Conservative", "Moderate", "Aggressive"],
            value="Moderate"
        )
        inflation_rate = st.slider("Expected Inflation Rate (%)", 3, 8, 6)
    
    # Calculate projections
    years_to_retirement = retirement_age - current_age
    
    # SIP calculation
    monthly_return = expected_return / 12 / 100
    total_months = years_to_retirement * 12
    
    if monthly_return > 0:
        future_value = monthly_sip * (((1 + monthly_return) ** total_months - 1) / monthly_return)
    else:
        future_value = monthly_sip * total_months
    
    # Create visualization
    st.subheader("🎯 Financial Projection")
    
    # Generate year-wise data
    years = list(range(current_age, retirement_age + 1))
    corpus_values = []
    
    for year in years:
        years_invested = year - current_age
        if years_invested == 0:
            corpus_values.append(0)
        else:
            months_invested = years_invested * 12
            if monthly_return > 0:
                value = monthly_sip * (((1 + monthly_return) ** months_invested - 1) / monthly_return)
            else:
                value = monthly_sip * months_invested
            corpus_values.append(value)
    
    # Create the projection chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=years,
        y=corpus_values,
        mode='lines+markers',
        name='Projected Corpus',
        line=dict(color='#1f77b4', width=3),
        fill='tonexty',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))
    
    fig.add_hline(
        y=target_corpus * 10000000,  # Convert crores to rupees
        line_dash="dash",
        line_color="red",
        annotation_text=f"Target: ₹{target_corpus} Cr"
    )
    
    fig.update_layout(
        title="Net Worth Projection Over Time",
        xaxis_title="Age",
        yaxis_title="Corpus Value (₹)",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "💰 Final Corpus",
            f"₹{future_value/10000000:.2f} Cr",
            delta=f"₹{(future_value - target_corpus*10000000)/10000000:.2f} Cr"
        )
    
    with col2:
        total_investment = monthly_sip * total_months
        st.metric(
            "📊 Total Investment",
            f"₹{total_investment/10000000:.2f} Cr"
        )
    
    with col3:
        returns = future_value - total_investment
        st.metric(
            "📈 Total Returns",
            f"₹{returns/10000000:.2f} Cr"
        )
    
    # Scenario comparison
    st.subheader("🔄 Scenario Comparison")
    
    scenarios = {
        "Conservative": {"return": 8, "description": "Low risk, steady growth"},
        "Moderate": {"return": 12, "description": "Balanced risk and return"},
        "Aggressive": {"return": 15, "description": "High risk, high return potential"}
    }
    
    comparison_data = []
    for scenario_name, scenario_data in scenarios.items():
        scenario_return = scenario_data["return"] / 12 / 100
        scenario_fv = monthly_sip * (((1 + scenario_return) ** total_months - 1) / scenario_return)
        
        comparison_data.append({
            "Scenario": scenario_name,
            "Expected Return": f"{scenario_data['return']}%",
            "Final Corpus": f"₹{scenario_fv/10000000:.2f} Cr",
            "Description": scenario_data["description"]
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)

def show_ai_assistant(analyzer):
    """Display AI assistant for financial queries"""
    st.header("🤖 AI Financial Assistant")
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about your finances..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your financial data..."):
                response = analyzer.query_rag(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Quick query buttons
    st.subheader("💡 Quick Questions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 What's my current financial health?"):
            query = "Analyze my current financial position and provide a comprehensive health report"
            response = analyzer.query_rag(query)
            st.info(response)
    
    with col2:
        if st.button("📈 How are my investments performing?"):
            query = "Analyze my investment portfolio performance and provide recommendations"
            response = analyzer.query_rag(query)
            st.info(response)
    
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("⚠️ What are my financial risks?"):
            query = "Identify potential financial risks and provide mitigation strategies"
            response = analyzer.query_rag(query)
            st.warning(response)
    
    with col4:
        if st.button("🎯 Am I on track for retirement?"):
            query = "Assess my retirement planning and provide guidance on achieving retirement goals"
            response = analyzer.query_rag(query)
            st.info(response)

def show_detailed_analysis():
    """Display detailed financial analysis"""
    st.header("📋 Detailed Financial Analysis")
    
    if not st.session_state.forecasts:
        st.warning("No forecast data available. Please load your financial data first.")
        return
    
    # Forecast visualizations
    for forecast_type, forecast_data in st.session_state.forecasts.items():
        st.subheader(f"📈 {forecast_type.replace('_', ' ').title()} Forecast")
        
        # Create forecast visualization
        forecast_df = forecast_data['forecast']
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'][:len(forecast_df)//2],
            y=forecast_df['yhat'][:len(forecast_df)//2],
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'][len(forecast_df)//2:],
            y=forecast_df['yhat'][len(forecast_df)//2:],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        # Confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_upper'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_lower'],
            mode='lines',
            name='Confidence Interval',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)'
        ))
        
        fig.update_layout(
            title=f"{forecast_type.replace('_', ' ').title()} Trend Analysis",
            xaxis_title="Date",
            yaxis_title="Value (₹)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Value", f"₹{forecast_data['current']:,.0f}")
        
        with col2:
            st.metric("Projected Value", f"₹{forecast_data['projected']:,.0f}")
        
        with col3:
            growth = ((forecast_data['projected'] - forecast_data['current']) / forecast_data['current']) * 100
            st.metric("Growth Rate", f"{growth:.1f}%")

if __name__ == "__main__":
    main()