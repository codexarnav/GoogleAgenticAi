#  Financial Intelligence Dashboard

A comprehensive financial analysis and forecasting dashboard built with Streamlit that provides AI-powered insights, anomaly detection, and future projections for your financial data.

##  Features

###  Home & Alerts
- **Anomaly Radar**: Real-time detection of financial risks and anomalies
- **Risk Assessment**: EMI-to-income ratio analysis, credit utilization monitoring
- **Investment Performance**: Underperformance alerts and portfolio analysis
- **Emergency Fund**: Automated checks for emergency fund adequacy

### Vision Board
- **What-If Scenarios**: Interactive financial planning with customizable parameters
- **Retirement Planning**: Corpus projection based on SIP investments
- **Scenario Comparison**: Conservative, Moderate, and Aggressive investment strategies
- **Goal Tracking**: Visual progress tracking toward financial targets

###  AI Assistant
- **RAG-Powered Analysis**: Context-aware financial advice using your data
- **Natural Language Queries**: Ask questions about your finances in plain English
- **Smart Recommendations**: Personalized financial guidance
- **Quick Insights**: Pre-built queries for common financial questions

###  Detailed Analysis
- **Time Series Forecasting**: Prophet-based predictions for financial growth
- **Multi-Asset Tracking**: Net worth, EPF, and mutual fund projections
- **Confidence Intervals**: Statistical uncertainty in forecasts
- **Growth Metrics**: Performance indicators and trend analysis

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd financial-intelligence-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   export GOOGLE_API_KEY="your_gemini_api_key_here"
   ```
   Or create a `.env` file:
   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

4. **Prepare your data directory**
   ```bash
   mkdir data
   # Place your JSON and PDF financial files in the data directory
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📦 Dependencies

```
streamlit
pandas
plotly
numpy
prophet
langchain-community
langchain-core
langchain-huggingface
langchain-google-genai
chromadb
sentence-transformers
pymupdf
python-dateutil
```

## 📁 Project Structure

```
financial-intelligence-dashboard/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── data/                 # Financial data directory
│   ├── *.json           # JSON financial data files
│   └── *.pdf            # PDF financial documents
├── README.md            # This file
└── .env                 # Environment variables (optional)
```

## 📊 Data Format

### JSON Data Structure
The application expects JSON files with the following structure:

```json
{
  "netWorthResponse": {
    "totalNetWorthValue": {
      "units": "1000000"
    }
  },
  "creditCards": [
    {
      "limit": 100000,
      "used": 25000
    }
  ],
  "uanAccounts": [
    {
      "rawDetails": {
        "est_details": [
          {
            "doj_epf": "01-01-2020",
            "pf_balance": {
              "net_balance": "500000"
            }
          }
        ]
      }
    }
  ],
  "transactions": [
    {
      "transactionDate": "2023-01-15",
      "purchasePrice": {
        "units": "15000"
      }
    }
  ]
}
```

### PDF Documents
- Financial statements
- Investment reports
- Banking documents
- Insurance policies

## 🔧 Configuration

### Google Gemini API
1. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set the `GOOGLE_API_KEY` environment variable
3. Update the API key in the code if needed

### Customization Options
- **Anomaly Thresholds**: Modify risk detection parameters in `detect_anomalies()`
- **Forecast Periods**: Adjust prediction timeframes in `forecast_series()`
- **UI Styling**: Update CSS in the `st.markdown()` sections
- **Data Sources**: Extend `load_financial_data()` for additional file formats

## 🎯 Usage Guide

### Getting Started
1. **Launch the application** and click "🔄 Load Financial Data"
2. **Navigate** through different sections using the sidebar
3. **Upload your financial data** in JSON or PDF format to the `data/` directory
4. **Explore insights** across all four main sections

### Best Practices
- **Data Quality**: Ensure your JSON files follow the expected structure
- **Regular Updates**: Refresh data periodically for accurate forecasts
- **Security**: Keep your API keys secure and never commit them to version control
- **Backup**: Maintain backups of your financial data

## 🛠️ Technical Details

### Architecture
- **Frontend**: Streamlit for interactive web interface
- **Backend**: Python with pandas for data processing
- **AI/ML**: Prophet for time series forecasting, Gemini for natural language processing
- **Vector Database**: ChromaDB for document similarity search
- **Embeddings**: HuggingFace sentence-transformers for text vectorization

### Key Algorithms
- **Prophet**: Facebook's time series forecasting algorithm
- **RAG**: Retrieval-Augmented Generation for context-aware responses
- **Anomaly Detection**: Statistical thresholds and rule-based systems
- **Portfolio Analysis**: Modern portfolio theory metrics

## 🔍 Troubleshooting

### Common Issues

**1. Data Loading Errors**
```
Error: No financial data found
```
- Ensure JSON files are in the correct format
- Check file permissions in the data directory
- Verify the data directory path

**2. API Connection Issues**
```
Error: Google API key not found
```
- Verify your Gemini API key is set correctly
- Check API quotas and usage limits
- Ensure internet connectivity

**3. Forecast Generation Problems**
```
Error: Insufficient data for forecasting
```
- Provide at least 2 data points for time series
- Check date formats in your JSON files
- Verify numeric values are properly formatted

### Performance Optimization
- Use `@st.cache_resource` for expensive computations
- Implement data pagination for large datasets
- Consider using Streamlit's session state efficiently

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📈 Future Enhancements

- [ ] **Real-time Data Integration**: Connect with bank APIs
- [ ] **Advanced ML Models**: Implement deep learning for better predictions
- [ ] **Mobile App**: React Native companion app
- [ ] **Multi-currency Support**: Handle international investments
- [ ] **Tax Optimization**: Automated tax planning recommendations
- [ ] **Social Features**: Share insights with financial advisors
- [ ] **Risk Modeling**: Monte Carlo simulations for portfolio analysis

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support, please:
1. Check the troubleshooting section above
2. Open an issue on GitHub
3. Contact the development team

## 🙏 Acknowledgments

- **Streamlit** for the amazing web framework
- **Prophet** for time series forecasting capabilities
- **Google Gemini** for AI-powered insights
- **LangChain** for RAG implementation
- **Plotly** for interactive visualizations

---

