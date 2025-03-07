# FinAssist: Financial AI Agent for Risk Assessment

## Project Overview

FinAssist is an AI-powered financial agent designed to help financial advisors provide personalized risk assessments and recommendations to customers. This system demonstrates a sophisticated approach to applying machine learning to solve complex financial problems while prioritizing customer experience and regulatory compliance.

## System Architecture

The overall architecture of FinAssist is designed to process multiple data sources, perform advanced analytics, and deliver personalized recommendations:

![FinAssist Architecture](FinAssist%20System%20Architecture%20Flow%20Chart.png)

## Key Features

1. **Multi-source Data Integration**
   - Customer financial history
   - Market data and indicators
   - News sentiment analysis 
   - Demographic information

2. **Machine Learning Risk Assessment**
   - Random forest classification for risk profile determination
   - Gradient boosting regression for investment return prediction
   - Sentiment analysis of financial news
   - Feature importance visualization

3. **Personalized Financial Recommendations**
   - Asset allocation suggestions
   - Debt management strategies
   - Retirement planning insights
   - Family financial security guidance
   - Tax efficiency suggestions

4. **Regulatory Compliance**
   - Built-in compliance checks
   - KYC (Know Your Client) verification
   - Suitability validation
   - Comprehensive audit logging

## Data Visualizations

### FinAssist Visualization Dashboard
![FinAssist Dashboard](finassist-dashboard.gif)

### Risk Profile Distribution
![Risk Profiles](risk-profile.gif)

### Average Risk Score by Age Group
![Risk by Age](risk-by-age.gif)

### Model Performance Metrics
![Model Performance](model-performance.gif)

## Technical Implementation

### Core Components

- **FinancialRiskAgent**: Main class that handles risk assessment and recommendation generation
- **FinancialNewsMonitor**: Class for financial news analysis and sentiment extraction
- **RegulatoryComplianceChecker**: Class that validates recommendations against regulatory requirements

### Dependencies

- Python 3.8+
- pandas, numpy, scikit-learn
- PyTorch (for sentiment analysis)
- matplotlib, seaborn (for visualization)
- React (for the user interface)

## Usage

```python
# Initialize the agent
agent = FinancialRiskAgent()

# Load and preprocess data
df, preprocessor = agent.load_and_preprocess_data()

# Train the models
agent.train_models(df, preprocessor)

# Example customer data
customer = {
    'customer_id': 12345,
    'age': 45,
    'income': 85000,
    'savings': 75000,
    'debt': 120000,
    'credit_score': 710,
    'education_level': 'Bachelor',
    'employment_status': 'Employed',
    'has_mortgage': 1,
    'has_dependents': 1,
    'market_sentiment': 0.2,
    'past_investment_returns': 0.06
}

# Analyze risk for the customer
risk_assessment = agent.analyze_customer_risk(customer)

# Print the assessment
print(json.dumps(risk_assessment, indent=2))
```

## Project Components

1. **financial-ai-agent-project.py**: Main Python implementation of the FinAssist agent
2. **finassist-plots.tsx**: React component for data visualizations
3. **cover-letter.md**: Sample cover letter for job applications

## Future Development

1. **Real-time Market Integration**
   - Connect to market data feeds for up-to-date analysis

2. **Expanded Recommendation Engine**
   - Product-specific recommendations based on offerings
   - Time-sensitive investment opportunities

3. **Interactive Customer Dashboard**
   - Visual representation of risk assessment
   - Scenario modeling for different financial decisions

## License

This project is available under the MIT License.

## Author

Milad Nasiri
