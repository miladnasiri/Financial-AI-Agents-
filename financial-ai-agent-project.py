"""
FinAssist: An AI Agent for Personalized Financial Risk Assessment
---------------------------------------------------------------

This project demonstrates a framework for an AI agent that helps financial advisors
and customers assess investment risk profiles based on multiple data sources including:
- Customer financial history
- Market data
- News sentiment analysis
- Demographic information

The agent uses machine learning techniques to create personalized risk scores and
recommendations that align with TD's customer-focused approach while maintaining
regulatory compliance.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib
import logging
import json
from datetime import datetime

class FinancialRiskAgent:
    """
    AI Agent for personalized financial risk assessment and recommendation.
    """
    
    def __init__(self, config_path="config.json"):
        """Initialize the financial risk assessment agent"""
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.models = {}
        self.tokenizer = None
        self.sentiment_model = None
        self.feature_importance = None
        self.initialize_models()
        
    def _setup_logging(self):
        """Set up logging configuration"""
        logger = logging.getLogger('FinancialRiskAgent')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def _load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {config_path}, using default configuration")
            return {
                "models": {
                    "risk_profile": {
                        "type": "random_forest",
                        "params": {"n_estimators": 100, "max_depth": 10}
                    },
                    "investment_return": {
                        "type": "gradient_boosting",
                        "params": {"n_estimators": 100, "learning_rate": 0.1}
                    }
                },
                "data_sources": {
                    "customer_data": "data/customer_financial_data.csv",
                    "market_data": "data/market_indicators.csv",
                    "news_data": "data/financial_news.csv"
                },
                "sentiment_analysis": {
                    "model": "finbert-sentiment",
                    "cache_dir": "models/"
                },
                "risk_thresholds": {
                    "low": 0.3,
                    "medium": 0.6,
                    "high": 0.9
                }
            }
    
    def initialize_models(self):
        """Initialize and load all models"""
        self.logger.info("Initializing models")
        
        # Initialize risk profile classification model
        risk_config = self.config["models"]["risk_profile"]
        if risk_config["type"] == "random_forest":
            self.models["risk_profile"] = RandomForestClassifier(**risk_config["params"])
        
        # Initialize investment return prediction model
        return_config = self.config["models"]["investment_return"]
        if return_config["type"] == "gradient_boosting":
            self.models["investment_return"] = GradientBoostingRegressor(**return_config["params"])
        
        # Initialize sentiment analysis model if needed
        if self.config.get("use_sentiment", True):
            try:
                model_name = self.config["sentiment_analysis"]["model"]
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.logger.info(f"Sentiment model loaded: {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load sentiment model: {e}")
    
    def load_and_preprocess_data(self, data_path=None):
        """Load and preprocess data from configured sources"""
        if data_path:
            # Load from specific path for testing/demo
            df = pd.read_csv(data_path)
            self.logger.info(f"Data loaded from {data_path}: {len(df)} records")
            return self._preprocess_data(df)
        
        # In a real implementation, we would load and merge data from multiple sources
        # (customer data, market data, news data)
        self.logger.info("Loading data from configured sources")
        
        try:
            # Load customer financial data
            customer_data = pd.read_csv(self.config["data_sources"]["customer_data"])
            self.logger.info(f"Customer data loaded: {len(customer_data)} records")
            
            # Load market indicators
            market_data = pd.read_csv(self.config["data_sources"]["market_data"])
            self.logger.info(f"Market data loaded: {len(market_data)} records")
            
            # Merge datasets (simplified for demonstration)
            # In real implementation, we would use more sophisticated data integration
            merged_data = customer_data.merge(market_data, on='date', how='left')
            
            return self._preprocess_data(merged_data)
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            # Return dummy data for demonstration
            return self._generate_dummy_data()
    
    def _preprocess_data(self, df):
        """Preprocess data for model training and prediction"""
        self.logger.info("Preprocessing data")
        
        # Handle missing values
        df = df.fillna(df.mean(numeric_only=True))
        
        # Create feature transformers
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return df, preprocessor
    
    def _generate_dummy_data(self):
        """Generate dummy data for demonstration"""
        self.logger.info("Generating dummy data for demonstration")
        
        # Create dummy customer financial data
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'customer_id': range(1, n_samples + 1),
            'age': np.random.randint(18, 80, n_samples),
            'income': np.random.normal(70000, 30000, n_samples),
            'savings': np.random.normal(50000, 40000, n_samples),
            'debt': np.random.normal(30000, 20000, n_samples),
            'credit_score': np.random.normal(700, 100, n_samples),
            'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'employment_status': np.random.choice(['Employed', 'Self-employed', 'Unemployed', 'Retired'], n_samples),
            'has_mortgage': np.random.choice([0, 1], n_samples),
            'has_dependents': np.random.choice([0, 1], n_samples),
            'market_sentiment': np.random.uniform(-1, 1, n_samples),
            'risk_tolerance': np.random.choice(['Low', 'Medium', 'High'], n_samples),
            'past_investment_returns': np.random.normal(0.07, 0.1, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create feature transformers for dummy data
        numeric_features = ['age', 'income', 'savings', 'debt', 'credit_score', 
                           'has_mortgage', 'has_dependents', 'market_sentiment', 
                           'past_investment_returns']
        
        categorical_features = ['education_level', 'employment_status', 'risk_tolerance']
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return df, preprocessor
    
    def train_models(self, df, preprocessor, target_col='risk_profile'):
        """Train the risk assessment models"""
        self.logger.info(f"Training models with target: {target_col}")
        
        # For risk profile classification
        X = df.drop([target_col, 'customer_id'], axis=1)
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train pipeline for risk profile model
        risk_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', self.models['risk_profile'])
        ])
        
        risk_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = risk_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.logger.info(f"Risk profile model trained. Accuracy: {accuracy:.4f}")
        
        # Extract feature importance
        if hasattr(self.models['risk_profile'], 'feature_importances_'):
            # Get feature names after preprocessing
            feature_names = []
            for name, transformer, features in preprocessor.transformers_:
                if hasattr(transformer, 'get_feature_names_out'):
                    transformed_features = transformer.get_feature_names_out(features)
                    feature_names.extend(transformed_features)
                else:
                    feature_names.extend(features)
            
            # Get feature importances and pair with names
            importances = self.models['risk_profile'].feature_importances_
            indices = np.argsort(importances)[::-1]
            
            self.feature_importance = {
                'features': [feature_names[i] for i in indices],
                'importances': [importances[i] for i in indices]
            }
        
        # Store the trained pipeline
        self.models['risk_pipeline'] = risk_model
        return accuracy
    
    def analyze_customer_risk(self, customer_data):
        """
        Analyze risk profile for a specific customer
        
        Args:
            customer_data: Dictionary with customer financial information
            
        Returns:
            Dictionary with risk assessment and recommendations
        """
        self.logger.info(f"Analyzing risk for customer")
        
        # Convert customer data to DataFrame format
        customer_df = pd.DataFrame([customer_data])
        
        # If we don't have a trained model, use rules-based approach
        if 'risk_pipeline' not in self.models:
            return self._rules_based_risk_assessment(customer_data)
        
        # Predict risk profile using the trained model
        risk_profile_prob = self.models['risk_pipeline'].predict_proba(customer_df)[0]
        risk_profile_idx = np.argmax(risk_profile_prob)
        risk_profile = self.models['risk_pipeline'].classes_[risk_profile_idx]
        confidence = risk_profile_prob[risk_profile_idx]
        
        # Create risk score (0-100)
        risk_score = self._calculate_risk_score(customer_data)
        
        # Generate personalized recommendations
        recommendations = self._generate_recommendations(risk_profile, risk_score, customer_data)
        
        # Include market context using sentiment analysis if available
        market_context = self._analyze_market_context() if self.sentiment_model else None
        
        result = {
            'customer_id': customer_data.get('customer_id', 'unknown'),
            'analysis_timestamp': datetime.now().isoformat(),
            'risk_profile': risk_profile,
            'risk_score': risk_score,
            'confidence': float(confidence),
            'key_factors': self._identify_key_risk_factors(customer_data),
            'recommendations': recommendations,
            'market_context': market_context
        }
        
        self.logger.info(f"Risk analysis completed: {risk_profile} profile with score {risk_score}")
        return result
    
    def _rules_based_risk_assessment(self, customer_data):
        """
        Fallback rules-based risk assessment when model is not available
        """
        # Simple rules for demo purposes
        age = customer_data.get('age', 40)
        income = customer_data.get('income', 0)
        savings = customer_data.get('savings', 0)
        debt = customer_data.get('debt', 0)
        
        # Calculate debt-to-income ratio
        dti_ratio = debt / income if income > 0 else 0
        
        # Calculate savings-to-income ratio
        sti_ratio = savings / income if income > 0 else 0
        
        # Basic risk score calculation
        risk_factors = [
            -0.3 if age > 60 else 0.1,  # Older clients tend to need more conservative approaches
            -0.2 if dti_ratio > 0.4 else 0.1,  # High debt is risky
            0.3 if sti_ratio > 1 else -0.1,  # High savings relative to income is good
            0.2 if customer_data.get('credit_score', 0) > 720 else -0.1  # Good credit is positive
        ]
        
        base_score = 50  # Middle score
        risk_adjustment = sum(risk_factors) * 100
        risk_score = max(0, min(100, base_score + risk_adjustment))
        
        # Determine profile based on score
        if risk_score < 30:
            risk_profile = 'Conservative'
        elif risk_score < 60:
            risk_profile = 'Moderate'
        else:
            risk_profile = 'Aggressive'
            
        # Generate recommendations
        recommendations = self._generate_recommendations(risk_profile, risk_score, customer_data)
        
        return {
            'customer_id': customer_data.get('customer_id', 'unknown'),
            'analysis_timestamp': datetime.now().isoformat(),
            'risk_profile': risk_profile,
            'risk_score': risk_score,
            'confidence': 0.7,  # Lower confidence for rules-based approach
            'key_factors': self._identify_key_risk_factors(customer_data),
            'recommendations': recommendations,
            'note': "Rules-based assessment (model not trained)"
        }
    
    def _calculate_risk_score(self, customer_data):
        """Calculate a normalized risk score (0-100)"""
        # This would be a more sophisticated calculation in a real implementation
        age = customer_data.get('age', 40)
        income = customer_data.get('income', 70000)
        debt = customer_data.get('debt', 30000)
        credit_score = customer_data.get('credit_score', 700)
        
        # Age factor (younger can take more risk)
        age_factor = max(0, (65 - age) / 45) if age <= 65 else 0
        
        # Income/debt factor
        debt_to_income = min(1, debt / income) if income > 0 else 1
        financial_stability = 1 - debt_to_income
        
        # Credit score factor
        credit_factor = (credit_score - 400) / 400 if credit_score > 400 else 0
        
        # Market sentiment factor (if available)
        sentiment_factor = customer_data.get('market_sentiment', 0)
        
        # Combine factors with different weights
        weighted_score = (
            age_factor * 0.3 +
            financial_stability * 0.25 + 
            credit_factor * 0.15 +
            (sentiment_factor + 1) / 2 * 0.1
        ) / 0.8  # Normalize to account for weights summing to 0.8
        
        # Convert to 0-100 scale
        return min(100, max(0, weighted_score * 100))
    
    def _identify_key_risk_factors(self, customer_data):
        """Identify key factors affecting the risk assessment"""
        factors = []
        
        # Age-based risk factor
        age = customer_data.get('age', 40)
        if age < 30:
            factors.append({"factor": "Age", "impact": "Positive", "description": "Younger age allows for longer investment horizon"})
        elif age > 60:
            factors.append({"factor": "Age", "impact": "Negative", "description": "Shorter time horizon may require more conservative approach"})
        
        # Income stability
        employment = customer_data.get('employment_status', '')
        if employment == 'Unemployed':
            factors.append({"factor": "Employment", "impact": "Negative", "description": "Lack of stable income increases financial vulnerability"})
        
        # Debt level
        income = customer_data.get('income', 1)
        debt = customer_data.get('debt', 0)
        dti_ratio = debt / income if income > 0 else 0
        if dti_ratio > 0.4:
            factors.append({"factor": "Debt-to-Income", "impact": "Negative", "description": "High debt relative to income may limit investment capacity"})
        elif dti_ratio < 0.2:
            factors.append({"factor": "Debt-to-Income", "impact": "Positive", "description": "Low debt level provides financial flexibility"})
        
        # Savings level
        savings = customer_data.get('savings', 0)
        months_of_expenses = savings / (income/12) if income > 0 else 0
        if months_of_expenses < 3:
            factors.append({"factor": "Emergency Fund", "impact": "Negative", "description": "Limited emergency savings increases vulnerability to financial shocks"})
        elif months_of_expenses > 6:
            factors.append({"factor": "Emergency Fund", "impact": "Positive", "description": "Strong emergency fund provides financial security"})
        
        return factors
    
    def _generate_recommendations(self, risk_profile, risk_score, customer_data):
        """Generate personalized financial recommendations based on risk profile"""
        recommendations = []
        
        age = customer_data.get('age', 40)
        income = customer_data.get('income', 70000)
        savings = customer_data.get('savings', 50000)
        debt = customer_data.get('debt', 30000)
        has_mortgage = customer_data.get('has_mortgage', 0)
        has_dependents = customer_data.get('has_dependents', 0)
        
        # Basic recommendations based on risk profile
        if risk_profile == 'Conservative':
            recommendations.append({
                "category": "Asset Allocation",
                "recommendation": "Consider a portfolio with 70-80% fixed income and 20-30% equities",
                "rationale": "Your risk profile suggests a focus on capital preservation over growth"
            })
        elif risk_profile == 'Moderate':
            recommendations.append({
                "category": "Asset Allocation",
                "recommendation": "Consider a balanced portfolio with 40-60% equities and 40-60% fixed income",
                "rationale": "Your risk profile suggests a balance between growth and income"
            })
        else:  # Aggressive
            recommendations.append({
                "category": "Asset Allocation",
                "recommendation": "Consider a growth-oriented portfolio with 70-80% equities and 20-30% fixed income",
                "rationale": "Your risk profile suggests a focus on long-term growth potential"
            })
        
        # Debt recommendations
        if debt > income * 0.4:
            recommendations.append({
                "category": "Debt Management",
                "recommendation": "Consider prioritizing debt reduction before increasing investments",
                "rationale": "Your debt-to-income ratio is above recommended levels"
            })
        
        # Emergency fund recommendations
        monthly_expenses = income / 12 * 0.7  # Estimated monthly expenses
        if savings < monthly_expenses * 3:
            recommendations.append({
                "category": "Emergency Fund",
                "recommendation": "Build emergency savings to cover at least 3-6 months of expenses",
                "rationale": "Having adequate emergency savings is essential before increasing investment risk"
            })
        
        # Retirement planning
        if age > 40 and savings < income * age / 10:
            recommendations.append({
                "category": "Retirement Planning",
                "recommendation": "Consider increasing retirement contributions to catch up",
                "rationale": "Based on your age and savings, you may need to accelerate retirement savings"
            })
        
        # Family security
        if has_dependents and not customer_data.get('has_insurance', False):
            recommendations.append({
                "category": "Protection",
                "recommendation": "Consider life and disability insurance to protect dependents",
                "rationale": "Insurance coverage is important for financial security of dependents"
            })
        
        # Tax efficiency
        if income > 100000:
            recommendations.append({
                "category": "Tax Efficiency",
                "recommendation": "Maximize contributions to tax-advantaged accounts",
                "rationale": "Your income bracket suggests tax-efficient investing strategies would be beneficial"
            })
        
        return recommendations
    
    def _analyze_market_context(self):
        """Analyze current market context using sentiment analysis"""
        # In a real implementation, this would pull recent financial news
        # and analyze sentiment using the NLP model
        
        # Dummy implementation for demonstration
        return {
            "market_sentiment": "neutral",
            "market_volatility": "moderate",
            "interest_rate_trend": "stable",
            "economic_outlook": "moderate growth expected"
        }
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of financial news text"""
        if not self.sentiment_model or not self.tokenizer:
            self.logger.warning("Sentiment model not available")
            return {"sentiment": "neutral", "score": 0.5}
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
        
        # Get probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        # Get predicted class (negative, neutral, positive)
        predicted_class = torch.argmax(probs, dim=1).item()
        
        # Map to sentiment labels
        sentiment_labels = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment = sentiment_labels.get(predicted_class, "neutral")
        
        return {
            "sentiment": sentiment,
            "score": probs[0][predicted_class].item(),
            "breakdown": {
                "negative": probs[0][0].item(),
                "neutral": probs[0][1].item(),
                "positive": probs[0][2].item()
            }
        }
    
    def save_models(self, directory="models"):
        """Save trained models to disk"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.models.items():
            if name != 'sentiment_model':  # Don't save the transformer model
                model_path = os.path.join(directory, f"{name}.joblib")
                joblib.dump(model, model_path)
                self.logger.info(f"Model {name} saved to {model_path}")
    
    def load_models(self, directory="models"):
        """Load trained models from disk"""
        import os
        
        for name in self.models.keys():
            if name != 'sentiment_model':  # Don't load the transformer model
                model_path = os.path.join(directory, f"{name}.joblib")
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
                    self.logger.info(f"Model {name} loaded from {model_path}")
    
    def plot_feature_importance(self, top_n=10):
        """Plot feature importance for the risk profile model"""
        if not self.feature_importance:
            self.logger.warning("Feature importance not available")
            return None
        
        plt.figure(figsize=(10, 6))
        features = self.feature_importance['features'][:top_n]
        importances = self.feature_importance['importances'][:top_n]
        
        sns.barplot(x=importances, y=features)
        plt.title(f"Top {top_n} Features for Risk Profile Prediction")
        plt.xlabel("Importance")
        plt.tight_layout()
        return plt

# Example usage of the agent
def demo_financial_risk_agent():
    """Demonstrate the financial risk agent functionality"""
    # Initialize the agent
    agent = FinancialRiskAgent()
    
    # Load and preprocess data
    df, preprocessor = agent.load_and_preprocess_data()
    
    # Add a risk profile column for training (in real implementation, this would come from labeled data)
    df['risk_profile'] = pd.cut(
        df['past_investment_returns'], 
        bins=[-np.inf, 0.03, 0.10, np.inf], 
        labels=['Conservative', 'Moderate', 'Aggressive']
    )
    
    # Train the models
    agent.train_models(df, preprocessor, target_col='risk_profile')
    
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
    
    # Plot feature importance
    agent.plot_feature_importance()
    plt.savefig('feature_importance.png')
    
    return risk_assessment

# Example implementation of a simple UI for the risk assessment agent
def create_demonstration_ui():
    """
    Create a simple demonstration UI for the financial risk agent
    This would typically be implemented with a web framework like Flask or Streamlit
    """
    import tkinter as tk
    from tkinter import ttk
    import json
    
    # Initialize the agent
    agent = FinancialRiskAgent()
    
    # Load and preprocess data
    df, preprocessor = agent.load_and_preprocess_data()
    
    # Add a risk profile column for training
    df['risk_profile'] = pd.cut(
        df['past_investment_returns'], 
        bins=[-np.inf, 0.03, 0.10, np.inf], 
        labels=['Conservative', 'Moderate', 'Aggressive']
    )
    
    # Train the models
    agent.train_models(df, preprocessor, target_col='risk_profile')
    
    # Create the UI
    root = tk.Tk()
    root.title("FinAssist: Financial Risk Assessment Agent")
    root.geometry("800x600")
    
    # Customer data entry frame
    input_frame = ttk.LabelFrame(root, text="Customer Financial Information")
    input_frame.pack(padx=10, pady=10, fill="x")
    
    # Customer data fields
    fields = [
        ("Customer ID", "customer_id", 12345),
        ("Age", "age", 45),
        ("Annual Income ($)", "income", 85000),
        ("Savings ($)", "savings", 75000),
        ("Total Debt ($)", "debt", 120000),
        ("Credit Score", "credit_score", 710)
    ]
    
    entries = {}
    for i, (label, field, default) in enumerate(fields):
        ttk.Label(input_frame, text=label).grid(row=i, column=0, sticky="w", padx=5, pady=5)
        entry = ttk.Entry(input_frame)
        entry.insert(0, str(default))
        entry.grid(row=i, column=1, sticky="ew", padx=5, pady=5)
        entries[field] = entry
    
    # Drop-down fields
    dropdown_fields = [
        ("Education Level", "education_level", ["High School", "Bachelor", "Master", "PhD"], "Bachelor"),
        ("Employment Status", "employment_status", ["Employed", "Self-employed", "Unemployed", "Retired"], "Employed"),
        ("Risk Tolerance", "risk_tolerance", ["Low", "Medium", "High"], "Medium")
    ]
    
    dropdown_vars = {}
    for i, (label, field, options, default) in enumerate(dropdown_fields):
        ttk.Label(input_frame, text=label).grid(row=i+len(fields), column=0, sticky="w", padx=5, pady=5)
        var = tk.StringVar(value=default)
        dropdown = ttk.Combobox(input_frame, textvariable=var, values=options)
        dropdown.grid(row=i+len(fields), column=1, sticky="ew", padx=5, pady=5)
        dropdown_vars[field] = var
    
    # Checkbox fields
    checkbox_fields = [
        ("Has Mortgage", "has_mortgage", 1),
        ("Has Dependents", "has_dependents", 1),
        ("Has Insurance", "has_insurance", 0)
    ]
    
    checkbox_vars = {}
    for i, (label, field, default) in enumerate(checkbox_fields):
        var = tk.IntVar(value=default)
        checkbox = ttk.Checkbutton(input_frame, text=label, variable=var)
        checkbox.grid(row=i+len(fields)+len(dropdown_fields), column=0, columnspan=2, sticky="w", padx=5, pady=5)
        checkbox_vars[field] = var
    
    # Results display
    result_frame = ttk.LabelFrame(root, text="Risk Assessment Results")
    result_frame.pack(padx=10, pady=10, fill="both", expand=True)
    
    result_text = tk.Text(result_frame, wrap="word", height=15)
    result_text.pack(padx=5, pady=5, fill="both", expand=True)
    
    # Function to analyze customer risk based on input data
    def analyze_risk():
        # Collect data from input fields
        customer_data = {}
        
        # Get values from text entries
        for field, entry in entries.items():
            try:
                # Convert numeric fields to appropriate types
                if field in ['customer_id', 'age', 'credit_score']:
                    customer_data[field] = int(entry.get())
                elif field in ['income', 'savings', 'debt']:
                    customer_data[field] = float(entry.get())
                else:
                    customer_data[field] = entry.get()
            except ValueError:
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, f"Error: Invalid value for {field}")
                return
        
        # Get values from dropdowns
        for field, var in dropdown_vars.items():
            customer_data[field] = var.get()
        
        # Get values from checkboxes
        for field, var in checkbox_vars.items():
            customer_data[field] = var.get()
        
        # Add a default market sentiment value
        customer_data['market_sentiment'] = 0.1
        customer_data['past_investment_returns'] = 0.06
        
        # Perform risk assessment
        result = agent.analyze_customer_risk(customer_data)
        
        # Display results
        result_text.delete(1.0, tk.END)
        
        # Format and display the results
        result_text.insert(tk.END, f"Customer ID: {result['customer_id']}\n")
        result_text.insert(tk.END, f"Analysis Date: {result['analysis_timestamp'][:10]}\n\n")
        
        result_text.insert(tk.END, f"RISK PROFILE: {result['risk_profile'].upper()}\n")
        result_text.insert(tk.END, f"Risk Score: {result['risk_score']:.1f}/100\n")
        result_text.insert(tk.END, f"Confidence: {result['confidence']:.2f}\n\n")
        
        result_text.insert(tk.END, "KEY FACTORS:\n")
        for factor in result['key_factors']:
            impact_symbol = "+" if factor['impact'] == "Positive" else "-"
            result_text.insert(tk.END, f"{impact_symbol} {factor['factor']}: {factor['description']}\n")
        
        result_text.insert(tk.END, "\nRECOMMENDATIONS:\n")
        for rec in result['recommendations']:
            result_text.insert(tk.END, f"• {rec['category']}: {rec['recommendation']}\n")
            result_text.insert(tk.END, f"  Rationale: {rec['rationale']}\n\n")
        
        if 'market_context' in result and result['market_context']:
            result_text.insert(tk.END, "\nMARKET CONTEXT:\n")
            for key, value in result['market_context'].items():
                result_text.insert(tk.END, f"• {key.replace('_', ' ').title()}: {value}\n")
    
    # Analyze button
    analyze_button = ttk.Button(input_frame, text="Analyze Risk Profile", command=analyze_risk)
    analyze_button.grid(row=len(fields)+len(dropdown_fields)+len(checkbox_fields), column=0, columnspan=2, pady=10)
    
    # Run the demo application
    root.mainloop()

class FinancialNewsMonitor:
    """
    Class to monitor financial news and extract sentiment relevant to investment decisions
    """
    def __init__(self, config=None):
        """Initialize the financial news monitor"""
        self.config = config or {}
        self.sources = self.config.get('news_sources', [
            'Bloomberg', 'Reuters', 'Wall Street Journal', 'Financial Times'
        ])
        self.topics = self.config.get('topics', [
            'interest rates', 'inflation', 'recession', 'stock market', 
            'federal reserve', 'housing market', 'unemployment'
        ])
        self.sentiment_model = None
        self.tokenizer = None
        self.initialize_sentiment_model()
        
    def initialize_sentiment_model(self):
        """Initialize the sentiment analysis model"""
        try:
            model_name = "finbert-sentiment"  # Or another appropriate financial sentiment model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            logging.info(f"Sentiment model loaded: {model_name}")
        except Exception as e:
            logging.error(f"Failed to load sentiment model: {e}")
            # Fallback to rule-based sentiment analysis
            
    def fetch_latest_news(self, limit=10):
        """
        Fetch latest financial news
        In a real implementation, this would connect to news APIs or RSS feeds
        """
        logging.info(f"Fetching latest news from {len(self.sources)} sources")
        
        # Demo implementation with mock news data
        mock_news = [
            {
                "title": "Federal Reserve Raises Interest Rates by 25 Basis Points",
                "source": "Bloomberg",
                "date": "2025-02-28",
                "url": "https://www.bloomberg.com/news/articles/2025-02-28/fed-raises-rates",
                "content": "The Federal Reserve raised interest rates by 25 basis points today, citing concerns about inflation."
            },
            {
                "title": "Bank of Canada Holds Rates Steady Amid Economic Uncertainty",
                "source": "Reuters",
                "date": "2025-03-01",
                "url": "https://www.reuters.com/markets/canada/bank-canada-holds-rates-steady",
                "content": "The Bank of Canada maintained its key interest rate at current levels, expressing caution about global economic uncertainties."
            },
            {
                "title": "TSX Reaches New Record High as Technology Stocks Surge",
                "source": "Financial Post",
                "date": "2025-03-02",
                "url": "https://www.financialpost.com/markets/tsx-record-high",
                "content": "The Toronto Stock Exchange hit a new record high today as technology stocks surged on positive earnings reports."
            },
            {
                "title": "Canadian Housing Market Shows Signs of Cooling as Mortgage Rates Rise",
                "source": "Globe and Mail",
                "date": "2025-03-03",
                "url": "https://www.theglobeandmail.com/business/article-housing-market-cooling",
                "content": "The Canadian housing market is showing signs of cooling as higher mortgage rates impact buyer demand."
            },
            {
                "title": "TD Bank Reports Strong Quarter with Growth in Digital Banking",
                "source": "Financial Times",
                "date": "2025-03-04",
                "url": "https://www.ft.com/content/td-bank-earnings",
                "content": "TD Bank reported better-than-expected quarterly earnings, with significant growth in digital banking services."
            }
        ]
        
        return mock_news[:limit]
    
    def analyze_news_sentiment(self, news_items):
        """Analyze sentiment for a list of news items"""
        logging.info(f"Analyzing sentiment for {len(news_items)} news items")
        
        results = []
        for item in news_items:
            # Combine title and content for better sentiment analysis
            text = f"{item['title']}. {item['content']}"
            
            # Analyze sentiment
            if self.sentiment_model and self.tokenizer:
                # Use pre-trained model
                sentiment = self._analyze_with_model(text)
            else:
                # Fallback to rule-based approach
                sentiment = self._rule_based_sentiment(text)
            
            # Add sentiment to the news item
            item_with_sentiment = item.copy()
            item_with_sentiment['sentiment'] = sentiment
            results.append(item_with_sentiment)
        
        return results
    
    def _analyze_with_model(self, text):
        """Analyze sentiment using the pre-trained model"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
        
        # Get probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        # Get predicted class (negative, neutral, positive)
        predicted_class = torch.argmax(probs, dim=1).item()
        
        # Map to sentiment labels
        sentiment_labels = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment = sentiment_labels.get(predicted_class, "neutral")
        
        return {
            "label": sentiment,
            "score": probs[0][predicted_class].item(),
            "breakdown": {
                "negative": probs[0][0].item(),
                "neutral": probs[0][1].item(),
                "positive": probs[0][2].item()
            }
        }
    
    def _rule_based_sentiment(self, text):
        """Simple rule-based sentiment analysis fallback"""
        text = text.lower()
        
        # Simple keyword-based approach
        positive_words = ['growth', 'surge', 'increase', 'gain', 'higher', 'record', 'strong', 'positive', 'bullish', 'optimistic']
        negative_words = ['decline', 'fall', 'drop', 'decrease', 'lower', 'weak', 'negative', 'bearish', 'pessimistic', 'concern']
        
        # Count occurrences
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        # Determine sentiment
        if positive_count > negative_count:
            sentiment = "positive"
            score = 0.5 + min(0.5, (positive_count - negative_count) / 10)
        elif negative_count > positive_count:
            sentiment = "negative"
            score = 0.5 - min(0.5, (negative_count - positive_count) / 10)
        else:
            sentiment = "neutral"
            score = 0.5
        
        # Create a similar structure to the model-based output
        return {
            "label": sentiment,
            "score": score,
            "breakdown": {
                "negative": negative_count / max(1, negative_count + positive_count),
                "neutral": 0.2,  # Arbitrary neutral component
                "positive": positive_count / max(1, negative_count + positive_count)
            },
            "method": "rule-based"  # Indicate this was rule-based
        }
    
    def get_market_sentiment_summary(self):
        """Get an overall market sentiment summary based on recent news"""
        # Fetch and analyze news
        news_items = self.fetch_latest_news(limit=10)
        analyzed_news = self.analyze_news_sentiment(news_items)
        
        # Calculate overall sentiment
        sentiment_scores = [item['sentiment']['score'] for item in analyzed_news]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5
        
        # Categorize the sentiment
        if avg_sentiment > 0.6:
            overall_sentiment = "positive"
        elif avg_sentiment < 0.4:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"
        
        # Group news by topic
        topics = {}
        for item in analyzed_news:
            for topic in self.topics:
                if topic in item['title'].lower() or topic in item['content'].lower():
                    if topic not in topics:
                        topics[topic] = []
                    topics[topic].append(item)
        
        # Calculate sentiment by topic
        topic_sentiment = {}
        for topic, items in topics.items():
            scores = [item['sentiment']['score'] for item in items]
            avg_score = sum(scores) / len(scores) if scores else 0.5
            if avg_score > 0.6:
                topic_sentiment[topic] = "positive"
            elif avg_score < 0.4:
                topic_sentiment[topic] = "negative"
            else:
                topic_sentiment[topic] = "neutral"
        
        return {
            "overall_sentiment": overall_sentiment,
            "average_score": avg_sentiment,
            "topic_sentiment": topic_sentiment,
            "analyzed_news_count": len(analyzed_news),
            "most_recent_date": max(item['date'] for item in news_items) if news_items else None
        }

class RegulatoryComplianceChecker:
    """
    Class to check if investment recommendations comply with regulatory requirements
    """
    def __init__(self, config=None):
        """Initialize the regulatory compliance checker"""
        self.config = config or {}
        self.regulations = self.config.get('regulations', [
            # Canadian regulations
            "KYC (Know Your Client)",
            "KYP (Know Your Product)",
            "Suitability Obligation",
            "Fee Disclosure",
            "Conflict of Interest Disclosure"
        ])
        self.risk_thresholds = self.config.get('risk_thresholds', {
            'low': 30,
            'medium': 60,
            'high': 90
        })
    
    def check_recommendation_compliance(self, customer_data, recommendation):
        """Check if a recommendation complies with regulatory requirements"""
        compliance_issues = []
        
        # Check KYC compliance - verify we have essential customer information
        required_kyc_fields = ['age', 'income', 'risk_tolerance']
        missing_kyc = [field for field in required_kyc_fields if field not in customer_data]
        if missing_kyc:
            compliance_issues.append({
                "regulation": "KYC (Know Your Client)",
                "issue": f"Missing required customer information: {', '.join(missing_kyc)}",
                "severity": "High"
            })
        
        # Check suitability based on risk tolerance
        customer_risk_tolerance = customer_data.get('risk_tolerance', 'Medium')
        recommendation_risk = self._assess_recommendation_risk(recommendation)
        
        if customer_risk_tolerance == 'Low' and recommendation_risk > self.risk_thresholds['low']:
            compliance_issues.append({
                "regulation": "Suitability Obligation",
                "issue": "Recommendation risk exceeds client's low risk tolerance",
                "severity": "High"
            })
        elif customer_risk_tolerance == 'Medium' and recommendation_risk > self.risk_thresholds['medium']:
            compliance_issues.append({
                "regulation": "Suitability Obligation",
                "issue": "Recommendation risk exceeds client's medium risk tolerance",
                "severity": "Medium"
            })
        
        # Check age-appropriate recommendations
        age = customer_data.get('age', 40)
        if age > 60 and recommendation_risk > self.risk_thresholds['medium']:
            compliance_issues.append({
                "regulation": "Suitability Obligation",
                "issue": "High-risk recommendation may not be suitable for client's age (over 60)",
                "severity": "Medium"
            })
        
        # Check income-appropriate recommendations
        income = customer_data.get('income', 70000)
        if income < 50000 and 'high' in recommendation.get('category', '').lower():
            compliance_issues.append({
                "regulation": "Suitability Obligation",
                "issue": "High-cost recommendation may not be suitable for client's income level",
                "severity": "Medium"
            })
        
        return {
            "compliant": len(compliance_issues) == 0,
            "issues": compliance_issues
        }
    
    def _assess_recommendation_risk(self, recommendation):
        """Assess the risk level of a recommendation"""
        # Simplified risk assessment based on keywords in the recommendation
        category = recommendation.get('category', '').lower()
        rec_text = recommendation.get('recommendation', '').lower()
        
        # Higher risk keywords
        high_risk_keywords = ['aggressive', 'growth', 'equities', '80%', '90%', 'speculative']
        medium_risk_keywords = ['balanced', 'moderate', '60%', '50%', 'mix']
        low_risk_keywords = ['conservative', 'income', 'preservation', 'fixed income', 'bonds', '30%', '20%']
        
        # Count occurrences
        high_risk_count = sum(1 for keyword in high_risk_keywords if keyword in rec_text)
        medium_risk_count = sum(1 for keyword in medium_risk_keywords if keyword in rec_text)
        low_risk_count = sum(1 for keyword in low_risk_keywords if keyword in rec_text)
        
        # Calculate weighted risk score
        risk_score = (high_risk_count * 3 + medium_risk_count * 2 + low_risk_count * 1) * 10
        
        # Adjust based on recommendation category
        if 'asset allocation' in category:
            risk_score += 10  # Higher baseline for investment recommendations
        elif 'protection' in category or 'insurance' in category:
            risk_score -= 10  # Lower for protection products
        
        return min(100, max(0, risk_score))
    
    def validate_assessment(self, customer_data, risk_assessment):
        """Validate a complete risk assessment for regulatory compliance"""
        issues = []
        
        # Check if we have proper KYC information
        required_kyc_fields = ['age', 'income', 'risk_tolerance']
        missing_kyc = [field for field in required_kyc_fields if field not in customer_data]
        if missing_kyc:
            issues.append({
                "regulation": "KYC (Know Your Client)",
                "issue": f"Missing required customer information: {', '.join(missing_kyc)}",
                "severity": "High"
            })
        
        # Check recommendations compliance
        for recommendation in risk_assessment.get('recommendations', []):
            result = self.check_recommendation_compliance(customer_data, recommendation)
            if not result['compliant']:
                issues.extend(result['issues'])
        
        # Check overall risk profile suitability
        customer_risk_tolerance = customer_data.get('risk_tolerance', 'Medium')
        risk_profile = risk_assessment.get('risk_profile', 'Moderate')
        
        # Map risk tolerance to expected profiles
        tolerance_to_profile = {
            'Low': ['Conservative'],
            'Medium': ['Conservative', 'Moderate'],
            'High': ['Conservative', 'Moderate', 'Aggressive']
        }
        
        if risk_profile not in tolerance_to_profile.get(customer_risk_tolerance, []):
            issues.append({
                "regulation": "Suitability Obligation",
                "issue": f"Risk profile '{risk_profile}' may not align with client's stated risk tolerance '{customer_risk_tolerance}'",
                "severity": "Medium"
            })
        
        return {
            "compliant": len(issues) == 0,
            "issues": issues
        }

if __name__ == "__main__":
    # Example usage
    demo_financial_risk_agent()
    
    # Alternatively, run the demonstration UI
    # create_demonstration_ui()