# config.py
"""
Configuration file for the Telecom Churn Dashboard
TODO: Move sensitive information to environment variables
"""

# Data Configuration
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "models/"
EXPORT_PATH = "exports/"

# Model Parameters
MODEL_PARAMS = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5
    },
    "gradient_boosting": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5
    },
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000
    }
}

# Business Rules
RISK_THRESHOLDS = {
    "high": 0.7,
    "medium": 0.4,
    "low": 0.0
}

# Notification Settings
ALERT_CHANNELS = {
    "email": {
        "smtp_server": "smtp.example.com",
        "port": 587
    },
    "slack": {
        "webhook_url": "https://hooks.slack.com/services/..."
    }
}

# Dashboard Configuration
DASHBOARD_SETTINGS = {
    "refresh_interval": 300,  # seconds
    "max_records_display": 10000,
    "cache_ttl": 3600  # seconds
}

# TODO: Add more configuration as needed