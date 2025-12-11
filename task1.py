import numpy as np
import pandas as pd 
from datetime import datetime 
data = {
    "application_id": ["A201", "A202", "A203", "A204", "A205", "A206", "A207"],
    "visa_type": ["H1B", "F1", "B2", "H4", "L1A", "F2", "H1B"],
    "country": ["India", None, "Mexico", "India", "UK", "South Korea", "India"],
    "service_center": ["California", "Vermont", "Texas", "Nebraska", "Vermont", None, "Texas"],
    "application_date": ["2023-01-10", "2023-02-05", "2023-02-06", "2023-03-14","2023-01-20", "2023-04-01", "2023-02-15"],
    "decision_date": ["2023-04-12", "2023-03-09", "2023-02-25", "2023-09-23","2023-06-22", "2023-05-07", "2023-06-18"],
    "applicant_age": [28, 22, 36, 30, None, 24, 33],
    "priority": ["Regular", "Regular", "Premium", "Regular", "Premium", "Regular", None],
    "status": ["Approved", "Approved", "Approved", "Pending","Approved", "Approved", "Approved"],
}
df = pd.DataFrame(data)
print(pd)
df["application_date"] = pd.to_datetime(df["application_date"])
df["decision_date"] = pd.to_datetime(df["decision_date"])
print(df)
df["country"] = df["country"].fillna("Unknown")
df["service_center"] = df["service_center"].fillna("Unknown")
df["priority"] = df["priority"].fillna("Regular")
df["applicant_age"] = df["applicant_age"].fillna(df["applicant_age"].median())
df["processing_days"] = (df["decision_date"] - df["application_date"]).dt.days
df["processing_days"] = df["processing_days"].fillna(df["processing_days"].median())
print(df)