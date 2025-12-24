import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# IMPORT DATA FROM TASK 1
# ===============================
from task1 import df

print("Initial Data:")
print(df.head())

# ===============================
# BASIC EDA
# ===============================
print("\nData Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ===============================
# CORRELATION ANALYSIS
# ===============================
plt.figure()
sns.heatmap(df[["applicant_age", "processing_days"]].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# ===============================
# PROCESSING DAYS BY VISA TYPE
# ===============================
plt.figure()
sns.boxplot(x="visa_type", y="processing_days", data=df)
plt.title("Processing Days by Visa Type")
plt.show()

# ===============================
# PROCESSING DAYS BY COUNTRY
# ===============================
plt.figure()
sns.barplot(x="country", y="processing_days", data=df)
plt.title("Average Processing Days by Country")
plt.xticks(rotation=30)
plt.show()

# ===============================
# FEATURE ENGINEERING
# ===============================

# 1. Month feature (Seasonality)
df["application_month"] = df["application_date"].dt.month

# 2. Seasonal Index
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

df["season"] = df["application_month"].apply(get_season)

# 3. Country-wise average processing time
country_avg = df.groupby("country")["processing_days"].mean()
df["country_avg_processing"] = df["country"].map(country_avg)

# ===============================
# FINAL DATASET
# ===============================
print("\nFinal Data with Engineered Features:")
print(df)

# ===============================
# VISUALIZE SEASONAL IMPACT
# ===============================
plt.figure()
sns.barplot(x="season", y="processing_days", data=df)
plt.title("Processing Days by Season")
plt.show()
#task2 submission