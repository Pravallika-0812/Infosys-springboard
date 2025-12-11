Milestone 1 – Data Collection & Preprocessing

Overview
This milestone prepares the dataset for visa processing time prediction.
It includes collecting data, cleaning it, encoding categories, and creating the target variable.
Tasks Completed

1. Data Collection
Created a sample/historical visa dataset with 7 records.
Includes fields like visa type, dates, country, service center, age, priority, and status.

2. Missing Value Handling
Filled missing:
country → “Unknown”
service_center → “Unknown”
priority → “Regular”
applicant_age → median age
Converted application and decision dates to datetime.

3. Target Label Creation
Added processing_days:
decision_date - application_date
