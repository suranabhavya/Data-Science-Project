# Data-Science-Project

## Boston Housing Violations Analysis
### Description
The project's aim is to analyze different violations around the city of Boston and identify systemic issues and trends.
Using violation records from the given datasets, we aim to provide actionable insights to improve housing quality.

### Goal(s)
1. Identify clusters of chronic housing vialoations and correlate them with:
- Property characteristics
- Neighbourhoods/Communitites
-  and any other common features.
2. Tax Prediction, Value Prediction, property condition clustering for Property assessment Dataset

### Data preprocessing
We performed data cleaning and preprocessing for three different datasets corresponding to different analytical objectives.

1. Building and Property Violations

    * We first check for null values and dropped the columns, 'value', 'violation_sthigh' and 'contact_addr2' as they had more than 70 percent null values.

    * Next, we then imputed 'description''s null values with 'Unknwown'.The description in this dataset contains the information about the type of violations We did not drop these as it was an important attribute and wanted to retain as much information as possible. We retained around 50% of the rows with null values by doing this.

    * We dropped the rest of the rows, around 300 out of 16591, which had null values. 

    * We then removed the duplicate rows. These were 16 rows.
    
    * We dropped status_dttm (information about the time of violation) and location(redundant information)
    
    * We also binary encoded status.

2. Public Works Violations Dataset
    * This is almost the same exact procedure as above. These two datasets have almost similar columns, with different values. 

    * We first dropped the attributes ('violation_sthigh', 'contact_addr2', 'violation_state', 'contact_zip', 'contact_state', 'contact_city', 'violation_suffix', 'ticket_no'). Out of these some had a considerable amount of null values while the rest seemed redundant at this moment.

    * The 'description' and 'code' had sub classes, so we clubbed them into a single one to make it simpler.

            Before:
            10a -> ['Illegal dumping park']
        
            10b -> ['Illegal dumping 1-5 cuyd.']
        
            10c -> ['Illegal dumping 5 cubic yd.']

            After:
            
            10 -> ['Illegal dumping']
    
3. Public Assessment Dataset

    * Here, we wanted to see if we can use the building details available to train a classifier to predict for a violation.

    * So we first concatenated the Public Works Violation Dataset to Property Assessment Dataset. The Public Works Violation and Street Address Management Dataset have a common attribute 'sam_id' == 'SAM_ADDRESS_ID'. Street Address Management dataset and Property Assessment Dataset have a common attribute ('PARCEL' == 'PID'). So based on this connection we added new attribute to Property Assessment Dataset, 'violation_bool' which is 1 if the corresponding 'PID' has a violation recorded, else 0.

    * We then did the usual data cleaning. We dropped attributes with a considerable amount of null values. We also then dropped the location based attributes, because our experiments on location didnt provide us with any insights (discussed in the next section).

    * We then imputed values for 'FULL_BTH', 'HLF_BTH', 'KITCHENS', 'FIREPLACES', as they seemed important for predicting violations, and then dropped the remaining null value rows.  



### Dataset(s)

- [Public Works Violations Dataset](https://data.boston.gov/dataset/public-works-violations)
- [Building and Property Violations Dataset](https://data.boston.gov/dataset/building-and-property-violations1)
- [Property Assessment Dataset](https://data.boston.gov/dataset/property-assessment)
- [Live Street Address Management Dataset](https://data.boston.gov/dataset/live-street-address-management-sam-addresses)
- Code Enforcement Violations (contains 8 lakh records across 60 violation description categories)


### Data Modelling
We have 

### Data Visualization
Current plans for visualization are:
- Heatmaps and 
- Scatterplots to explore relationships between variables.

### Test plans

The test plan is two try two different splits:
 1. Time based split for temporal analysis validations, and
 2. Randomized split for generalizability.

We plan to use machine learning algorithms such as Linear Regression, Hierarchical clustering, Random Forest, CatBoost, and XGBoost, along with data science analytics tools.

