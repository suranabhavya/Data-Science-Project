# Data-Science-Project

## Mid-Term Presentation
https://youtu.be/n2QhbOy2V7o 

## Boston Housing Violations Analysis
### Description
The project's aim is to analyze different violations around the city of Boston and identify systemic issues and trends.
Using violation records from the given datasets, we aim to provide actionable insights to improve housing quality.

### Dataset(s)

- [Public Works Violations Dataset](https://data.boston.gov/dataset/public-works-violations)
- [Building and Property Violations Dataset](https://data.boston.gov/dataset/building-and-property-violations1)
- [Property Assessment Dataset](https://data.boston.gov/dataset/property-assessment)
- [Live Street Address Management Dataset](https://data.boston.gov/dataset/live-street-address-management-sam-addresses)

### Project Blueprint: Making Sense of the Notebooks

* The 'Public Works Violations' dataset can be cleaned using PWV_Cleaning.ipynb notebook. This will generate a processed file named PWV_processed.csv.

* The generated PWV_processed.csv file should be loaded to TDS_Midterm_Clustering_EDA_6.ipynb to replicate our analysis and find answers to key questions discussed in "Insights" section. 

* The location plots can be created for visualization by running DS_Midterm_Clustering_EDA_6.ipynb and boston_violations_map.ipynb files with the same processed data.

* Download the 'Building and Property Violations' dataset and run it through the Property_Violations.ipynb notebook to obtain a preprocessed dataset and draw insights of data distribution.

* Our baseline model results can be reproduced using the init_pred_modelling.ipynb file.

### Data preprocessing
We performed data cleaning and preprocessing for three different datasets corresponding to different analytical objectives.

1. Building and Property Violations

    * We began by checking for null values in the dataset. The columns 'value', 'violation_sthigh', and 'contact_addr2' were dropped as they contained over 70% missing values.

    * Next, we then imputed the null values in 'Description' column with 'Unknwown' value. This column provides valuable information about the type of violation, so we opted to retain it rather than drop the Null records. This allowed us to retain around 50% of the rows with atleast one null value.

    * We dropped the remaining rows with missing values, around 300 out of 16591 entries. 

    * Duplicate rows (16 in total) were also identified and removed.
    
    * Additionally, we dropped the 'status_dttm' column, which contained timestamp information that was not relevant for our current analysis, and the 'location' column due to information redundancy.
    
    * Finally, we applied binary encoding to the 'status' column for further processing.

2. Public Works Violations Dataset
    * The preprocessing steps for this dataset were mostly similar to those applied to the Building and Property Violations dataset, as both share a similar structure but contain different values.

    * We began by dropping several attributes, namely 'violation_sthigh', 'contact_addr2', 'violation_state', 'contact_zip', 'contact_state', 'contact_city', 'violation_suffix' and 'ticket_no'. Some of these had a significant number of missing values, while others were deemed redundant at this stage.

    * The 'description' and 'code' columns contained subcategories, so we consolidated into broader categories to simplify the dataset.

            Before:
            10a -> ['Illegal dumping park']
        
            10b -> ['Illegal dumping 1-5 cuyd.']
        
            10c -> ['Illegal dumping 5 cubic yd.']

            After:
            10 -> ['Illegal dumping']
    
3. Public Assessment Dataset

    * For this dataset, our current goal was to explore whether building characteristics could be used to train a classifier to predict the likelihood of a violation.

    * To enable this, we merged the Public Works Violations dataset with the Property Assessment dataset. The linkage was established through the Street Address Management dataset. 
    
    * The attribute 'sam_id' from the Public Works dataset corresponds to 'SAM_ADDRESS_ID' in the Street Address Management dataset. The attribute 'PARCEL' from the Street Address Management dataset corresponds to 'PID' in the Property Assessment dataset. Based on this connection we added new attribute to Property Assessment Dataset, 'violation_bool' has value 1 if the corresponding 'PID' has a violation recorded, else 0.

    * As part of standard data cleaning process, we dropped attributes with a considerable amount of null values. We also removed location-based attributes which were not useful for current analysis objectives.

    * Finally, we imputed missing values for key attributes such as 'FULL_BTH', 'HLF_BTH', 'KITCHENS', and 'FIREPLACES', which we considered potentially important for predicting violations. Any remaining rows with null values were then removed.



### Current Insights

1. We were able to answer a few key questions regarding the Public Works Violations dataset:
    
    * Q: Which address has the most repeated violations? A: 74 Clarendon ST Suite A
    
        ![Properties with violations](images/add_with_rep_violations.png)

    * Q: Which zipcode has the most violations? A: 02127
    
        ![Zipcodes with violations](images/zipcode_pwv.png)    

    * Q: Which is the most common violation? A: Improper trash disposal

        ![Top violations](images/top_Complaints_pwv.png)


2. As part of our analysis, we explored clustering the violation descriptions from the Public Works Violations dataset into broader categories as a step towards downstream multiclass classification. Although the results were not definitive, hierarchical clustering applied to the t-SNE components of the description embeddings produced the most promising grouping results.

3. The preliminary analysis on the Building and Property Violations dataset revealed that the majority of violations were related to "mechanical execution of work".

![Top violations in Building and Property](images/top_Complaints_pnbv.png)

    Additionally, Dorchester emerged as the neighborhood with the highest number of recorded violations.

![Top violation cities in Building and Property](images/top_violationCities_pnbv.png)

4. Heatmaps of building violations offer valuable visual insights into spatial distribution patterns, helping identify hotspots where violations are most concentrated. By grouping violation data by geographic coordinates and applying a weighted color gradient, we can quickly pinpoint areas that may require urgent attention.

    * In our analysis, we began by preprocessing the data to ensure the accuracy of latitude and longitude values. We then aggregated the records by location and computed the number of violations at each point. 
    * Using the Folium library, we overlaid a heatmap on a base map of Boston to visualize violation density across the city.
    * This approach revealed both citywide trends and specific high-risk areas. Notably, neighborhoods such as Dorchester and Roxbury emerged as significant hotspots, indicating potential long-term compliance issues in these locations. These insights can inform targeted enforcement efforts and guide the more efficient allocation of resources for improving building safety.

![Heatmap](images/heatmap.png)

![Pinpoint Info](images/pinpoint_info.png)

5. We attempted to visualize violations by mapping them to their geographic locations to identify whether certain areas had a higher concentration of specific types of violations.

    The spatial distribution appeared uniform as violations of all types seemed to occur across all regions. When all violation types were plotted together, trash-related violations overwhelmingly dominated the map due to their high frequency. When plotted separately by category, the violations either appeared too sparse to identify patterns or too dense to differentiate meaningfully.

### Preliminary results
Using our modified Public Assessment dataset, we trained a Random Forest Classifier, achieving an accuracy of 86% and a weighted F1 score of 0.83. However, the precision and recall for the violations class were relatively low, likely due to the skewed distribution of the dataset.

To address this issue, our next steps involve experimenting with techniques to mitigate class imbalance. This includes exploring undersampling/oversampling methods or implementing weighted sampling strategies. Additionally, we plan to evaluate XGBoost with class weights to potentially improve performance on the minority class.


### Next Steps

1. We have tried to answer most of the base questions from the original project's document. But our insights feel a little incohesive at the moment. We need to try to get solid trends from the data.

2. Using the 'status_dttm', we should be able to get time data for the violations, using which we could get better insights like recurrence of violation, a specific rise in violations etc.

3. See if we can find any other target variables that we can predict from the data available.

4. We are currently hindered by our lack of domain knowledge to impute a lot a null values. We need to find better ways.

5. We need to do a better clean up of data's location attributes using Street Address Management dataset.

6. A better organization of code.
