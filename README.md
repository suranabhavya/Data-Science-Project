# Data-Science-Project

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
     The Public Works Violation and Street Address Management Dataset have a common attribute 'sam_id' == 'SAM_ADDRESS_ID'. Street Address Management dataset and Property Assessment Dataset have a common attribute ('PARCEL' == 'PID'). So based on this connection we added new attribute to Property Assessment Dataset, 'violation_bool' which is 1 if the corresponding 'PID' has a violation recorded, else 0.

    * We then did the usual data cleaning. We dropped attributes with a considerable amount of null values. We also then dropped the location based attributes, because our experiments on location didnt provide us with any insights (discussed in the next section).

    * We then imputed values for 'FULL_BTH', 'HLF_BTH', 'KITCHENS', 'FIREPLACES', as they seemed important for predicting violations, and then dropped the remaining null value rows.  



### Current Insights

1. We tried plotting the violations to their locations to see if we can get any insights about a particular area having an increased number of violations of a particular category. This didn't give us the results we hoped we would get.

It just looked like all violations take place in all the places. When plotted together, trash related violations dominate the whole map as there are in a majority. When tried plotting separately, they were either too sparse or too dense.

2. We then tried to answer a couple of questions regarding the Public Works Violations dataset:
    
    * Q: Which address has the most repeated violations? A: 74 Clarendon ST Suite A
        ![Properties with violations](images/add_with_rep_violations.png)

    * Q: Which zipcode has the most violations? A: 02127
        ![Zipcodes with violations](images/zipcode_pwv.png)    

    * Q: Which is the most common violation? A: Improper trash disposal

        ![Top violations](images/top_Complaints_pwv.png)



3. We also tried clustering different violation descriptions in this dataset into categories we could use for something like a multiclass classification. This wasn't perfect but using hierarchical clustering on the t-SNE components of the descriptions' embeddings gave us the best result for now.

4. We also did some initial analysis on Building and Property Works Violations dataset, we observed that most of the violations were related to 'mechanical execution of work' and Dorchester was the place with most violations.

![Top violations in Building and Property](images/top_Complaints_pnbv.png)


![Top violation cities in Building and Property](images/top_violationCities_pnbv.png)

5. Heatmaps of building violations provide a powerful visual insight into spatial patterns and help identify hotspots where violations are most frequent. By grouping violation data based on geographic coordinates and applying a weighted color gradient, we can quickly discern areas that require urgent attention. In our analysis, we preprocessed the data by ensuring accurate latitude and longitude values, grouped the records by location, and calculated the number of violations at each point. We then used the Folium library to overlay a heatmap on a base map of Boston, highlighting areas with high densities of violations. This method not only reveals overall trends but also enables us to focus on specific addresses; for instance, locations like "Dorchester", "Roxbury" emerged as significant hotspots, suggesting that these buildings or neighborhoods might be facing chronic compliance issues. This location-based insight supports targeted enforcement and more efficient allocation of resources for building safety improvements.

![Heatmap](images/heatmap.png)

![Pinpoint Info](images/pinpoint_info.png)

### Preliminary results
On our modified public assessment dataset, we tried to fit a RandomForestClassifer, where we achieved a decent result of 86% accuracy and a weighted F1 score of 0.83. But the precision and recall for the violations class were pretty low, which can be explained by the skewed distribution of the dataset.

Our goal would be to rectify the class imbalance by using undersampling techinques or some kind of weighted sampling technique. We also want to try using XGBoost with weighted classes.


### Next Steps

1. We have tried to answer most of the base questions from the original project's document. But our insights feel a little incohesive at the moment. We need to try to get solid trends from the data.

2. Using the 'status_dttm', we should be able to get time data for the violations, using which we could get better insights like recurrence of violation, a specific rise in violations etc.

3. See if we can find any other target variables that we can predict from the data available.

4. We are currently hindered by our lack of domain knowledge to impute a lot a null values. We need to find better ways.

5. We need to do a better clean up of data's location attributes using Street Address Management dataset.

6. A better organization of code.
