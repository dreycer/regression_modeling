# Kaggle Challenge: Ames Housing Data

## Danielle Reycer
---

## Problem Statement

The ability to appropriately and accurately predict sale prices of homes is incredibly beneficial in a market that is currently oversaturated with buyers. To gain an edge in the market, it would be helpful to know the approximate final selling price of a home before putting in an offer or putting a home on the market.

As a leader in Iowa real estate, we aim to accurately predict the Sale Prices for homes in Ames, Iowa through data cleaning and creating models using Multiple Linear Regression, Lasso, and Ridge. We will evaluate success by interpreting our R^2 score and the Root Mean Squared Error for each model as well as examining whether the model has an appropriate number of variables.

---

## Background

Ames is a city in Story County, Iowa, United States, located approximately 30 miles (48 km) north of Des Moines in central Iowa. It is best known as the home of Iowa State University (ISU), with leading agriculture, design, engineering, and veterinary medicine colleges. A United States Department of Energy national laboratory, Ames Laboratory, is located on the ISU campus.

In 2019, Ames had a population of 66,258. Iowa State University was home to 33,391 students as of fall 2019, which make up approximately one half of the city's population.

The Ames Housing Dataset was introduced by Professor Dean De Cock in 2011 as an alternative to the Boston Housing Dataset (Harrison and Rubinfeld, 1978). It contains 2,919 observations of housing sales in Ames, Iowa between 2006 and 2010. There are 23 nominal, 23 ordinal, 14 discrete, and 20 continuous features describing each house’s size, quality, area, age, and other miscellaneous attributes.

Sources: [Wikipedia](https://en.wikipedia.org/wiki/Ames,_Iowa) and [NYC Data Science Academy](https://nycdatascience.com/blog/student-works/machine-learning/machine-learning-project-ames-housing-dataset/)

---

## My Notebooks

- [01_cleaning_eda_and_modeling](https://git.generalassemb.ly/dreycer/project_2/blob/master/code/01_cleaning_eda_and_modeling.ipynb)
- [02_scaling_and_modeling](https://git.generalassemb.ly/dreycer/project_2/blob/master/code/02_scaling_and_modeling.ipynb)
- [03_engineering_features_extension](https://git.generalassemb.ly/dreycer/project_2/blob/master/code/03_engineering_features_extension.ipynb)


---

## Data

### Data

* [`train.csv`](../datasets/train.csv): contains all of the training data for my model.
* [`train_clean1.csv`](../datasets/train_clean1.csv): After cleaning the dataset, it was saved in order to use it in future notebooks.
* [`train_imputed1.csv`](../datasets/train_imputed1.csv): This CSV contains many imputed values. It was saved separately so as to not corrupt the original data.
* [`test.csv`](../datasets/test.csv): contains the test data for my model. This is the data from which to build the Kaggle submission. Note: the `SalePrice` column has been removed from this dataset. 
* [`test_clean1.csv`](../datasets/test_clean1.csv): After cleaning the dataset, it was saved in order to use it in future notebooks.
* [`test_imputed1.csv`](../datasets/test_imputed1.csv): This CSV contains many imputed values. It was saved separately so as to not corrupt the original data.

### Data Dictionary

|Feature|Type|Description|
|---|---|---|
|**pid**|*integer*|Parcel identification number  - can be used with city web site for parcel review| 
|**ms_subclass**|*integer*|Identifies the type of dwelling involved in the sale|
|**ms_zoning**|*object*|Identifies the general zoning classification of the sale| 
|**lot_frontage**|*float*|Linear feet of street connected to property|
|**lot_area**|*float*|Lot size in square feet| 
|**street**|*object*|Type of road access to property| 
|**alley**|*object*|Type of alley access to property|
|**lot_shape**|*object*|General shape of property| 
|**land_contour**|*object*|Flatness of the property|
|**utilities**|*object*|Type of utilities available| 
|**lot_config**|*object*|Lot configuration| 
|**land_slope**|*object*|Slope of property|
|**neighborhood**|*object*|Physical locations within Ames city limits| 
|**condition_1**|*object*|Proximity to various conditions|
|**condition_2**|*object*|Proximity to various conditions (if more than one is present)| 
|**bldg_type**|*object*|Type of dwelling| 
|**house_style**|*object*|Style of dwelling|
|**overall_qual**|*integer*|Rates the overall material and finish of the house| 
|**overall_cond**|*integer*|Rates the overall condition of the house|
|**year_built**|*integer*|Original construction date| 
|**year_remod/add**|*integer*|Remodel date (same as construction date if no remodeling or additions)| 
|**roof_style**|*object*|Type of roof|
|**roof_matl**|*object*|Roof material| 
|**exterior_1**|*object*|Exterior covering on house| 
|**exterior_2**|*object*|Exterior covering on house (if more than one material)|
|**mas_vnr_type**|*object*|Masonry veneer type| 
|**mas_vnr_area**|*float*|Masonry veneer area in square feet|
|**exter_qual**|*object*|Evaluates the quality of the material on the exterior| 
|**exter_cond**|*object*|Evaluates the present condition of the material on the exterior| 
|**foundation**|*object*|Type of foundation|
|**bsmt_cond**|*object*|Evaluates the general condition of the basement| 
|**bsmt_exposure**|*object*|Refers to walkout or garden level walls|
|**bsmtfin_type_1**|*object*|Rating of basement finished area| 
|**bsmtfin_sf_1**|*float*|Type 1 finished square feet| 
|**bsmtfintype_2**|*object*|Rating of basement finished area (if multiple types)|
|**bsmtfin_sf_2**|*float*|Type 2 finished square feet| 
|**bsmt_unf_sf**|*float*|Unfinished square feet of basement area|
|**total_bsmt_sf**|*float*|Total square feet of basement area| 
|**heating**|*object*|Type of heating| 
|**heating_qc**|*object*|Heating quality and condition|
|**central_air**|*object*|Central air conditioning| 
|**electrical**|*object*|Electrical system| 
|**1st_flr_sf**|*float*|First Floor square feet|
|**2nd_flr_sf**|*float*|Second Floor square feet| 
|**low_qual_fin_sf**|*float*|Low quality finished square feet (all floors)|
|**gr_liv_area**|*float*|Above grade (ground) living area square feet| 
|**bsmt_full_bath**|*integer*|Basement full bathrooms| 
|**bsmt_half_bath**|*integer*|Basement half bathrooms|
|**full_bath**|*integer*|Full bathrooms above grade| 
|**half_bath**|*integer*|Half baths above grade|
|**bedroom**|*integer*|Bedrooms above grade (does NOT include basement bedrooms)| 
|**kitchen**|*integer*|Kitchens above grade| 
|**kitchen_qual**|*object*|Kitchen quality|
|**tot_rms_abv_grd**|*integer*|Total rooms above grade (does not include bathrooms)| 
|**functional**|*object*|Home functionality (Assume typical unless deductions are warranted)|
|**fireplaces**|*integer*|Number of fireplaces| 
|**fireplace_qu**|*object*|Fireplace quality| 
|**garage_type**|*object*|Garage location| 
|**garage_yr_blt**|*integer*|Year garage was built|
|**garage_finish**|*object*|Interior finish of the garage| 
|**garage_cars**|*integer*|Size of garage in car capacity|
|**garage_area**|*float*|Size of garage in square feet| 
|**garage_qual**|*object*|Garage quality|
|**garage_cond**|*object*|Garage condition| 
|**paved_drive**|*object*|Paved driveway|
|**wood_deck_sf**|*float*|Wood deck area in square feet| 
|**open_porch_sf**|*float*|Open porch area in square feet| 
|**enclosed_porch**|*float*|Enclosed porch area in square feet| 
|**3-ssn_porch**|*float*|Three season porch area in square feet|
|**screen_porch**|*float*|Screen porch area in square feet| 
|**pool_area**|*float*|Pool area in square feet|
|**pool_qc**|*object*|Pool quality| 
|**fence**|*object*|Fence quality|
|**misc_feature**|*object*|Miscellaneous feature not covered in other categories| 
|**misc_val**|*float*|$Value of miscellaneous feature| 
|**mo_sold**|*integer*|Month Sold (MM)| 
|**yr_sold**|*integer*|Year Sold (YYYY)|
|**sale_type**|*object*|Type of sale| 
|**sale_condition**|*object*|Condition of sale|
|**saleprice**|*float*|Sale price $$| 


|Engineered Feature|Type|Description|
|---|---|---|
|**all_baths**|*float*|Sum of all bathrooms|
|**huge_lot**|*integer*|Whether or not a house was an outlier in regards to 'lot_size'| 
|**recession**|*integer*|Whether or not the house was sold during the US recession|
|**kitchen_qual_ord**|*integer*|Assigned ordinal values to 'kitchen_qual'| 
|**heating_qc_ord**|*integer*|Assigned ordinal values to 'heating_qc'| 
|**exter_qual_ord**|*integer*|Assigned ordinal values to 'exter_qual'| 
|**overall_qual^2**|*integer*|Polynomial feature, giving more weight to higher overall quality ratings| 

Note: a full data dictionary, with all possible values within each feature can be [found here](http://jse.amstat.org/v19n3/decock/DataDocumentation.txt)


---

## Conclusions

My final model that I propose this Iowa Real Estate Agency uses is a linear regression model with 8 features. (Overall Quality ^2, Year Built, Total Basement Square Feet, 1st Floor Square Feet, Above Ground Living Area (sqft), Neighboorhood scalar, Kitchen Quality (ordinal value - 1-5 scale)). 

I greatly limited the number of variables in my final model due to the fact that we only had a data set of around 2,000 points. I also needed the model to be usable for the real estate agents. Collecting and entering 8 values in a model is a reasonable ask to determine the sale price of home with an average error of around $30,000. I would not recommend my more complicated models even though they had a slightly lower average error. In order to improve the error significantly, the model would require upwards of 24 variables/features. 

My final model also performed well on unseen data, which is an indication that it would be useful in its predictive power. 

As a final product, my equation was provided to the realty company, as well as my neighborhood scalar list. Real estate agents would then be able to calculate predicted sale price by hand. However, I agreed to work with the team further to develop an internal app so agents could calculate the value simply by entering the 7 pieces of numerical information needed along with choosing a neighborhood from a drop-down menu. 

![Equation for Predicted Sale Price](https://git.generalassemb.ly/dreycer/project_2/blob/master/images_and_presentation/equation.png)

![Neighborhood Scalars](https://git.generalassemb.ly/dreycer/project_2/blob/master/images_and_presentation/neighborhood_scalars.png)








