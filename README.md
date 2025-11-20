# Analyzing the Relationship Between Environmental and Clinical Risk Factors and National GDP

**Authors:**  
Siddhant Saxena, Joanna Jipson, Nhu Le, Justin Li  
**Institution:**   
University of North Carolina at Chapel Hill (COMP 560)  
**Address**:  
232 S Columbia St, Chapel Hill, NC 27514  
**Emails:**  
sisaxena@unc.edu, jjips@ad.unc.edu, nhule@unc.edu, jql@ad.unc.edu

***

## Abstract

This project examines how national environmental risks and clinical health statistics relate to economic performance, measured by country-level GDP. Using a consolidated dataset of environmental, health, demographic, and economic indicators, we train linear regression, logistic regression, ridge regression, and stochastic gradient descent models to identify key predictors of GDP. We then use the Anthropic API to summarize model outputs and highlight the most important patterns and associations.

***

## 1. Introduction

This project examines how national environmental risk factors and clinical health statistics relate to economic performance, measured through country level GDP. Traditional economic models focus on capital, labor, and institutions, but growing evidence suggests that environmental pressures and population health also influence long term development. Using a consolidated dataset combining environmental, clinical, demographic, and GDP indicators, we train several supervised learning models and use the Anthropic API to interpret the resulting patterns.

## 2. Preparing Data

We merged three datasets covering the years 1990 to 2009:

* WHO risk factor mortality data (subset of 15 selected variables)
* Population data from HYDE, Gapminder, and UN WPP
* GDP data from WITS

Because the datasets used different naming conventions, all country names were standardized before merging. We removed territories, regions, and very small or incomplete populations to improve data quality. The final dataset contains 35 variables and 1,981 country year rows.

***

## 3. Methods

We implemented three supervised learning models:

Linear regression with stochastic gradient descent
Used to estimate GDP assuming a linear relationship between predictors and economic output.

Logistic regression
Used to classify each country year as high or low GDP based on logistic loss.

Ridge regression
Applied as a regularized linear model to stabilize coefficient magnitudes.

The models were trained on a split of the dataset and evaluated using MSE, R squared, accuracy, and F1 scores. Coefficient magnitudes were examined to identify influential variables. To enhance interpretability, model outputs were formatted as JSON and summarized using the Anthropic Claude API, guided by a structured prompt to ensure academic clarity.

## 4. Results

We evaluated both regression and classification models using environmental and clinical features:

- **Regression (GDP as continuous):**  
  - Linear: MSE ≈ 2.14 × 10²³, R² ≈ 0.47  
  - Ridge: MSE ≈ 2.50 × 10²³, R² ≈ 0.37  
  - SGD: MSE ≈ 3.04 × 10²³, R² ≈ 0.24  
  → Simple linear regression performed best  
  → Across methods, generally stronger performance on higher-GDP countries than on low-GDP countries.

- **Classification (high vs low GDP):**  
  - Logistic (liblinear): accuracy ≈ 0.85, F1 ≈ 0.86  
  - Logistic (GD): accuracy ≈ 0.84, F1 ≈ 0.82  
  → Models reliably distinguish between high- and low-GDP countries.

Models using **combined environmental + clinical features** outperformed those using either group alone. LLM-based interpretation (Anthropic API) helped summarize which features were most influential and how they differed between high- and low-GDP profiles.

***

## 5. Discussion

During development, we encountered missing values, inconsistent formatting, and kernel instability that required additional cleaning and preprocessing. SGD based linear regression initially produced unstable outputs until learning rates and iteration counts were adjusted. Ridge regression provided more stable and interpretable coefficients. The Anthropic API summaries helped contextualize model behavior, but limitations remain: the analysis is correlational, lacks institutional and political variables, and does not capture dynamic changes over time. These constraints highlight the need for richer datasets and more advanced causal or time series models.

## 6. Conclusion

Environmental and clinical factors show meaningful statistical relationships with national GDP in our models. While the findings do not imply causation, they support the idea that public health and environmental conditions provide informative signals about economic outcomes. By combining regression based modeling with LLM supported interpretation, this project offers a foundation for future work exploring how global health and environmental vulnerabilities interact with long term economic development.

***

## References

* Varpit94. "Worldwide Deaths by Country/Risk Factors." Kaggle, n.d., www.kaggle.com/datasets/varpit94/worldwide-deaths-by-risk-factors
* "Sources of Our Population Dataset." Our World in Data, 15 July 2024, https://ourworldindata.org/grapher/sources-population-dataset
* World Bank. "GDP by Country in Current US$, 1988-2022 (Indicator: NY.GDP.MKTP.CD)." World Integrated Trade Solution (WITS), n.d., https://wits.worldbank.org/CountryProfile/en/country/by-country/startyear/ltst/endyear/ltst/indicator/NY-GDP-MKTP-CD
