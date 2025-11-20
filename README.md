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

## 1. Abstract

This project examines how national environmental risks and clinical health statistics relate to economic performance, measured by country level GDP. Using a consolidated dataset of environmental, health, demographic, and economic indicators, we train linear regression, logistic regression, ridge regression, and stochastic gradient descent models to identify key predictors of GDP. We then use the Anthropic API to summarize model outputs and highlight major patterns.

***

## 2. Preparing Data

We merged three datasets for 1990 to 2009:

* WHO risk factor mortality data
* Global population estimates from HYDE, Gapminder, and UN WPP
* GDP data from WITS

Country names were standardized across sources, and we removed territories and very small or incomplete populations. The final dataset contains 35 variables and 1,981 country year rows.

***

## 3. Methods

We implemented three supervised learning models:

Linear regression with stochastic gradient descent

Logistic regression for high versus low GDP classification

Ridge regression as a regularized linear baseline

Models were trained on a train test split and evaluated using MSE, R squared, accuracy, and F1 score. Coefficient magnitudes were used to study feature influence. Model outputs were also formatted as JSON and summarized with the Anthropic Claude API.

## 4. Results

Regression (GDP as continuous):

Linear: MSE ≈ 2.14 × 10²³, R² ≈ 0.47

Ridge: MSE ≈ 2.50 × 10²³, R² ≈ 0.37

SGD: MSE ≈ 3.04 × 10²³, R² ≈ 0.24

Classification (high vs low GDP):

Logistic (liblinear): accuracy ≈ 0.85, F1 ≈ 0.86

Logistic (gradient descent): accuracy ≈ 0.84, F1 ≈ 0.82

Generally stronger performance on higher-GDP countries than on low-GDP countries.  
LLM-based interpretation (Anthropic API) helped summarize which features were most influential

***

## 5. Discussion and Conclusion

Models perform better on high GDP countries than on low GDP countries, and the analysis is correlational rather than causal. Still, environmental and clinical indicators show meaningful statistical relationships with GDP, and LLM based interpretation helps highlight which features distinguish high and low GDP profiles.


***

## References

* Varpit94. "Worldwide Deaths by Country/Risk Factors." Kaggle, n.d., www.kaggle.com/datasets/varpit94/worldwide-deaths-by-risk-factors
* "Sources of Our Population Dataset." Our World in Data, 15 July 2024, https://ourworldindata.org/grapher/sources-population-dataset
* World Bank. "GDP by Country in Current US$, 1988-2022 (Indicator: NY.GDP.MKTP.CD)." World Integrated Trade Solution (WITS), n.d., https://wits.worldbank.org/CountryProfile/en/country/by-country/startyear/ltst/endyear/ltst/indicator/NY-GDP-MKTP-CD
