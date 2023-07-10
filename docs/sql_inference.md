# Fast Causal Inference SQL Reference
Fast Causal Inference suports a declarative query language based on SQL that is identical to the ANSI SQL standard in many cases.

## Deltamethod
Delta method is a method for estimating the asymptotic distribution of a function. It is commonly used to estimate the asymptotic distribution of a function given the asymptotic distribution of a random variable.

**Query**
```sql
SELECT
  Deltamethod('x1/x2')(numerator, denominator) as std
FROM
  test_data_small
```
**Result**
```text
┌───────────────std─┐
│ 35.15462409884396 │
└───────────────────┘
```

## Ttest
The t-test is a statistical test used to determine if there is a significant difference between the means of two groups. It is commonly used when comparing the means of two independent samples, such as comparing the average scores of two groups of students who received different treatments or interventions.

#### Ttest_1samp

**Query**
```sql
SELECT
  Ttest_1samp('x1/x2','two-sided')(numerator, denominator)  AS ttest_result
FROM
  test_data_small;
```

**Result**
```text
┌─ttest_result─────────────────────────────────────────────────────────────
│ estimate    stderr      t-statistic p-value     lower       upper        │
  814.455640  35.154624   23.167810   0.000000    745.553737  883.357542   │
└───────────────────────────────────────────────────────────────────────────
```

**Query**
```sql
SELECT
  Ttest_2samp('x1/x2','two-sided')(numerator, denominator, treatment)  AS ttest_result
FROM
  test_data_small;
```

**Result**
```text
┌─ttest_result─────────────────────────────────────────────────────────────
│ estimate    stderr      t-statistic p-value     lower       upper        │
  -50.589793  70.303156   -0.719595   0.471774    -188.381868 87.202282    │
└───────────────────────────────────────────────────────────────────────────
```

## OLS
OLS(Ordinary Least Squares) is a method for estimating the parameters of a linear regression model. It works by minimizing the sum of the squared differences between the observed values and the predicted values of the dependent variable.

**Query**
```sql
SELECT
  Ols(Y, X1, X2, X3) 
FROM
  test_data_small;
```

**Result**
```text
┌─Ols(Y, X1, X2, X3)─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Call:
  lm( formula = y ~ x1 + x2 + x3 )

  Coefficients:
  .		Estimate		Std. Error	t value		Pr(>|t|)
  (Intercept)	351.038825  	67.626843   	5.190821    	0.000000
  x1		-45.850440  	7.927278    	-5.783882   	0.000000
  x2		-55.279399  	33.750086   	-1.637904   	0.101442
  x3		3548.638856 	31.204135   	113.723355  	0.000000

  Residual standard error: 15528.723989 on 793196 degrees of freedom
  Multiple R-squared: 0.016051, Adjusted R-squared: 0.016047
  F-statistic: 4313.071274 on 3 and 793196 DF,  p-value: 0.000000
 │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## SRM
SRM is a method for selecting the best model among a set of candidate models. It works by balancing the complexity of the model (i.e., the number of parameters) with its ability to fit the data.

**Query**
```sql
SELECT  
  SRM(X1, treatment, [1,2])
FROM  
  test_data_small
```

**Result**
```text
┌─SRM(X1, treatment, [1, 2])───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ groupname   f_obs       ratio       chisquare   p-value
  0           2.124e+06   1.000000    5.315e+05   0.000000
  1           2.124e+06   2.000000
 │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Lasso
Lasso (Least Absolute Shrinkage and Selection Operator) is a method for variable selection and regularization in linear regression. It works by adding a penalty term to the ordinary least squares (OLS) objective function, which encourages the coefficients of some variables to be shrunk towards zero.
```sql
SELECT 
  stochasticLinearRegression(0.001, 0.1, 15, 'Lasso')(Y, X1, X2, X3)
FROM
  test_data_small
```

```text
┌─stochasticLinearRegression(0.001, 0.1, 15, 'Lasso')(Y, X1, X2, X3)─────────┐
│ [2936.992174261725,784.0511223992013,1587.4709271977881,560.2428325299145] │
└────────────────────────────────────────────────────────────────────────────┘
```


