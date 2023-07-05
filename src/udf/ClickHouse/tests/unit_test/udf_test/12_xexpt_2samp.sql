SELECT 
  Xexpt_Ttest_2samp(0.03, 0.004, 0.6, 'X=x3/x4')(numerator, denominator, X1, X2, X3, treatment)
FROM causal_inference_test;

SELECT 
  Xexpt_Ttest_2samp(0.3, 0.4, 0.12, 'X=x3')(numerator, denominator, X1, X3, treatment)
FROM causal_inference_test;

SELECT 
  Xexpt_Ttest_2samp(0.3, 0.4, 0.12)(numerator, denominator, X1, treatment)
FROM causal_inference_test;
