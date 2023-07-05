SELECT 
  kolmogorovSmirnovTest('two-sided')(X1, treatment)
FROM causal_inference_test;

SELECT 
  kolmogorovSmirnovTest('less')(X2, treatment)
FROM causal_inference_test;

SELECT 
  kolmogorovSmirnovTest('greater')(X3, treatment)
FROM causal_inference_test;
