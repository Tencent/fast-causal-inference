SELECT 
  SRM(numerator, treatment, [1, 2])
FROM causal_inference_test;

SELECT 
  SRM(numerator, toString(treatment), [1, 20])
FROM causal_inference_test;

SELECT 
  SRM(numerator, treatment, [1.2,2.3])
FROM causal_inference_test;
