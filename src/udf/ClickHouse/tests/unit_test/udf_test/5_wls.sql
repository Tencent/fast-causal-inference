SELECT
  Wls(Y, X1, X2, X3, weight)
FROM causal_inference_test;

SELECT
  Wls(false)(Y, X1, X2, X3, weight)
FROM  causal_inference_test;

SELECT
  Wls(true)(Y, X1, X2, X3, weight)
FROM  causal_inference_test;
