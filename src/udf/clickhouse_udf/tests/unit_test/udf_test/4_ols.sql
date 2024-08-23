SELECT
  Ols(Y, X1, X2, X3, weight)
FROM causal_inference_test;

SELECT
  Ols(false)(Y, X1, X2, X3, weight)
FROM  causal_inference_test;

SELECT
  Ols(true)(Y, X1, X2, X3, weight)
FROM  causal_inference_test;
