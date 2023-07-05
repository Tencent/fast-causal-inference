SELECT 
  MatrixMultiplication(true, true)(X1, X2, X3)
FROM causal_inference_test;

SELECT 
  MatrixMultiplication(true, false)(X1, X2, X3)
FROM causal_inference_test;

SELECT 
  MatrixMultiplication(false, true)(X1, X2, X3)
FROM causal_inference_test;
SELECT 
  MatrixMultiplication(false, false)(X1, X2, X3)
FROM causal_inference_test;
