WITH (
        SELECT OlsState(false)(Y, X1, X2, X3)
        FROM causal_inference_test
    ) AS model
SELECT round(evalMLMethod(model, X1, X2, X3), 4) AS result
FROM causal_inference_test
ORDER BY result ASC
LIMIT 10;

WITH (
        SELECT OlsState(true)(Y, X1, X2, X3)
        FROM causal_inference_test
    ) AS model
SELECT round(evalMLMethod(model, X1, X2, X3), 4) AS result
FROM causal_inference_test
ORDER BY result ASC
LIMIT 10;

WITH (
  SELECT WlsState(true)(Y, X1, X2, X3, weight)
        FROM causal_inference_test
    ) AS model
SELECT round(evalMLMethod(model, X1, X2, X3), 4) AS result
FROM causal_inference_test
ORDER BY result ASC
LIMIT 10;


WITH (
  SELECT WlsState(true)(Y, X1, X2, X3, weight)
        FROM causal_inference_test
    ) AS model
SELECT round(evalMLMethod(model, X1, X2, X3), 4) AS result
FROM causal_inference_test
ORDER BY result ASC
LIMIT 10;
