WITH (
  SELECT DistributedNodeRowNumber(1)(0)
  FROM numbers(10000000)
) AS pa SELECT BootStrap('Deltamethod("x1")', 10000000, 3, pa)(number) FROM numbers(10000000);

WITH (
  SELECT DistributedNodeRowNumber(0)
  FROM causal_inference_test
) AS pa SELECT BootStrap('sum', 123456789, 5, pa)(1) FROM causal_inference_test;

WITH (
  SELECT DistributedNodeRowNumber(1)(0)
  FROM numbers(10000)
) AS pa,
(SELECT 
  BootStrapState('Ols', 1234567, 5, pa)(1, number, number*number)
FROM numbers(10000)
) as model
SELECT 
  round(sum(evalMLMethod(model, 1, 1, 0)), 4)
FROM numbers(10000)
