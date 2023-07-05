WITH (
        SELECT stochasticLogisticRegressionState(0.1, 0.1, 1., 'Lasso')(Y, X1, X2, X3)
        FROM causal_inference_test
    ) AS model
SELECT round(sum(evalMLMethod(model, X1, X2, X3))/1000)
FROM causal_inference_test;

