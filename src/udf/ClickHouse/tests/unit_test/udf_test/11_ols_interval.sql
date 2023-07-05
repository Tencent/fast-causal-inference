WITH (
        SELECT OlsIntervalState(Y, X1, X2, X3)
        FROM causal_inference_test
    ) AS model
SELECT
    round(sum(res[1]), 0),
    round(sum(res[2]), 0),
    round(sum(res[3]), 0)
FROM
(
    SELECT evalMLMethod(model, 'confidence', 0.1, X1, X2, X3) AS res
    FROM causal_inference_test
);

WITH (
        SELECT OlsIntervalState(Y, X1, X2, X3)
        FROM causal_inference_test
    ) AS model
SELECT
    round(sum(res[1]), 0),
    round(sum(res[2]), 0),
    round(sum(res[3]), 0)
FROM
(
    SELECT evalMLMethod(model, 'confidence', X1, X2, X3) AS res
    FROM causal_inference_test
);

WITH (
        SELECT OlsIntervalState(Y, X1, X2, X3)
        FROM causal_inference_test
    ) AS model
SELECT
    round(sum(res[1]), 0),
    round(sum(res[2]), 0),
    round(sum(res[3]), 0)
FROM
(
    SELECT evalMLMethod(model, 'prediction', X1, X2, X3) AS res
    FROM causal_inference_test
);

