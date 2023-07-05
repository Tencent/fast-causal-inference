SELECT arrayJoin(GroupSet(X1, treatment, X2, X3)) AS a
FROM causal_inference_test
ORDER BY a.6 ASC;

SELECT arrayJoin('Column1', 'Column2')(GroupSet(X1, treatment, X2, X3)) AS a
FROM causal_inference_test
ORDER BY a.6 ASC;
