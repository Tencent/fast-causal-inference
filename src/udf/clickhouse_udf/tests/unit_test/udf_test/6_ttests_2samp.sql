select Ttests_2samp('x1/x2')(numerator, denominator, treatment), 
Ttests_2samp('x1/x2', 'less')(numerator, denominator, treatment), 
Ttests_2samp('x1/x2', 'two-sided')(numerator, denominator, treatment), 
Ttests_2samp('x1/x2', 'greater')(numerator, denominator, treatment), 
Ttests_2samp('x1/x2', 'less', 'X=x3/x4+x5')(numerator, denominator, X1, X2, X3, treatment), 
Ttests_2samp('x1/x2', 'less', 'X=x3/x4+x5')(numerator, denominator, X1, X2, X3, treatment),
Ttests_2samp('x1/x2', 'less', 'X=x3/x4+x5', 0.95)(numerator, denominator, X1, X2, X3, treatment),
Ttests_2samp('x1/x2', 'less', 0.95, 'X=x3/x4+x5')(numerator, denominator, X1, X2, X3, treatment) from causal_inference_test;


