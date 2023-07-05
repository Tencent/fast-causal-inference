select 'Ttest_1samp', 
Ttest_1samp('x1/x2', 'less', 0)(numerator, denominator), 
Ttest_1samp('x1/x2', 'less')(numerator, denominator), 
Ttest_1samp('x1/x2')(numerator, denominator), 
Ttest_1samp('x1/x2', 'two-sided', 1)(numerator, denominator), 
Ttest_1samp('x1/x2', 'greater', 1.111)(numerator, denominator), 
Ttest_1samp('x1/x2', 'less', 1.111, 'X=x3/x4+x5')(numerator, denominator, X1, X2, X3), 
Ttest_1samp('x1/x2', 'less', 1.111, 'X=x3/x4+x5')(numerator, denominator, X1, X2, X3),
Ttest_1samp('x1/x2', 'less', 1.111, 0.95, 'X=x3/x4+x5')(numerator, denominator, X1, X2, X3),
Ttest_1samp('x1/x2', 'less', 1.111, 'X=x3/x4+x5', 0.95)(numerator, denominator,X1, X2, X3) from causal_inference_test;


select Ttest_2samp('x1/x2')(numerator, denominator, treatment), 
Ttest_2samp('x1/x2', 'less')(numerator, denominator, treatment), 
Ttest_2samp('x1/x2', 'two-sided')(numerator, denominator, treatment), 
Ttest_2samp('x1/x2', 'greater')(numerator, denominator, treatment), 
Ttest_2samp('x1/x2', 'less', 'X=x3/x4+x5')(numerator, denominator, X1, X2, X3, treatment), 
Ttest_2samp('x1/x2', 'less', 'X=x3/x4+x5')(numerator, denominator, X1, X2, X3, treatment),
Ttest_2samp('x1/x2', 'less', 'X=x3/x4+x5', 0.95)(numerator, denominator, X1, X2, X3, treatment),
Ttest_2samp('x1/x2', 'less', 0.95, 'X=x3/x4+x5')(numerator, denominator, X1, X2, X3, treatment) from causal_inference_test;

