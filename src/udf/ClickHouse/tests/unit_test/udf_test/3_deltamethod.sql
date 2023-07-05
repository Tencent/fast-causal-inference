SELECT 'Deltamehtod',  
  round(Deltamethod('x1/x2')(numerator, denominator), 5),  
  round(Deltamethod('x1/x2', true)(numerator, denominator), 5), 
  round(Deltamethod('x1/x2', false)(numerator, denominator), 5), 
  round(Deltamethod('x1/x2 + x3/x4', false)(numerator, denominator, X1, X2), 5), 
  round(Deltamethod('x1/x2/x3+x4', false)(numerator, denominator, X1, X2), 5) 
FROM  causal_inference_test
