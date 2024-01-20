# Market-Price-Prediction
I propose to analyze the General Electric (GE) dataset to predict its stock prices using Random Forest Algorithm, assessing performance via Mean Squared Error.
The first thing I did in the project was to construct the Random Forest Model where a
Random Forest Regressor is chosen as the predictive model. To simplify the model and
potentially avoid overfitting, certain parameters are adjusted, n_estimator, Max depth,
and Randomstate. For n_estimator I tried [10,20,50], Max depth we used [5,10,15] and
random state I used [2,15,30,42]. My best run was with n_estimator of 50, Max depth of
5, and random state of 30.
