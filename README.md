# MPG Predictor

1. Put `auto-mpg.csv` in this folder.
2. `pip install -r requirements.txt`
3. `python train.py`  (compares Linear Regression, Random Forest, Gradient Boosting; saves the best to `mpg_model.joblib`)
4. `streamlit run app.py`

Preprocessing is inside the saved sklearn Pipeline, so the app passes raw inputs:
median imputation (horsepower has `?` values) + standard scaling for numeric features,
most-frequent imputation + one-hot encoding for `origin`. `car name` is dropped (high-cardinality identifier).
The best model is chosen by cross-validated RMSE on the training split; the test split is only reported.
