# SMS Spam Detection — Random Forest

A Streamlit web app for classifying SMS messages as `spam` or `ham`.

## Files required

Keep these files in the same GitHub repository:

- `app.py`
- `requirements.txt`
- `random_forest_sms_model.pkl` (exported from your Google Colab notebook)

## Deploy

1. Upload all three files to a GitHub repository.
2. Open https://share.streamlit.io/
3. Sign in with GitHub and choose **Create app**.
4. Select your repository, branch (`main`), and main file path (`app.py`).
5. Click **Deploy**.

The app uses the saved scikit-learn Pipeline, so the TF-IDF vectorizer and Random Forest model are loaded together.
