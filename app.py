import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("mail_data.csv")
df['Category'] = df['Category'].map({'spam': 0, 'ham': 1})

X = df['Message']
Y = df['Category']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

vectorizer = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_features, Y_train)

def main():
    st.set_page_config(
        page_title="Email Spam Classifier",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #ffffff;
        }
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border-radius: 25px !important;
            padding: 15px 30px !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2) !important;
            transition: all 0.3s ease !important;
        }
        .stButton>button:hover {
            transform: scale(1.05) !important;
            box-shadow: 0px 6px 20px rgba(0, 0, 0, 0.3) !important;
        }
        .stTextInput>div>div>input {
            background-color: rgba(255, 255, 255, 0.9) !important;
            color: #333333 !important;
            border-radius: 25px !important;
            padding: 15px 20px !important;
            font-size: 16px !important;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2) !important;
            transition: all 0.3s ease !important;
        }
        .stTextInput>div>div>input:focus {
            border-color: #667eea !important;
            box-shadow: 0px 0px 10px 4px rgba(102, 126, 234, 0.5) !important;
        }
        .footer {
            text-align: center;
            padding: 20px 0;
            color: rgba(255, 255, 255, 0.7);
            font-size: 14px;
        }
        .title {
            font-size: 48px;
            font-weight: 600;
            text-align: center;
            margin-bottom: 20px;
            animation: fadeIn 2s ease-in-out;
        }
        .subtitle {
            font-size: 24px;
            font-weight: 300;
            text-align: center;
            margin-bottom: 40px;
            animation: fadeIn 3s ease-in-out;
        }
        .result {
            font-size: 28px;
            font-weight: 600;
            text-align: center;
            margin-top: 30px;
            animation: fadeIn 1s ease-in-out;
        }
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<h1 class="title">Email Spam Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Enter your email content below to check if it\'s spam or not.</p>', unsafe_allow_html=True)


    email_input = st.text_input('Enter the email content here')

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button('Classify'):
            if email_input.strip() == '':
                st.error('Please enter an email.')
            else:
                email_features = vectorizer.transform([email_input])
                prediction = model.predict(email_features)

                if prediction[0] == 1:
                    st.markdown('<p class="result" style="color: #4caf50;">✅ True email</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="result" style="color: #f44336;">❌ Spam</p>', unsafe_allow_html=True)

    st.markdown('<p class="footer">Developed by <a href="https://garvanand-github-io-git-main-garvanand.vercel.app/" target="_blank" style="color: rgba(255, 255, 255, 0.9);">Garv Anand</a></p>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
