import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from wordcloud import WordCloud

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('tfidf.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

#css
st.markdown(
    """
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #333;
        text-align: center;
        padding-bottom: 20px;
        border-bottom: 2px solid #ccc;
        margin-bottom: 20px;
    }
    .header {
        font-size: 24px;
        font-weight: bold;
        color: #555;
        margin-bottom: 10px;
    }
    .subheader {
        font-size: 18px;
        font-weight: bold;
        color: #777;
        margin-bottom: 10px;
    }
    .prediction {
        font-size: 20px;
        font-weight: bold;
        margin-top: 10px;
    }
    .prediction-real {
        color: #4CAF50;
    }
    .prediction-fake {
        color: #FF5733;
    }
    .footer {
        font-size: 14px;
        color: #999;
        text-align: center;
        margin-top: 30px;
    }
    .logo {
        position: absolute;
        top: 10px;
        left: 20px;
        width: 150px;  /* Adjust logo width as needed */
        z-index: 999;  /* Ensure logo is above other content */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Logo image
st.image("image.png", width=150, caption='Company Logo', output_format='PNG')

st.markdown('<h1 class="title">Brainwaves Fake News Detection</h1>', unsafe_allow_html=True)
st.markdown('---')
st.markdown('<h2 class="header">Detect if a news article is real or fake</h2>', unsafe_allow_html=True)

#input
user_input = st.text_area("Enter the news text here:")

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if st.button('Predict'):
    if not user_input.strip():
        st.warning("Please enter some news text to predict.")
    else:
        try:
            input_tfidf = tfidf.transform([user_input])          
            # prediction probability
            prediction = model.predict(input_tfidf)
            probability = model.predict_proba(input_tfidf)[0][prediction[0]]

            #confidence
            st.markdown('---')
            st.markdown('<h3 class="subheader">Prediction:</h3>', unsafe_allow_html=True)
            if prediction[0] == 1:
                st.markdown(f'<p class="prediction prediction-real">The news is <strong>REAL</strong>.</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="prediction prediction-fake">The news is <strong>FAKE</strong>.</p>', unsafe_allow_html=True)
            st.markdown('<h3 class="subheader">Prediction Confidence:</h3>', unsafe_allow_html=True)
            st.write(f"{probability * 100:.2f}%")
            st.session_state.prediction_history.append({'News Text': user_input, 'Prediction': 'Real' if prediction[0] == 1 else 'Fake', 'Confidence': probability * 100})
        except Exception as e:
            st.error(f"Error during prediction: {e}")

st.markdown('---')
st.markdown('<h2 class="header">How to Use:</h2>', unsafe_allow_html=True)
st.write("- Enter the news text in the text area above.")
st.write("- Click on the 'Predict' button to see the prediction result and confidence.")

# Data visualization section
st.markdown('---')
st.markdown('<h2 class="header">Prediction History and Visualization</h2>', unsafe_allow_html=True)
prediction_df = pd.DataFrame(st.session_state.prediction_history)

if not prediction_df.empty:
    st.dataframe(prediction_df)
    # Plot the distribution of real vs fake predictions
    plt.figure(figsize=(10, 5))
    sns.countplot(x='Prediction', data=prediction_df, palette=['#4CAF50', '#FF5733'])
    plt.title('Distribution of Real vs Fake News Predictions')
    plt.xlabel('Prediction')
    plt.ylabel('Count')
    st.pyplot(plt)

    #Word Cloud
    st.markdown('---')
    st.markdown('<h2 class="header">Word Cloud of Predictions</h2>', unsafe_allow_html=True)

    fake_news = ' '.join(prediction_df[prediction_df['Prediction'] == 'Fake']['News Text'])
    real_news = ' '.join(prediction_df[prediction_df['Prediction'] == 'Real']['News Text'])

    col1, col2 = st.columns(2)

    with col1:
        if fake_news:
            st.write("Fake News Word Cloud")
            fake_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(fake_news)
            plt.figure(figsize=(8, 4))
            plt.imshow(fake_wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
        else:
            st.write("No fake news text available for word cloud.")

    with col2:
        if real_news:
            st.write("Real News Word Cloud")
            real_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(real_news)
            plt.figure(figsize=(8, 4))
            plt.imshow(real_wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
        else:
            st.write("No real news text available for word cloud.")
else:
    st.write("No predictions made yet.")

st.markdown('---')
st.subheader('Learn More:')
st.write("Interested in learning more about fake news detection and critical thinking? Check out these resources:")
st.write("- [Fake news collection](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)")
st.write("- [True news collection](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=True.csv)")
st.write("- [How to Spot Fake News](https://www.bbc.co.uk/news/how-to-spot-fake-news)")
st.markdown('---')
st.markdown('<p class="footer">Created with ❤️ by AMRUTH REDDY G</p>', unsafe_allow_html=True)
