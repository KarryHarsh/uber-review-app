import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


DATA_URL = (
    "https://raw.githubusercontent.com/KarryHarsh/uber-review-app/main/preprocessed_data.csv"
)

st.title("Sentiment Analysis of Uber Reviews")
st.sidebar.title("Sentiment Analysis of Uber Reviews")


@st.cache(persist=True)
def load_data():
    data = pd.read_csv(DATA_URL)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

data = load_data()
data = data[data['Clean_Review'].notna()]
st.sidebar.subheader("Show random Reviews")
random_review = st.sidebar.radio('Sentiment', ('Positive', 'Neutral', 'Negative'))
st.sidebar.markdown(data.query("Sentiment == @random_review")[["Review"]].sample(n=1).iat[0, 0])

st.sidebar.markdown("### Number of Reviewss by sentiment")
select = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='1')
sentiment_count = data['Sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Review':sentiment_count.values})
if not st.sidebar.checkbox("Hide", True):
    st.markdown("### Number of Reviews by sentiment")
    if select == 'Bar plot':
        fig = px.bar(sentiment_count, x='Sentiment', y='Review', color='Review', height=500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count, values='Review', names='Sentiment')
        st.plotly_chart(fig)



st.sidebar.subheader("Total number of Reviews for each Rating")
each_rating = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='2')
rating_sentiment_count = data.groupby('Rating')['Sentiment'].count().sort_values(ascending=False)
rating_sentiment_count = pd.DataFrame({'Rating':rating_sentiment_count.index, 'Review':rating_sentiment_count.values.flatten()})
if not st.sidebar.checkbox("Close", True, key='2'):
    if each_rating == 'Bar plot':
        st.subheader("Total number of Reviews for each Rating")
        fig_1 = px.bar(rating_sentiment_count, x='Rating', y='Review', color='Review', height=500)
        st.plotly_chart(fig_1)
    if each_rating == 'Pie chart':
        st.subheader("Total number of Review for each Rating")
        fig_2 = px.pie(rating_sentiment_count, values='Review', names='Rating')
        st.plotly_chart(fig_2)


@st.cache(persist=True)
def plot_sentiment(rating):
    df = data[data['Rating']==rating]
    count = df['Sentiment'].value_counts()
    count = pd.DataFrame({'Sentiment':count.index, 'Review':count.values.flatten()})
    return count


st.sidebar.subheader("Breakdown Rating by sentiment")
choice = st.sidebar.multiselect('Pick Rating', (1,2,3,4,5))
if len(choice) > 0:
    st.subheader("Breakdown Rating by sentiment")
    breakdown_type = st.sidebar.selectbox('Visualization type', ['Pie chart', 'Bar plot', ], key='3')
    fig_3 = make_subplots(rows=1, cols=len(choice), subplot_titles=choice)
    if breakdown_type == 'Bar plot':
        for i in range(1):
            for j in range(len(choice)):
                fig_3.add_trace(
                    go.Bar(x=plot_sentiment(choice[j]).Sentiment, y=plot_sentiment(choice[j]).Review, showlegend=False),
                    row=i+1, col=j+1
                )
        fig_3.update_layout(height=600, width=800)
        st.plotly_chart(fig_3)
    else:
        fig_3 = make_subplots(rows=1, cols=len(choice), specs=[[{'type':'domain'}]*len(choice)], subplot_titles=choice)
        for i in range(1):
            for j in range(len(choice)):
                fig_3.add_trace(
                    go.Pie(labels=plot_sentiment(choice[j]).Sentiment, values=plot_sentiment(choice[j]).Review, showlegend=True),
                    i+1, j+1
                )
        fig_3.update_layout(height=600, width=800)
        st.plotly_chart(fig_3)
st.sidebar.subheader("Breakdown Rating by sentiment")
choice = st.sidebar.multiselect('Pick Rating', (1,2,3,4,5), key=0)
if len(choice) > 0:
    choice_data = data[data.Rating.isin(choice)]
    fig_0 = px.histogram(
                        choice_data, x='Rating', y='Sentiment',
                         histfunc='count', color='Sentiment',
                         facet_col='Sentiment', labels={'Sentiment':'Review'},
                          height=600, width=800)
    st.plotly_chart(fig_0)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.header("Word Cloud")
word_sentiment = st.sidebar.radio('Display word cloud for what sentiment?', ('Positive', 'Neutral', 'Negative'))
if not st.sidebar.checkbox("Close", True, key='3'):
    st.subheader('Word cloud for %s sentiment' % (word_sentiment))
    df = data[data['Sentiment']==word_sentiment]
    words = ' '.join(df['Clean_Review'])
    #processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.pyplot()
