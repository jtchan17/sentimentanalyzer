import streamlit as st 
import pandas as pd 
import plotly_express as px 
import altair as alt
import torch
import matplotlib.pyplot as plt
import pdfkit
import jinja2
# from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader
from streamlit.components.v1 import iframe
import plotly.io as pio
import matplotlib.pyplot as plt
import os
from wordcloud import WordCloud, STOPWORDS
import json
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb

##########################################################################################################################################
PDF_TEMPLATE_FILE = 'PDFtemplate.html'
IMG_FOLDER = os.path.join(os.getcwd(), 'tmp/image')
WKHTMLTOPDF_PATH = os.path.join(os.getcwd(), 'wkhtmltopdf\\bin\\wkhtmltopdf.exe')
os.makedirs(IMG_FOLDER, exist_ok=True)
##########################################################################################################################################
alt.themes.enable("dark")

@st.cache_data
def load_financial_data():
    with open("financial_news.json", "r") as json_file:
        financialnews_data = json.load(json_file)
    # df = pd.read_csv('Financial_News.csv')
    return financialnews_data

@st.cache_data
def load_stock_data():
    with open("stockprices.json", "r") as json_file:
        stockprices_data = json.load(json_file)
    return stockprices_data

df_fn = load_financial_data()
df_sp = load_stock_data()
df_fn = pd.DataFrame(df_fn)
df_sp = pd.DataFrame(df_sp)

##########################################################################################################################################

##########################################################################################################################################
# Side bar (Login)
with st.sidebar:
    # st.title(f'Welcome {st.session_state.role}! :sunglasses:')
    st.title(f'Welcome {st.session_state.username}! :sunglasses:')
    custom_colors = {
        'Red': '#EF553B',
        'Blue': '#636EFA',
        'Green': '#00CC96',
        'Yellow': '#FECB52',
        'Pink': '#E45756',
        'Orange': '#FFA15A',
        'Purple': '#AB63FA',
        'Cyan': '#19D3F3',
        'Lime': '#B6E880',
        'Magenta': '#FF6692'
    }
    #------------------------------------------------------------------------
    #Text Colour
    font_colors = {
        'White': ':white',
        'Violet': ':violet',
        'Blue':':blue',
        'Green': ':green',
        'Orange': ':orange',
        'Red': ':red',
        'Purple': ':purple'
    }
    font_family = {
        'Sans-serif': 'sans-serif',
        'Serif': 'serif',
        'Monospace': 'monospace',
        'Lucida Console': 'Lucida Console',
        'Courier New': 'Courier New'
    }
    font_style = ['normal', 'italic']

    st.subheader('Text', divider=True)
    font_family_selection = st.selectbox('Font Family', options=list(font_family.keys()), index=0)
    final_font_family = font_family[font_family_selection]
    font_style_selection = st.selectbox('Style', font_style, index=0)
    #------------------------------------------------------------------------
    #Companies Colour
    st.subheader('Tickers', divider=True)
    companies_default_colors = {
        'AAPL': 'Blue',
        'AMZN': 'Orange',
        'TSLA': 'Green',
        'MSFT': 'Red',
        'META': 'Purple'
    }

    #set the default colours for each companies
    cust_aapl_selection = st.selectbox('AAPL', list(custom_colors.keys()), index=list(custom_colors.keys()).index(companies_default_colors['AAPL']))
    cust_amzn_selection = st.selectbox('AMZN', list(custom_colors.keys()), index=list(custom_colors.keys()).index(companies_default_colors['AMZN']))
    cust_tsla_selection = st.selectbox('TSLA', list(custom_colors.keys()), index=list(custom_colors.keys()).index(companies_default_colors['TSLA']))
    cust_msft_selection = st.selectbox('MSFT', list(custom_colors.keys()), index=list(custom_colors.keys()).index(companies_default_colors['MSFT']))
    cust_meta_selection = st.selectbox('META', list(custom_colors.keys()), index=list(custom_colors.keys()).index(companies_default_colors['META']))

    #final colour selection
    final_aapl_colour = custom_colors[cust_aapl_selection]
    final_amzn_colour = custom_colors[cust_amzn_selection]
    final_tsla_colour = custom_colors[cust_tsla_selection]
    final_msft_colour = custom_colors[cust_msft_selection]
    final_meta_colour = custom_colors[cust_meta_selection]

    #------------------------------------------------------------------------
    #Sentiment Colour
    st.subheader('Sentiment', divider=True)
    sentiment_default_colors = {
        'positive': 'Green',
        'negative': 'Red',
        'neutral': 'Blue'
    }
    cust_pos_selection = st.selectbox('Positive: ', list(custom_colors.keys()), index=list(custom_colors.keys()).index(sentiment_default_colors['positive']))
    cust_neg_selection = st.selectbox('Negative:', list(custom_colors.keys()), index=list(custom_colors.keys()).index(sentiment_default_colors['negative']))
    cust_neu_selection = st.selectbox('Neutral:', list(custom_colors.keys()), index=list(custom_colors.keys()).index(sentiment_default_colors['neutral']))

    final_pos_colour = custom_colors[cust_pos_selection]
    final_neg_colour = custom_colors[cust_neg_selection]
    final_neu_colour = custom_colors[cust_neu_selection]

    

##########################################################################################################################################

##########################################################################################################################################
#   Dashboard
# st.title(f'📈 {final_font_colour}[Dashboard of Stock Prices and Financial News]')
st.markdown(
    f"""
    <p style="font-family: {final_font_family}; font-size: 40px; font-style: {font_style_selection}; font-weight: bold;">
    📈 Dashboard of Stock Prices and Financial News
    </p>
    """,
    unsafe_allow_html=True,
)
#-------------------------------------------------------------------------------------------
#CSS Injection
#Button
st.markdown(
    f"""
    <style>
    .stButton{{
        background-color: #111111; /* Button background color */
        border-radius: 10px; /* Rounded corners */
        padding: 5px 10px;
    }}
    .stButton > button > div > p {{
        font-family: {final_font_family} !important; /* Change to desired font */
        font-size: 16px; /* Adjust font size */
        font-style: {font_style_selection}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# #Download Button
st.markdown(
    f"""
    <style>
    .stDownloadButton {{
        background-color: #111111; /* Button background color */
        border-radius: 10px; /* Rounded corners */
        padding: 5px 10px;
    }}
    .stDownloadButton > button > div > p{{
        font-family: {final_font_family} !important; /* Change to desired font */
        font-size: 16px; /* Adjust font size */
        font-style: {font_style_selection}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# #Popover
st.markdown(
    f"""
    <style>
    .stPopover {{
        background-color: #111111; /* Button background color */
        border-radius: 10px; /* Rounded corners */
        padding: 5px 10px;
    }}

    .stPopover > div > button > div > p{{
        font-family: {final_font_family}; /* Change to desired font */
        font-size: 16px; /* Adjust font size */
        font-style: {font_style_selection}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

#Table
st.markdown(
    f"""
    <style>
    .stTable{{
        font-family: {final_font_family}; /* Change to desired font */
        font-style: {font_style_selection}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

#DataFrame
st.markdown(
    f"""
    <style>
    div[data-testid="stDataFrame"] > div[data-testid="stDataFrameResizable"] > div[class="stDataFrameGlideDataEditor gdg-wmyidgi"] > div > div[class="gdg-s1dgczr6"] > div[class="dvn-underlay"] {{
        background-color: #555555
        font-family: {final_font_family}; /* Change to desired font */
        font-style: {font_style_selection}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

#Tabs
st.markdown(
    f"""
    <style>
    .stTabs > div > div > div > button > div > p{{
        font-family: {final_font_family}; /* Change to desired font */
        font-style: {font_style_selection}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

#Chart
def chart_update_layout(chart):
    chart.update_layout(
            font=dict(
                family=font_family_selection,
                style=font_style_selection
            )
        )
#-------------------------------------------------------------------------------------------

#search and filtering
fil_col1, fil_col2, fil_col3, fil_col4 = st.columns([1.25, 2, 4, 1])

with fil_col1:
    years = ['All', '2021', '2022', '2023']
    popover = st.popover("Select Year")
    select_year = popover.radio(label='Select Year',options=years, key='select_year', label_visibility="collapsed")

with fil_col2:
    popover = st.popover("Ticker")
    aapl = popover.checkbox('AAPL', key='aapll', value=True)
    amzn = popover.checkbox('AMZN', key='amznn', value=True)
    meta = popover.checkbox('META', key='metaa', value=True)
    msft = popover.checkbox('MSFT', key='msftt', value=True)
    tsla = popover.checkbox('TSLA', key='tslaa', value=True)
    companies = {'AAPL': aapl, 'AMZN': amzn, 'META': meta, 'MSFT': msft, 'TSLA': tsla}

    if 'aapll' not in st.session_state:
        st.session_state['aapll'] = False
    if 'amznn' not in st.session_state:
        st.session_state['amznn'] = False
    if 'metaa' not in st.session_state:
        st.session_state['metaa'] = False
    if 'msftt' not in st.session_state:
        st.session_state['msftt'] = False
    if 'tslaa' not in st.session_state:
        st.session_state['tslaa'] = False
    if 'select_year' not in st.session_state:
        st.session_state['select_year'] = 'All'

with fil_col3:
    def clear_filters():
        st.session_state['aapll'] = True
        st.session_state['amznn'] = True
        st.session_state['metaa'] = True
        st.session_state['msftt'] = True
        st.session_state['tslaa'] = True
        st.session_state['select_year'] = 'All'
        st.session_state['positive'] = True
        st.session_state['negative'] = True
        st.session_state['neutral'] = True
        st.session_state['politics'] = True
        st.session_state['economy'] = True
        st.session_state['technology'] = True
        st.session_state['health'] = True
        st.session_state['sports'] = True
        st.session_state['entertainment'] = True
        st.session_state['science'] = True
        st.session_state['business'] = True
        st.session_state['travel'] = True
        st.session_state['education'] = True
        st.session_state['lifestyle'] = True
        st.session_state['finance'] = True
        st.session_state['investing'] = True
        st.session_state['wellness'] = True

    btn_clear = st.button('Clear All Filter', key='clearFilter', on_click=clear_filters)

###### Filter based on Year and Company ######
def filter_data(dfname, year, companies):
    df=''
    # determine the column name
    if dfname == 'df_fn':
        date_column = 'published date'
        df = df_fn
    else:
        date_column = 'Date'
        df = df_sp

    df[date_column] = pd.to_datetime(df[date_column])
    if year !='All':
        df = df[df[date_column].dt.year == int(year)]
    if companies:
        df = df[df['company'].isin(companies)]

    return df

# List of selected companies
selected_companies = [company for company, selected in companies.items() if selected]

# Determine queries based on selected filters
if select_year == 'All' and not selected_companies:
    filtered_df_fn = df_fn
    filtered_df_sp = df_sp
else:
    filtered_df_fn = filter_data('df_fn', select_year, selected_companies)
    filtered_df_sp = filter_data('df_sp', select_year, selected_companies)

#======================================================================================================================================== 
#ROW 1
r1c1, r1c2 = st.columns((7, 3), gap='small')
with r1c1:
    # st.subheader(f'{final_font_colour}[Historical Stock Data]')
    st.markdown(
        f"""
        <p style="font-family: {final_font_family}; font-size: 24px; font-style: {font_style_selection}; font-weight: bold;">
            Historical Stock Data
        </p>
        """,
        unsafe_allow_html=True,
    )
    chart_HistoricalStockData = px.line(filtered_df_sp, x='Date', y='Adj Close', template='gridon', color='company', 
                                        color_discrete_map={'AAPL': final_aapl_colour,
                                                            'AMZN': final_amzn_colour,
                                                            'TSLA': final_tsla_colour,
                                                            'MSFT': final_msft_colour,
                                                            'META': final_meta_colour})
    chart_update_layout(chart_HistoricalStockData)
    st.plotly_chart(chart_HistoricalStockData, key='chart_HistoricalStockData', use_container_width=True)   

with r1c2:
    # st.subheader(f'{final_font_colour}[Highest Price Across Years]')
    st.markdown(
        f"""
        <p style="font-family: {final_font_family}; font-size: 24px; font-style: {font_style_selection}; font-weight: bold;">
            Highest Price Across Years
        </p>
        """,
        unsafe_allow_html=True,
    )

    def filter_years(df, year):
        df['Date'] = pd.to_datetime(df['Date'])
        if year != 'All':
            # df_filtered = df[df['date']].dt.year
            df['year'] = df['Date'].dt.year
        else:
            df['year'] = df['Date'].dt.year
            
        result = df.groupby(['year', 'company']).agg({'High': 'max'}).reset_index()
        result.rename(columns={'year': 'Year', 'company': 'Companies', 'High': 'Highest'}, inplace=True)
        result = result.sort_values(by=['Year', 'Highest'], ascending=[True, False])
        result = result.reset_index(drop=True)
        return result
    
    table_HighestPriceAcrossYear = filter_years(filtered_df_sp, select_year)
    table_HighestPriceAcrossYear = table_HighestPriceAcrossYear
    st.table(table_HighestPriceAcrossYear)
#======================================================================================================================================== 
#ROW 2
r2c1, r2c2 = st.columns((3, 5), gap='small')
with r2c1:
    # st.subheader(f'{final_font_colour}[Number of News Across Companies]')
    st.markdown(
        f"""
        <p style="font-family: {final_font_family}; font-size: 24px; font-style: {font_style_selection}; font-weight: bold;">
            Number of News Across Companies
        </p>
        """,
        unsafe_allow_html=True,
    )
    table_NumberofNewsAcrossCompanies = filtered_df_fn.groupby('company')['title'].count().reset_index(name='Total')
    table_NumberofNewsAcrossCompanies = table_NumberofNewsAcrossCompanies
    st.table(table_NumberofNewsAcrossCompanies)

with r2c2:
    # st.subheader(f'{final_font_colour}[Frequency of News Over Time]')
    st.markdown(
        f"""
        <p style="font-family: {final_font_family}; font-size: 24px; font-style: {font_style_selection}; font-weight: bold;">
            Frequency of News Over Time
        </p>
        """,
        unsafe_allow_html=True,
    )
    df_article_freq = filtered_df_fn.groupby(['published date', 'company']).size().unstack(fill_value=0)
    df_article_freq = df_article_freq.reset_index()
    df_melted = pd.melt(df_article_freq, id_vars='published date', var_name='company', value_name='frequency')
    chart_FrequencyofNewsOverTime = px.line(df_melted, x='published date', y="frequency", template='gridon', color='company',
                                            color_discrete_map={'AAPL': final_aapl_colour,
                                                                'AMZN': final_amzn_colour,
                                                                'TSLA': final_tsla_colour,
                                                                'MSFT': final_msft_colour,
                                                                'META': final_meta_colour})
    chart_update_layout(chart_FrequencyofNewsOverTime)
    st.plotly_chart(chart_FrequencyofNewsOverTime,use_container_width=True)
#======================================================================================================================================== 
#ROW 3
pop_col1, pop_col2 = st.columns([5,5])
with pop_col1:
    sentiment_popover = st.popover('Choose sentiments')
    positive = sentiment_popover.checkbox('Positive', key='positive', value=True)
    negative = sentiment_popover.checkbox('Negative', key='negative', value=True)
    neutral = sentiment_popover.checkbox('Neutral', key='neutral', value=True)

    # List of selected companies
    sentiments = {'Positive': positive, 'Negative': negative, 'Neutral': neutral}
    selected_sentiments = [sentiment for sentiment, selected in sentiments.items() if selected]

    def filter_sentiment(df):
        if selected_sentiments:
            df = df[df['sentiment_score'].isin(selected_sentiments)]
        return df

with pop_col2:
    topic_popover = st.popover('Choose topics')
    politics = topic_popover.checkbox('Politics', key='politics', value=True)
    economy = topic_popover.checkbox('Economy', key='economy', value=True)
    technology = topic_popover.checkbox('Technology', key='technology', value=True)
    health = topic_popover.checkbox('Health', key='health', value=True)
    sports = topic_popover.checkbox('Sports', key='sports', value=True)
    entertainment = topic_popover.checkbox('Entertainment', key='entertainment', value=True)
    science = topic_popover.checkbox('Science', key='science', value=True)
    business = topic_popover.checkbox('Business', key='business', value=True)
    travel = topic_popover.checkbox('Travel', key='travel', value=True)
    education = topic_popover.checkbox('Education', key='education', value=True)
    lifestyle = topic_popover.checkbox('Lifestyle', key='lifestyle', value=True)
    finance = topic_popover.checkbox('Finance', key='finance', value=True)
    investing = topic_popover.checkbox('Investing', key='investing', value=True)
    wellness = topic_popover.checkbox('Wellness', key='wellness', value=True)

    # #List of Topics
    topics = {
        'Politics': politics,
        'Economy': economy,
        'Technology': technology,
        'Health': health,
        'Sports': sports,
        'Entertainment': entertainment,
        'Science': science,
        'Business': business,
        'Travel': travel,
        'Education': education,
        'Lifestyle': lifestyle,
        'Finance': finance,
        'Investing': investing,
        'Wellness': wellness,}

    selected_topics = [topic for topic, select in topics.items() if select]

def filter_topics(dffn):
    if selected_topics:
        dffn = dffn[dffn['topic'].isin(selected_topics)]
    return dffn
    
r3c1, r3c2 = st.columns((5,5), gap='small')
with r3c1:
    # st.subheader(f'{final_font_colour}[Sentiment Score Over Time]')
    st.markdown(
        f"""
        <p style="font-family: {final_font_family}; font-size: 24px; font-style: {font_style_selection}; font-weight: bold;">
            Sentiment Score Over Time
        </p>
        """,
        unsafe_allow_html=True,
    )

    #sentiment score
    def plot_pie():
        df_sentiment = filtered_df_fn.groupby('sentiment_score').size().reset_index(name='Total')
        df_fn1 = filter_sentiment(df_sentiment)
        chart_SentimentScoreOverTime = px.pie(df_fn1, values='Total', names='sentiment_score', color="sentiment_score",
                                            color_discrete_map={'Negative': final_neg_colour, 
                                                                'Positive': final_pos_colour, 
                                                                'Neutral': final_neu_colour},
                                            hole=0.5)
        chart_SentimentScoreOverTime.update_traces(textposition='inside')
        return chart_SentimentScoreOverTime
    
    chart_SentimentScoreOverTime = plot_pie()
    chart_update_layout(chart_SentimentScoreOverTime)
    st.plotly_chart(chart_SentimentScoreOverTime, use_container_width=True)

    #Sentiments Score Across Companies
    df_fn1 = filter_sentiment(filtered_df_fn)
    grouped_sentiment_df_fn = df_fn1.groupby(['company', 'sentiment_score']).size().unstack(fill_value=0)
    table_SentimentFrequency = grouped_sentiment_df_fn.reset_index()
    grouped_sentiment_df_fn.rename(columns={'company': 'Companies', 'negative': 'Negative', 'neutral': 'Neutral', 'positive': 'Positive'}, inplace=True)
    table_SentimentFrequency = grouped_sentiment_df_fn
    st.table(table_SentimentFrequency)

with r3c2:
    # st.subheader(f'{final_font_colour}[Sentiment Score Across Companies]')
    st.markdown(
        f"""
        <p style="font-family: {final_font_family}; font-size: 24px; font-style: {font_style_selection}; font-weight: bold;">
            Sentiments Distribution by Topic
        </p>
        """,
        unsafe_allow_html=True,
    )
    #Sentiments Distribution by Topic
    df_fn1 = filter_topics(df_fn1)
    chart_TopicFrequency = px.histogram(
        df_fn1, 
        x='sentiment_score', 
        color='topic', 
        labels={'sentiment_score': 'Sentiments', 'count': 'Total'},
        template='plotly_dark'
    )
    chart_update_layout(chart_TopicFrequency)
    st.plotly_chart(chart_TopicFrequency)

    #WordCloud
    # Start with one review:
    text = " ".join(title for title in filtered_df_fn.title)

    # Create stopword list:
    stopwords = set(STOPWORDS)
    stopwords.update(filtered_df_fn['company'].tolist())
    stopwords.update(filtered_df_fn['publisher'].tolist())
    stopwords.update(['Apple', 'Tesla', 'Meta', 'Amazon', 'Microsoft'])

    # Generate a word cloud image
    wordcloud = WordCloud(stopwords=stopwords, background_color="black").generate(text)
    # Display the generated image:
    # the matplotlib way:
    word_frequncy = plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    word_frequncy = st.pyplot(plt)
#======================================================================================================================================== 
#ROW 4
r4c1, r4c2 = st.columns((3, 7), gap='small')

with r4c1:
    # st.subheader(f'{final_font_colour}[Top 10 Publishers]'+':newspaper:')
    st.markdown(
        f"""
        <p style="font-family: {final_font_family}; font-size: 24px; font-style: {font_style_selection}; font-weight: bold;">
            Top 10 Publishers 📰
        </p>
        """,
        unsafe_allow_html=True,
    )
    df_fn1 = (filtered_df_fn.groupby('publisher').size().reset_index(name='Total'))
    table_TopPublishers = (df_fn1.sort_values(by="Total", ascending=False)).head(10)
    table_TopPublishers = table_TopPublishers
    df = st.dataframe(table_TopPublishers,
                column_order=("publisher", "Total"),
                hide_index=True,
                width=None,
                column_config={
                    "publisher": st.column_config.TextColumn("Publisher",),
                    "Total": st.column_config.ProgressColumn("Total",format="%f",min_value=0,max_value=max(df_fn1.Total),)
                    }
                )
with r4c2:
    # st.subheader(f'{final_font_colour}[Publishers]'+':newspaper:')
    st.markdown(
        f"""
        <p style="font-family: {final_font_family}; font-size: 24px; font-style: {font_style_selection}; font-weight: bold;">
            Publishers 📰
        </p>
        """,
        unsafe_allow_html=True,
    )
    df_fn1 = filtered_df_fn.groupby('publisher').size().reset_index(name='Total')
    chart_Publishers = px.bar(df_fn1,x='Total', y='publisher', template='seaborn')
    chart_Publishers.update_traces(text=df_fn1['publisher'], textposition='inside')
    chart_update_layout(chart_Publishers)
    st.plotly_chart(chart_Publishers, use_container_width=True, height = 1000)
    
#======================================================================================================================================== 
# Tab
pricing_data, news = st.tabs(['Stock Price', 'News'])
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.close()
    processed_data = output.getvalue()
    return processed_data

with pricing_data:
    historical_csv=filtered_df_sp.to_csv(index = False).encode('utf-8')
    historical_xlsx = to_excel(filtered_df_sp)
    st.download_button(label=':material/download: CSV File', data= historical_csv, file_name='Historical Data.csv')
    st.download_button(label=':material/download: Excel File', data= historical_xlsx, file_name='Historical Data.xlsx')

with news:
    news_csv=filtered_df_fn.to_csv(index = False).encode('utf-8')
    news_xlsx = to_excel(filtered_df_fn)
    st.download_button(label=':material/download: CSV File', data= news_csv, file_name='Financial News.csv')
    st.download_button(label=':material/download: Excel File', data= news_xlsx, file_name='Financial News.xlsx')


#======================================================================================================================================== 
with fil_col4:
    templateLoader = jinja2.FileSystemLoader(searchpath="./")
    templateEnv = jinja2.Environment(loader=templateLoader)
    # env = Environment(loader=FileSystemLoader("."), autoescape=select_autoescape())
    template = templateEnv.get_template(PDF_TEMPLATE_FILE)

    #save dataframe to html
    @st.cache_data
    def getTableHTML(table, index, border):
        table_html = table.to_html(index=index, escape=False, border=border)
        return table_html
    
    #save plotly_express chart into png format
    @st.cache_data
    def save_plotly_plot(name, fig):
        # st.write('Reached save_plotly_plot')
        file_name = os.path.join(IMG_FOLDER, f"{ name }.png")
        # st.write(file_name)
        try:
            fig.write_image(file_name, engine="kaleido", scale=1)
            # st.write('reached here')
        except Exception as e:
            st.error(f"Error saving plotly plot: {e}")
            st.stop()  # Stop execution if there's a critical error
        return file_name
    
    #save altair chart into png format
    @st.cache_data
    def save_altair_plot(name, fig):
        file_name = os.path.join(IMG_FOLDER, f"{ name }.png")
        fig.save(file_name)
        return file_name

    def save_word_cloud(name, fig):
        file_name = os.path.join(IMG_FOLDER, f"{ name }.png")
        # Save the image in the img folder:
        wordcloud.to_file(file_name)
        return file_name
    
    #Saving graph and table to html
    hsd_html = save_plotly_plot('historicalprice_line', chart_HistoricalStockData)
    fnot_html = save_plotly_plot('news_line', chart_FrequencyofNewsOverTime)
    ssot_html = save_plotly_plot('sentiment_pie', chart_SentimentScoreOverTime)
    publisher = save_plotly_plot('publiser_bar', chart_Publishers)
    sdbt_html = save_plotly_plot('topics_bar', chart_TopicFrequency)
    wf_html = save_word_cloud('wordcloud', word_frequncy)
    # ssac_html = save_altair_plot('companies_sentiment_bar', chart_SentimentScoreAcrossCompanies)
    hpay_table_html = getTableHTML(table_HighestPriceAcrossYear, False, 1)
    nnac_table_html = getTableHTML(table_NumberofNewsAcrossCompanies, False, 1)
    ssac_table_html = getTableHTML(table_SentimentFrequency, True, 2)
    tp_table_html = getTableHTML(table_TopPublishers, False, 1)

    wkhtml_path = pdfkit.configuration(wkhtmltopdf = '/usr/bin/wkhtmltopdf')
    html = template.render(
        hsd_url = hsd_html,
        fnot_url = fnot_html,
        ssot_url = ssot_html,
        sdbt_url = sdbt_html,
        wf_url = wf_html,
        publishers_url = publisher,
        hpay_table = hpay_table_html,
        nnac_table = nnac_table_html,
        ssac_table = ssac_table_html,
        tp_table = tp_table_html,
        selected_font = final_font_family,
        selected_font_style = font_style_selection,
    )
    try:
        pdf = pdfkit.from_string(html, configuration = wkhtml_path, options = {"enable-local-file-access": "", "zoom": "1.3"})
    except(ValueError, TypeError):
        export_button = st.button('Export⬇️')
        print('Button with label only')

    submit = st.download_button(
                "Export⬇️",
                data=pdf,
                file_name="Stock Prices Report.pdf",
                mime="application/pdf",
            )

    if submit:
        st.balloons()
##########################################################################################################################################
