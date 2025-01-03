import streamlit as st 
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline, GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
from peft import PeftModel
import json

with st.sidebar:

    tickers = ['AAPL', 'AMZN', 'META', 'MSFT', 'TSLA']
    ticker = st.selectbox('Ticker', options=tickers, index=0)

    years = ['2021', '2022', '2023']
    year = st.selectbox('Year', options=years, index=0)
    # date = st.slider('Date', min_value='2021-01-01', max_value='2023-12-31')

st.title('🔎 Financial Analysis')

#load data
@st.cache_data
def load_stock_data():
    with open("stockprices.json", "r") as json_file:
        stockprices_data = json.load(json_file)
    return stockprices_data

@st.cache_data
def load_merge_data():
    with open("merged_data.json", "r") as json_file:
        merged_data = json.load(json_file)
    return merged_data

@st.cache_data
def load_analysis_data():
    with open("stock_analysis.json", "r") as json_file:
        analysis_data = json.load(json_file)
    return analysis_data

df_fn = load_merge_data()
df_sp = load_stock_data()
df_analysis = load_analysis_data()
df_fn = pd.DataFrame(df_fn)
df_sp = pd.DataFrame(df_sp)

#Filter all news and data with related ticker
def filter_data(dataframe, year, companies):
    df = ''
    if dataframe == 'df_sp':
        df = df_sp
    else:
        df = df_fn
    
    df['Date'] = pd.to_datetime(df['Date'])
    if year:
        df = df[df['Date'].dt.year == int(year)]
    if companies:
        df = df[df['company'] == ticker]

    return df

filtered_df_sp = filter_data('df_sp', year, ticker)
filtered_df = filter_data('df_fn', year, ticker)

r1c1, r1c2 = st.columns(2)
with r1c1:
    #Overall Sentiment
    sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    filtered_df['sentiments'] = filtered_df['sentiment_score'].map(sentiment_mapping)
    overall_sentiment_score = filtered_df['sentiments'].mean()
    if overall_sentiment_score > 0:
        overall_sentiment = 'Positive'
    elif overall_sentiment_score < 0:
        overall_sentiment = 'Negative'
    else:
        overall_sentiment = 'Neutral'
    st.markdown('#### :blue[Overall Sentiment]')
    st.write(f'{overall_sentiment} ({overall_sentiment_score:.3f})')

with r1c2:
    #Total News
    total_news = len(filtered_df)
    st.markdown('#### :blue[Total News for Year]')
    st.write(f'{total_news}')

r2c1,r2c2 = st.columns(2)

with r2c1:
    #Overall Stock Volume
    overall_stock_volume = filtered_df_sp['Volume'].mean()
    st.markdown('#### :blue[Average Stock Volume]')
    st.write(f'{overall_stock_volume:.2f}')

with r2c2:
    #Overall Stock Price
    overall_stock_price_avg = filtered_df_sp['Adj Close'].mean()
    st.markdown('#### :blue[Average Stock Prices]')
    st.write(f'{overall_stock_price_avg:.2f}')

#Stock Price Change
starting_stock_price = filtered_df_sp['Open'].iloc[0]  # First available stock price
ending_stock_price = filtered_df_sp['Close'].iloc[-1]
price_change = (ending_stock_price - starting_stock_price) / starting_stock_price * 100
st.markdown(f'#### :blue[Stock Price Change for {year}]')
st.write(f'{price_change:.2f}%')

#Stock Analysis
st.markdown('#### :blue[Overall Stock Analysis:]')
# st.markdown('**[Positive Developments]** \n1. Apple reported its first $100 billion revenue quarter, driven by strong iPhone 12 sales. \n2. The launch of the M1 chip showcased innovation, drawing positive market sentiment. \n3. Apple increased its share buybacks, signaling confidence in its future performance.')
# st.markdown('**[Potential Concerns]** \n 1. Semiconductor shortages disrupted production timelines. \n 2. Regulatory pressure on App Store policies raised questions about long-term profitability. \n 3. Stock faced significant volatility during tech sell-offs, affecting investor sentiment.')
# st.markdown('**[Prediction & Analysis]** \n\n**Prediction:** Steady growth supported by innovation and strong customer demand, tempered by supply chain issues. \n\n**Analysis:** The average closing price reflected resilience, with consistent trading activity during key events like product launches and earnings calls.')

ticker_data = df_analysis.get(ticker)
if ticker_data:
    year_data = ticker_data[str(year)]
    if year_data:
        st.markdown(f"**[Positive Developments]**")
        for point in year_data.get("Positive Development"):
            st.markdown(f"- {point}")
        st.write("")

        st.markdown(f"**[Potential Concerns]**")
        for point in year_data.get("Potential Concerns"):
            st.markdown(f"- {point}")
        st.write("")

        st.markdown(f"**[Prediction & Analysis]**")
        st.markdown(f"**Prediction:** {year_data['Prediction & Analysis']['Prediction']}")
        st.markdown(f"**Analysis:** {year_data['Prediction & Analysis']['Analysis']}")
#####################################################################################
#''''BLOOM'''''

#####################################################################################
# Load Bloom model and tokenizer
# model_name = "bigscience/bloom"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Summarize headlines for AAPL
# headlines = "\n".join([item["title"] for item in filtered_df_fn])
# prompt = f"""
# Summarize the following financial news headlines for AAPL:

# {headlines}

# Trends and key insights:
# """

# # # Tokenize and generate
# inputs = tokenizer(prompt, return_tensors="pt")
# outputs = model.generate(inputs['input_ids'], max_length=300, temperature=0.7)
# summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

# st.write("AAPL Summary:")
# st.write(summary)
#####################################################################################
#''''FinGPT-Forecast'''''

#####################################################################################
# base_model = AutoModelForCausalLM.from_pretrained(
#     'meta-llama/Llama-2-7b-chat-hf',
#     trust_remote_code=True,
#     device_map="auto",
#     torch_dtype=torch.float16,   # optional if you have enough VRAM
# )
# tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')

# model = PeftModel.from_pretrained(base_model, 'FinGPT/fingpt-forecaster_dow30_llama2-7b_lora')
# model = model.eval()

# B_INST, E_INST = "[INST]", "[/INST]"
# B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# SYSTEM_PROMPT = "You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies' stock price movement for the upcoming week. Your answer format should be as follows:\n\n[Positive Developments]:\n1. ...\n\n[Potential Concerns]:\n1. ...\n\n[Prediction & Analysis]:\n...\n"

# prompt = """
# [Company Introduction]:

# {name} is a leading entity in the {finnhubIndustry} sector. Incorporated and publicly traded since {ipo}, the company has established its reputation as one of the key players in the market. As of today, {name} has a market capitalization of {marketCapitalization:.2f} in {currency}, with {shareOutstanding:.2f} shares outstanding. {name} operates primarily in the {country}, trading under the ticker {ticker} on the {exchange}. As a dominant force in the {finnhubIndustry} space, the company continues to innovate and drive progress within the industry.

# From {startDate} to {endDate}, {name}'s stock price {increase/decrease} from {startPrice} to {endPrice}. Company news during this period are listed below:

# [Headline]: ...
# [Summary]: ...

# [Headline]: ...
# [Summary]: ...

# Some recent basic financials of {name}, reported at {date}, are presented below:

# [Basic Financials]:
# {attr1}: {value1}
# {attr2}: {value2}
# ...

# Based on all the information before {curday}, let's first analyze the positive developments and potential concerns for {symbol}. Come up with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from company-related news. Then make your prediction of the {symbol} stock price movement for next week ({period}). Provide a summary analysis to support your prediction.

# """

# prompt = B_INST + B_SYS + {SYSTEM_PROMPT} + E_SYS + {YOUR_PROMPT} + E_INST
# inputs = tokenizer(
#     prompt, return_tensors='pt'
# )
# inputs = {key: value.to(model.device) for key, value in inputs.items()}
        
# res = model.generate(
#     **inputs, max_length=4096, do_sample=True,
#     eos_token_id=tokenizer.eos_token_id,
#     use_cache=True
# )
# output = tokenizer.decode(res[0], skip_special_tokens=True)
# answer = re.sub(r'.*\[/INST\]\s*', '', output, flags=re.DOTALL) # don't forget to import re

#####################################################################################
#''''GPT2'''''

#####################################################################################
# Load the pre-trained GPT-2 model and tokenizer from Hugging Face
# model_name = 'gpt2'  # You can choose other variants like 'gpt2-medium', 'gpt2-large', etc.
# model = GPT2LMHeadModel.from_pretrained(model_name)
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)


# # Example data: headline, sentiment, and stock price
# headlines = filtered_df['title'].head(30).tolist()
headlines = filtered_df['title'].tolist()
# all_headlines = " ".join(headlines)

# # Combine the data into a prompt
# prompt = f"Headline: {all_headlines}\nSentiment: {overall_sentiment_score} \nStock Price: {overall_stock_price_avg} \n\nGenerate positive development, potential concerns and provide analysis for {ticker} in {year}:"

# # Encode the prompt
# inputs = tokenizer.encode(prompt, return_tensors='pt')

# # Generate the stock analysis text
# with torch.no_grad():
#     outputs = model.generate(
#         inputs, 
#         max_new_tokens=300,  # You can adjust the length of the generated output
#         num_return_sequences=1,  # Generate one analysis at a time
#         no_repeat_ngram_size=2,  # Avoid repeating phrases
#         temperature=0.1,  # Control randomness (lower is more deterministic)
#         top_k=50,  # Limit the sampling pool to top-k tokens
#         top_p=0.95,  # Use nucleus sampling (probability mass for candidate selection)
#         pad_token_id=tokenizer.eos_token_id  # Pad with the end-of-sequence token
#     )

# # Decode and print the generated analysis
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Extract the stock analysis part after the "Generate stock analysis:" prompt
# # Check if the expected "Generate stock analysis:" prompt exists in the output
# if "Generate stock analysis:" in generated_text:
#     # Extract the stock analysis part after the "Generate stock analysis:" prompt
#     generated_analysis = generated_text.split("Generate stock analysis:")[1].strip()
# else:
#     # If "Generate stock analysis:" is not found, use the entire generated text
#     generated_analysis = generated_text

# st.subheader(':blue[Overall Stock Analysis:]')
# st.write(generated_analysis)

#####################################################################################
#''''GPT-Neo'''''

#####################################################################################

# generator = pipeline('text-generation',
# model='EleutherAI/gpt-neo-1.3B')

# text = generator(prompt , do_sample=True, max_length=200)

# print(text[0]['generated_text'])

#####################################################################################
#''''GPT 4.0'''''

#####################################################################################
# from openai_unofficial import OpenAIUnofficial
# # Initialize the client
# client = OpenAIUnofficial()

# # Basic chat completion
# response = client.chat.completions.create(
#     messages=[{"role": "user", "content": f"Provide Positive Developments, Potential Concerns, and a Prediction & Analysis for {ticker} in Year {year} based on Headline: {headlines}\nSentiment: {overall_sentiment_score} \nStock Price: {overall_stock_price_avg}"}],
#     model="gpt-4o"
# )
# # print(response.choices[0].message.content)
# st.markdown('#### :blue[Overall Stock Analysis:]')
# print(response.choices[0].message.content)
# st.write(response.choices[0].message.content)
