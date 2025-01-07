import streamlit as st

st.title("Help Centre")

st.header("Dashboard Page", divider=True)
st.write("💡Purpose")
st.write("The Dashboard page provides users with an interactive platform to explore historical stock price data and related financial news.")
st.subheader("What’s included on the Dashboard page?")
st.markdown("- **Historical Stock Data:** \nThe stock price data displayed on this page is sourced from **Yahoo Finance**. Users can view the daily, weekly, or monthly stock prices for various companies over specific time periods.")
st.markdown("- **Financial News:** \nThe latest news headlines relevant to the selected companies are gathered through web scraping from **Google News**, allowing users to stay informed about market trends and company-specific updates.")
st.markdown("- **Filtering and Customization Options** \nUsers can filter the data by **company name** (ticker symbol) and year to customize their view. This makes it easy to analyze the stock performance of a specific company over a particular period.")
st.subheader("🔧How to Filter It?")
st.write("1. Select companies (e.g., AAPL, MSFT, TSLA) from the dropdown list.")
st.write("2. Choose the year you want to analyze.")
st.write("3. Select which sentiments to show (e.g., positive, negative, neutral) from dropdown list.")
st.write("4. Select which topics to show (e.g. Politics, Technology, Sports) from dropdown list.")
st.write("5. View the filtered historical stock prices along with relevant news articles to gain insights into market trends.")
st.subheader("🔧How to Customize It?")
st.write("1. Choose font family (e.g., serif, monospace, sans-serif) from the dropdown list.")
st.write("2. Choose font style (e.g. italic) from dropdown list.")
st.write("3. Select specified colour (e.g., red, purple, orange) for company to be visualized in graphs.")
st.write("4. Select specified colour (e.g., red, purple, orange) for sentiment to be visualized in graphs.")
st.write("5. View the customized graphs in the dashboard.")

st.header("Sentiment Analysis", divider=True)
st.write("💡Purpose")
st.write("The Sentiment Analysis page helps users understand the overall market sentiment (positive, negative, or neutral) regarding a particular company based on the news headlines and articles collected.")
st.subheader("What’s included on the Sentiment Analysis page?")
st.markdown("- **Sentiment Score:** \nThe sentiment analysis tool uses natural language processing (NLP) to analyze financial news and headlines. It assigns a sentiment score to each news article, indicating whether the sentiment is positive, negative, or neutral.")
st.subheader("🔧How to Use It?")
st.write("1. Copy an online news headline and paste it to the text field.")
st.write("2. Click on predict button")
st.write("3. The page will display a sentiment score based on headline input")

st.header("Analysis", divider=True)
st.write("💡Purpose")
st.write("The Analysis page provides detailed insights and predictions for each company based on stock performance and sentiment data.")
st.subheader("What’s included on the Analysis page?")
st.markdown("- **Yearly Company Insights:** \nFor each company, users can see positive developments, potential concerns, and predictions & analysis for a specific year. This includes key events that impacted the stock price, risks to be aware of, and an outlook for future performance.")
st.markdown("- **Financial Insights:** \nThe analysis is based on both historical stock data and sentiment trends, giving users a comprehensive view of the company's financial standing and market perception.")
st.subheader("🔧How to Use It?")
st.write("1. Select a company and year you want to analyze.")
st.write("2. Review the detailed insights to understand key developments, risks, and future outlook.")
st.write("3. Use this information to make better-informed investment or business decisions.")

st.header("Edit Profile", divider=True)
st.write("💡Purpose")
st.write("The Edit Profile page allows users to personalize their account by changing their username. This is useful for maintaining up-to-date user information.")
st.subheader("What’s included on the Edit Profile page?")
st.markdown("- **Username Update:** \nUsers can **update their username** to reflect any changes (e.g., a new display name or nickname).")
st.markdown("- **Email Address (Read-Only):** \nThe email address associated with the user’s account is displayed but **cannot be edited** for security reasons.")
st.markdown("- **Password Reset:** \nA Reset Password button is provided. Clicking this button will trigger an automated email to the user’s registered email address with instructions to reset their password.")
st.subheader("🔧How to Use It?")
st.write("1. Go to the Edit Profile page.")
st.write("2. Enter your new desired username in the text field and save your changes.")
st.write("3. To reset your password, click the Reset Password button.")
st.write("4. Check your registered email inbox for a password reset link and follow the instructions to change your password.")

st.header("Overall Guidance for Users")
st.write("This Help Centre is designed to give users a clear understanding of the app's features and how they can use each page effectively. Whether you're exploring historical stock data, analyzing sentiment trends, or viewing detailed insights for companies, each page offers valuable tools to support your financial research.")
st.write("If you need further assistance, refer to the relevant sections in this Help Centre for a step-by-step guide on how to navigate and use the app.")