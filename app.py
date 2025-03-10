import streamlit as st
import pandas as pd
from sqlalchemy.sql import text
import os
# import firebase_admin
# from firebase_admin import credentials, auth, firestore
import pyrebase
import json

# Page configuration
st.set_page_config(
    page_title="Sentiment Analyzer Dashboard",
    page_icon=":bar_chart:",
    layout="wide")

#initialize firebase
config = {
    "apiKey": st.secrets["firebase"]["apiKey"],
    "authDomain": st.secrets["firebase"]["authDomain"],
    "projectId": st.secrets["firebase"]["projectId"],
    "databaseURL": st.secrets["firebase"]["databaseURL"],
    "storageBucket": st.secrets["firebase"]["storageBucket"],
    "messagingSenderId": st.secrets["firebase"]["messagingSenderId"],
    "appId": st.secrets["firebase"]["appId"],
    "measurementId": st.secrets["firebase"]["measurementId"],
}
# fb_credentials = st.secrets["firebase"]['my_project_settings']
firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
db = firebase.database()

#######################################################################################################

#######################################################################################################

if "role" not in st.session_state:
    st.session_state.role = None

if "username" not in st.session_state:
    st.session_state.username = None

if "email" not in st.session_state:
     st.session_state.email = None

if "localID" not in st.session_state:
     st.session_state.localID = None

if "user" not in st.session_state:
     st.session_state.user = None

ROLES = [None, "User", "Guest"]

#Sign up dialog to create new user account
@st.dialog("Create New Account")
def signup():

    new_email = st.text_input("Email", key="signupEmail")
    new_username = st.text_input("Username", key='signupUsername')
    new_password = st.text_input("Password (Please include a uppercase, lowercase, numeric and non-alphanumeric characters)", key='signupPassword', type='password')
    new_confirm_password = st.text_input("Confirm Password (Please include a uppercase, lowercase, numeric and non-alphanumeric characters)", key='signupConfirmPassword', type='password')

    signupButton = st.button("Sign up")

    if signupButton:
        if not new_username or not new_email or not new_password or not new_confirm_password:
            st.error('Please fill in all fields.')
        elif new_password != new_confirm_password:
             st.error("Password do not match!")
        else:
            try:
                # Create user in Firebase Authentication
                user = auth.create_user_with_email_and_password(email=new_email, password=new_password)
                #Set id and username in real-time database
                db.child(user['localId']).child("Username").set(new_username)
                db.child(user['localId']).child("ID").set(user['localId'])
                st.success("You have successfully created a valid Account")
                st.info('Please proceed to login using email and password.')
            except Exception as e:
                    st.error(f"Error: {json.loads(e.args[1])['error']['message']}")

#A dialog for user who forgot the password
@st.dialog('Forgot your password?')
def forgotPassword():
    email  = st.text_input('Please provide your email')
    submit = st.button('Submit', key='forgotPasswordSubmitButton')
    if submit:
        if email:
            try:
                # Send password reset email
                auth.send_password_reset_email(email)
                st.success("Password reset email sent! Check your inbox.")
            except Exception as e:
                st.error(f"Error: {json.loads(e.args[1])['error']['message']}")
        else:
            st.error("Please enter your email address.")

#Home function (show login page first)
def home():
    # st.title('🎉 :blue[Finalyze]')
    st.markdown("<h1 class='custom-title' style='text-align:center'>🎉 <span style='color: LightBlue'>Finalyze</span></h1>", unsafe_allow_html=True)
    st.divider()
    chooseRole, abtUs = st.columns([2,3])
    with chooseRole:
        with st.container(border=True):
            role = st.selectbox("Choose your role", ROLES)
            if role == "User":
                st.subheader("Log in", divider=True)
                loginEmail = st.text_input("Email", key='loginEmail')
                # loginUsername = st.text_input("Username", key='loginUsername')
                loginPassword = st.text_input("Password", key='loginPassword', type='password')

                col1, col2 = st.columns([1.75,1])
                with col1:
                    login_Button = st.button("Login")
                with col2:
                    st.button('Forgot Password', on_click=forgotPassword)

                st.write('')

                col3, col4, col5 = st.columns(3)
                with col4:
                    st.button('New User? Click here.', on_click=signup)

                if login_Button:
                    try:
                        user = auth.sign_in_with_email_and_password(email=loginEmail, password=loginPassword)
                        st.write('signing in with email and password')
                        st.write(user)
                        # Fetch user details from Firestore
                        # user_data = db.child(user['localId']).child("Username").get().val() 
                        st.session_state.role = role
                        # st.session_state.username = user_data
                        st.session_state.email = loginEmail
                        st.session_state.localID = user['localId']
                        st.session_state.user = user
                        st.rerun()
                    except Exception as e:
                        # st.error(f"Error: {json.loads(e.args[1])['error']['message']}")
                        st.error(f"Full Error: {e}")
                        st.warning('Incorrect Username/Password.')
            elif role == "Guest":
                if st.button("Log in"):
                    st.session_state.role = role
                    st.session_state.username = None
                    st.session_state.email = None
                    st.session_state.localID = None
                    st.rerun()

    with abtUs:
        st.subheader('💡About Finalyze💡')
        st.write('- Finalyze is  website that analyze stock prices and online news using sentiment analysis tool.')
        st.markdown('- **:rainbow[Sentiment analysis]** is the process of classifying whether a block of text is :red[positive], :green[negative], or :blue[neutral].')
        st.markdown("<p style='text-align:center'><b>How to choose your role?</b></p>", unsafe_allow_html=True)
        dataframe = pd.DataFrame({
            "Functionalities": ["Data Visualization", "Customize & Filter Data", "Report Generator", "Profile Editing", "Sentiment Analysis", "Data Insights"],
            "🎩User": ['✅' ,'✅' ,'✅' ,'✅' ,'✅' ,'✅'],
            "👤Guest": ['✅', '❌', '❌', '❌', '❌', '❌']
        })
        dataframe.index = [''] * len(dataframe)
        st.table(dataframe)
    
    st.markdown("<footer><p class='footer1' style='text-align:left'>📧 Contact Us: u2102802@siswa.um.edu.my</p></footer>", unsafe_allow_html=True)
    st.markdown("<footer><p class='footer2' style='text-align:left'>⚠️ Disclaimer: This tool is for informational purposes only. Use it responsibly!</p></footer>", unsafe_allow_html=True)

def logout():
    st.session_state.role = None
    st.session_state.username = None
    st.session_state.email = None
    st.session_state.localID = None
    st.session_state.user = None
    st.rerun()

role = st.session_state.role
username = st.session_state.username
email = st.session_state.email
user = st.session_state.localID
user = st.session_state.user

logout_page = st.Page(logout, title="Log out", icon=":material/logout:")
# settings_page = st.Page("users/edit_profile.py", title="Edit Profile", icon=":material/edit:")

user= st.Page(
    "users/user_dashboard.py",
    title="Dashboard",
    icon=":material/dashboard:",
    default=(role == "User"),
)

user_SA = st.Page(
     "users/sentiment_analyzer.py",
     title="Sentiment Analyzer",
     icon= ':material/insert_emoticon:',
)

user_analysis = st.Page(
    "users/analysis.py",
    title="Analysis",
    icon= ':material/analytics:',
)

user_help = st.Page(
    "users/help.py",
    title="Help Centre",
    icon=':material/help:'
)

guest = st.Page(
    "guest/guest_dashboard.py",
    title="Dashboard",
    icon=":material/home:",
    default=(role == "Guest"),
)

account_pages = [logout_page]
users_pages = [user, user_SA, user_analysis, user_help]
guest_pages = [guest]

# st.title("Request manager")
# st.logo("images/horizontal_blue.png", icon_image="images/icon_blue.png")
page_dict = {}
if st.session_state.role in ["User"]:
    page_dict["User"] = users_pages
if st.session_state.role in ["Guest"]:
    page_dict["Guest"] = guest_pages

if len(page_dict) > 0:
    pg = st.navigation({"": account_pages} | page_dict)
else:
    pg = st.navigation([st.Page(home)])

pg.run()
