import streamlit as st
import pandas as pd
from sqlalchemy.sql import text
import os
# import firebase_admin
# from firebase_admin import credentials, auth, firestore
import pyrebase
import json

#initialize firebase
config = {
  "apiKey": "AIzaSyAaRg0x_NjEc-xPfgXA0DW0wgmkzq8rIMw",
  "authDomain": "sentimentanalyzer-4bd42.firebaseapp.com",
  "projectId": "sentimentanalyzer-4bd42",
  "databaseURL": "https://sentimentanalyzer-4bd42-default-rtdb.asia-southeast1.firebasedatabase.app/",
  "storageBucket": "sentimentanalyzer-4bd42.firebasestorage.app",
  "messagingSenderId": "694050098907",
  "appId": "1:694050098907:web:a0f7e83f9e997350fdc6fe",
  "measurementId": "G-QJMP88TBJ1"
}
firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
db = firebase.database()
storage = firebase.storage()

# Page configuration
st.set_page_config(
    page_title="Sentiment Analyzer Dashboard",
    page_icon="📈",
    layout="wide")

#######################################################################################################
# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

#######################################################################################################

if "role" not in st.session_state:
    st.session_state.role = None

if "username" not in st.session_state:
    st.session_state.username = None

if "email" not in st.session_state:
     st.session_state.email = None

ROLES = [None, "User", "Guest"]

#Sign up dialog to create new user account
@st.dialog("Create New Account")
def signup():

    new_email = st.text_input("Email", key="signupEmail")
    new_username = st.text_input("Username", key='signupUsername')
    new_password = st.text_input("Password", key='signupPassword', type='password')
    new_confirm_password = st.text_input("Confirm Password", key='signupConfirmPassword', type='password')

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
                    st.error("Error!")

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
    st.title('🎉 Welcome to :blue[Sentiment Analyzer Dashboard]')
    st.divider()
    role = st.selectbox("Choose your role", ROLES)
    if role == "User":
        st.subheader("Log in", divider=True)
        loginEmail = st.text_input("Email", key='loginEmail')
        # loginUsername = st.text_input("Username", key='loginUsername')
        loginPassword = st.text_input("Password", key='loginPassword', type='password')

        col1, col2 = st.columns([3.5, 1])
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
                 # Fetch user details from Firestore
                db.child(user['localId']).child("Username").get()
                user_data = db.child(user['localId']).child("Username").get().val()
                st.write(user_data)
                st.session_state.role = role
                st.session_state.username = user_data
                st.session_state.email = loginEmail
                st.rerun()
            except Exception as e:
                st.error(f"Error: {json.loads(e.args[1])['error']['message']}")
                st.warning('Incorrect Username/Password.')
    elif role == "Guest":
        if st.button("Log in"):
            st.session_state.role = role
            st.session_state.username = None
            st.session_state.email = None
            st.rerun()

def logout():
    st.session_state.role = None
    st.session_state.username = None
    st.session_state.email = None
    st.rerun()

role = st.session_state.role
username = st.session_state.username
email = st.session_state.email

logout_page = st.Page(logout, title="Log out", icon=":material/logout:")
settings_page = st.Page("users/edit_profile.py", title="Edit Profile", icon=":material/edit:")

user= st.Page(
    "users/user_dashboard.py",
    title="Dashboard",
    icon=":material/home:",
    default=(role == "User"),
)

user_SA = st.Page(
     "users/sentiment_analyzer.py",
     title="Sentiment Analyzer",
     icon= ':material/analytics:',
)

user_analysis = st.Page(
    "users/analysis.py",
    title="Analysis",
    icon= ':material/analytics:',
)

guest = st.Page(
    "guest/guest_dashboard.py",
    title="Dashboard",
    icon=":material/home:",
    default=(role == "Guest"),
)

account_pages = [logout_page]
users_pages = [user, user_SA, settings_page]
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