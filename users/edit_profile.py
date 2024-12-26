import streamlit as st
import pyrebase
import json

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
user = auth.current_user
st.write(user)

@st.dialog('Forgot your password?')
def resetPassword():
    # Send password reset email
    auth.send_password_reset_email(f"{st.session_state.email}")
    st.success("Password reset email sent! Check your inbox.")

st.header("Update Your Profile")
st.write(f"Your username is :violet[{st.session_state.username}].")
editprofileButton = st.button('Edit Profile', key='editProfileButton')
email = st.text_input("Email", value=f"{st.session_state.email}", disabled=True)
resetPasswordButton = st.button('Reset Password',key='resetPasswordButton', on_click=resetPassword)

if editprofileButton:
    # Form for profile updates
    with st.form(key='profile_form'):
        # Username and Password change fields
        newUsername = st.text_input("Username", value=f"{st.session_state.username}")

        # Submit button 
        submit_button = st.form_submit_button("Save Changes")

        if submit_button:
            if (newUsername != st.session_state.username) and (newUsername != ''):
                db.child(user['localId']).child("Username").set(newUsername)
                st.session_state.username = newUsername
            else:
                st.warning("Please provide a valid username.")