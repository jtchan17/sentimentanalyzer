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

@st.dialog('Forgot your password?')
def resetPassword():
    # Send password reset email
    auth.send_password_reset_email(f"{st.session_state.email}")
    st.success("Password reset email sent! Check your inbox.")

st.header("Update Your Profile")
st.write(f"Your username is :violet[{st.session_state.username}].")
editprofileButton = st.button('Edit Profile', key='editProfileButton')
resetPasswordButton = st.button('Reset Password',key='resetPasswordButton', on_click=resetPassword)
email = st.text_input("Email", value=f"{st.session_state.email}", disabled=True)

if editprofileButton:
    # Form for profile updates
    with st.form(key='profile_form'):
        # Username and Password change fields
        username = st.text_input("Username", value=f"{st.session_state.username}")
        new_password = st.text_input("New Password", type='password')
        confirm_password = st.text_input("Confirm New Password", type='password')

        # Submit button 
        submit_button = st.form_submit_button("Save Changes")

        if submit_button:
            if username != st.session_state.username:
                db.child(user['localId']).child("Username").set(username)
                st.session_state.username = username
            elif new_password and confirm_password:
                if new_password == confirm_password:
                    try:
                        # Send password reset email
                        auth.send_password_reset_email(email)
                        st.success("Password reset email sent! Check your inbox.")
                    except Exception as e:
                        st.error(f"Error: {json.loads(e.args[1])['error']['message']}")
                else:
                    st.error('Passwords do not match!')
            else:
                st.warning("Please provide a valid username and password.")