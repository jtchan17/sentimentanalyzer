import streamlit as st
import pyrebase
import time

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

if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False

@st.dialog('Forgot your password?')
def resetPassword():
    # Send password reset email
    auth.send_password_reset_email(f"{st.session_state.email}")
    st.success("Password reset email sent! Check your inbox.")

st.header("⚙️ Update Your Profile")
st.write(f"Your username is :violet[{st.session_state.username}].")
editprofileButton = st.button('Edit Profile', key='editProfileButton')
email = st.text_input("Email", value=f"{st.session_state.email}", disabled=True)
resetPasswordButton = st.button('Reset Password',key='resetPasswordButton', on_click=resetPassword)

if editprofileButton:
    st.session_state.edit_mode = True

if st.session_state.edit_mode:
    # Form for profile updates
    with st.form(key='profile_form'):
        # Username and Password change fields
        newUsername = st.text_input("Username", value=f"{st.session_state.username}")

        # Submit button with session state tracking
        if "submit_clicked" not in st.session_state:
            st.session_state.submit_clicked = False
            
        # Submit button 
        submit_button = st.form_submit_button("Save Changes")

        if submit_button:
            st.session_state.submit_clicked = True

        if st.session_state.submit_clicked:
            if newUsername.strip() == "":
                st.error("Username cannot be empty.")
                st.session_state.submit_clicked = False
            elif newUsername == st.session_state.username:
                st.error("New username cannot be same as old username.")
                st.session_state.submit_clicked = False
            else:
                try:
                    db.child(st.session_state.localID).update({"Username": newUsername})
                    st.session_state.username = newUsername
                    st.success(f"Username successfully updated to: {newUsername}")
                    st.session_state.submit_clicked = False
                    st.session_state.edit_mode = False
                    time.sleep(2)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error updating username: {str(e)}")
                    st.session_state.submit_clicked = False