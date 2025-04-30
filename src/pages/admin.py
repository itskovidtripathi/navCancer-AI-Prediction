import streamlit as st
import pandas as pd
from database import Database
import os
from datetime import datetime

def admin_login():
    """Handle admin login"""
    st.title("🔐 Admin Login")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        db = Database()
        if db.verify_admin(username, password):
            st.session_state['admin_logged_in'] = True
            st.rerun()
        else:
            st.error("Invalid credentials!")

def view_predictions():
    """Display all predictions"""
    st.title("📊 Prediction Records")
    
    db = Database()
    predictions = db.get_all_predictions()
    
    if not predictions:
        st.info("No predictions found in the database.")
        return
    
    # Convert predictions to DataFrame
    df = pd.DataFrame(predictions, columns=[
        'Name', 'Email', 'Phone',
        'Prediction', 'Confidence',
        'Date', 'Image Path', 'Report Path'
    ])
    
    # Add view buttons for image and report
    st.markdown("### All Predictions")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        name_filter = st.text_input("Filter by Name")
    with col2:
        email_filter = st.text_input("Filter by Email")
    with col3:
        prediction_filter = st.selectbox(
            "Filter by Prediction",
            ["All"] + list(df['Prediction'].unique())
        )
    
    # Apply filters
    filtered_df = df.copy()
    if name_filter:
        filtered_df = filtered_df[filtered_df['Name'].str.contains(name_filter, case=False)]
    if email_filter:
        filtered_df = filtered_df[filtered_df['Email'].str.contains(email_filter, case=False)]
    if prediction_filter != "All":
        filtered_df = filtered_df[filtered_df['Prediction'] == prediction_filter]
    
    # Display statistics
    st.markdown("### Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Users", len(df['Email'].unique()))
    with col2:
        st.metric("Total Predictions", len(df))
    with col3:
        st.metric("Cancer Cases", len(df[df['Prediction'] != "Normal"]))
    with col4:
        st.metric("Normal Cases", len(df[df['Prediction'] == "Normal"]))
    
    # Display predictions table
    st.markdown("### Detailed Records")
    for index, row in filtered_df.iterrows():
        with st.expander(f"Prediction for {row['Name']} - {row['Date']}"):
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.image(row['Image Path'], caption="X-ray Image", use_column_width=True)
            
            with col2:
                st.markdown(f"**Patient Details:**")
                st.write(f"Name: {row['Name']}")
                st.write(f"Email: {row['Email']}")
                st.write(f"Phone: {row['Phone']}")
                st.markdown("---")
                st.markdown(f"**Diagnosis Details:**")
                st.write(f"Prediction: {row['Prediction']}")
                st.write(f"Confidence: {row['Confidence']:.2%}")
                st.write(f"Date: {row['Date']}")
                
                # Add buttons to view/download report
                if os.path.exists(row['Report Path']):
                    with open(row['Report Path'], 'r') as f:
                        report_content = f.read()
                        st.download_button(
                            label="📄 Download Report",
                            data=report_content,
                            file_name=f"report_{row['Name']}_{datetime.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain",
                            key=f"download_button_{index}_{row['Date'].replace(' ', '_')}"
                        )

def admin_panel():
    """Main admin panel function"""
    if 'admin_logged_in' not in st.session_state:
        st.session_state['admin_logged_in'] = False
    
    if not st.session_state['admin_logged_in']:
        admin_login()
    else:
        st.sidebar.title("Admin Panel")
        if st.sidebar.button("Logout"):
            st.session_state['admin_logged_in'] = False
            st.rerun()
        
        view_predictions()

if __name__ == "__main__":
    admin_panel() 