import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration including title, icon, and layout
st.set_page_config(page_title="Credit Risk Modelling", page_icon=":moneybag:", layout='wide')

# CSS styling for customizing the layout
st.markdown("""
<style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #0071E3;
        text-align: center;
        background-color: #FFD700; /* Bright yellow background */
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 20px;
        font-weight: bold;
        color: #4CAF50; /* Green color */
        text-align: center;
        margin-bottom: 10px;
    }
    .info {
        font-size: 16px;
        color: #555555;
        text-align: center;
        margin-bottom: 20px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 20px;
    }
    th, td {
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }
    th {
        background-color: #0071E3;
        color: white;
    }
    tr:hover {
        background-color: #f2f2f2;
    }
</style>
""", unsafe_allow_html=True)

# Title, subtitle, and info
st.markdown('<h1 class="title"> Loan Approval Estimator</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Discover your chances of credit approval with our advanced prediction model, providing insights into borrower risk assessment and categorization</p>', unsafe_allow_html=True)
st.markdown('<p class="info">Unveil the models ability to gauge the probability of credit default and segment borrowers into four distinct risk categories (P1, P2, P3, P4), enabling informed decision-making and proactive risk management</p>', unsafe_allow_html=True)

st.subheader("")
st.write("Note: Make sure the excel file contains the following columns with defined values and the column names are the same as the names in Variable Name: ")
columns_df = pd.read_excel("info.xlsx")
num_columns = len(columns_df.columns)
half_columns = num_columns // 2
col1, col2 = st.columns(2)
col1.table(columns_df.iloc[:, :half_columns])
col2.table(columns_df.iloc[:, half_columns:])
# Load the trained model
model = pd.read_pickle('best_xgb_model.pkl')

# Define column names
cols_in_df = ['pct_tl_open_L6M', 'pct_tl_closed_L6M', 'Tot_TL_closed_L12M', 'pct_tl_closed_L12M', 'Tot_Missed_Pmnt', 
              'CC_TL', 'Home_TL', 'PL_TL', 'Secured_TL', 'Unsecured_TL', 'Other_TL', 'Age_Oldest_TL', 'Age_Newest_TL', 
              'time_since_recent_payment', 'max_recent_level_of_deliq', 'num_deliq_6_12mts', 'num_times_60p_dpd', 
              'num_std_12mts', 'num_sub', 'num_sub_6mts', 'num_sub_12mts', 'num_dbt', 'num_dbt_12mts', 'num_lss', 
              'recent_level_of_deliq', 'CC_enq_L12m', 'PL_enq_L12m', 'time_since_recent_enq', 'enq_L3m', 
              'NETMONTHLYINCOME', 'Time_With_Curr_Empr', 'CC_Flag', 'PL_Flag', 'pct_PL_enq_L6m_of_ever', 
              'pct_CC_enq_L6m_of_ever', 'HL_Flag', 'GL_Flag', 'MARITALSTATUS', 'EDUCATION', 'GENDER', 
              'last_prod_enq2', 'first_prod_enq2']

st.set_option('deprecation.showPyplotGlobalUse', False)
# Function to preprocess and predict loan approval

def plot_graphs(df):
    col1,col2 = st.columns(2)
    fig, ax = plt.subplots(figsize=(6, 5.20))
    palette_color = sns.color_palette("deep")

    with col1:
            st.subheader("Distribution of Predicted Outcomes: ")
            plt.figure()
            plt.pie(df['Approved Flag'].value_counts(), labels=df['Approved Flag'].value_counts().index,autopct='%1.1f%%',colors=palette_color)
            plt.axis('equal')
            st.pyplot()
            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader("Gender distribution in the data: ")
        
            plt.figure()
            plt.pie(df['GENDER'].value_counts(),labels=df['GENDER'].value_counts().index, autopct='%1.1f%%',colors=palette_color)
            plt.axis('equal')
            st.pyplot()
            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader("Education distribution in the data: ")
            
            plt.figure()
            plt.pie(df['EDUCATION'].value_counts(),labels=df['EDUCATION'].value_counts().index,autopct='%1.1f%%',colors=palette_color)
            plt.axis('equal')
            st.pyplot()
            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader("Distribution of Last product enquired for:")
            plt.figure()
            palette_color = sns.color_palette("deep")
            plt.pie(df['last_prod_enq2'].value_counts(),labels=df['last_prod_enq2'].value_counts().index,colors=palette_color,autopct='%.0f%%')
            plt.axis('equal')
            st.pyplot()
            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader("Distribution of first product enquired for: ")
        
            plt.figure()
            plt.pie(df['first_prod_enq2'].value_counts(),labels=df['first_prod_enq2'].value_counts().index,colors=palette_color,autopct='%.0f%%')
            plt.axis('equal')
            st.pyplot()

    with col2:
            st.subheader("")
            st.subheader("")
            plt.figure()
            sns.barplot(df['Approved Flag'].value_counts())
            st.pyplot()

            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader("")
            plt.figure()
            sns.barplot(df['GENDER'].value_counts())
            st.pyplot()

            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader("")
            
            
            plt.figure()
            sns.barplot(df['EDUCATION'].value_counts(),orient='h',ax=ax)
            st.pyplot(fig)

            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader("")
            plt.figure()
            palette_color = sns.color_palette("deep")
            sns.barplot(df['last_prod_enq2'].value_counts())
            st.pyplot()
            st.subheader("")
            st.subheader("")
            st.subheader("")
            st.subheader('')
            st.subheader("")
        
            plt.figure()
            palette_color = sns.color_palette("deep")
            sns.barplot(df['first_prod_enq2'].value_counts())
            st.pyplot()    

def predict_loan_approval(file):
    try:
        # Read the uploaded Excel file
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        df2 = df.copy()
        # Preprocess the data
      

        df2['EDUCATION'] = df2['EDUCATION'].map({"SSC":5, "10TH":5, "12TH":0, "GRADUATE":1, "UNDER GRADUATE":6, 
                                                "POST-GRADUATE":3, "OTHERS":2, "PROFESSIONAL":4})
        df2['EDUCATION'] = df2['EDUCATION'].astype(int)

        # Perform one-hot encoding
        df_encoded = pd.get_dummies(df2, columns=['MARITALSTATUS',  'GENDER', 
              'last_prod_enq2', 'first_prod_enq2'])
        df_encoded_unseen = df_encoded[['pct_tl_open_L6M', 'pct_tl_closed_L6M', 'Tot_TL_closed_L12M', 'pct_tl_closed_L12M', 'Tot_Missed_Pmnt', 'CC_TL', 'Home_TL', 'PL_TL', 'Secured_TL', 'Unsecured_TL', 'Other_TL', 'Age_Oldest_TL', 'Age_Newest_TL', 'time_since_recent_payment', 'max_recent_level_of_deliq', 'num_deliq_6_12mts', 'num_times_60p_dpd', 'num_std_12mts', 'num_sub', 'num_sub_6mts', 'num_sub_12mts', 'num_dbt', 'num_dbt_12mts', 'num_lss', 'recent_level_of_deliq', 'CC_enq_L12m', 'PL_enq_L12m', 'time_since_recent_enq', 'enq_L3m', 'NETMONTHLYINCOME', 'Time_With_Curr_Empr', 'CC_Flag', 'PL_Flag', 'pct_PL_enq_L6m_of_ever', 'pct_CC_enq_L6m_of_ever', 'HL_Flag', 'GL_Flag', 'EDUCATION', 'MARITALSTATUS_Married', 'MARITALSTATUS_Single', 'GENDER_F', 'GENDER_M', 'last_prod_enq2_AL', 'last_prod_enq2_CC', 'last_prod_enq2_ConsumerLoan', 'last_prod_enq2_HL', 'last_prod_enq2_PL', 'last_prod_enq2_others', 'first_prod_enq2_AL', 'first_prod_enq2_CC', 'first_prod_enq2_ConsumerLoan', 'first_prod_enq2_HL', 'first_prod_enq2_PL', 'first_prod_enq2_others']]
        # Make predictions
     
        predictions = model.predict(df_encoded_unseen)
        
        # Map prediction labels
        prediction_labels = {1:'P2', 2:'P3', 3:'P4', 0:'P1'}
        df['Approved Flag'] = [prediction_labels[prediction] for prediction in predictions]
        
        # Download the predicted results as an Excel file
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        excel_data = excel_buffer.getvalue()
        
        # Display download button for the Excel file
        st.download_button(label="Download Predicted Results", data=excel_data, file_name='predicted_results.xlsx', 
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
                           help="Click to download the predicted results as an Excel file.")
        
        # Display success message
        st.success("Credit approval predictions generated successfully!")
        #st.write(df['Target_Variable'])

        plot_graphs(df)

        

    except Exception as e:
        st.error(f"An error occurred: {e}")

# File uploader widget
uploaded_file = st.file_uploader("Upload Excel file", type=["xls", "xlsx","csv"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Perform loan approval prediction
    predict_loan_approval(uploaded_file)


































