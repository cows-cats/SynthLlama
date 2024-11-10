import streamlit as st
import pandas as pd
import requests
import json

st.title('Synthetic Data Generator')

# File upload
uploaded_file = st.file_uploader("Upload a PDF file (optional)", type="pdf")

st.header('Describe Your Data Request')
data_description = st.text_area("Data description")

# Get format options from backend
formats_response = requests.get("http://localhost:8000/formats")
format_options = formats_response.json()["formats"]

st.header('Data Format')
format_type = st.selectbox(
    'Select the data format',
    format_options
)

st.header('Data Amount')
data_amount = st.selectbox(
    'Select the number of records',
    [10, 50, 100, 500]
)

if st.button('Generate Data'):
    if not data_description and not uploaded_file:
        st.error('Please either enter a description or upload a PDF')
    else:
        # Prepare request
        files = {"file": uploaded_file} if uploaded_file else None
        data = {
            "format_type": format_type,
            "data_amount": data_amount,
            "description": data_description
        }
        
        # Make request to backend
        response = requests.post(
            "http://localhost:8000/process",
            files=files,
            data={"request": json.dumps(data)}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result["status"] == "success":
                # Parse and display the generated data
                generated_data = result["data"]
                try:
                    df = pd.DataFrame(json.loads(generated_data))
                    st.header('Generated Data')
                    st.dataframe(df)
                    
                    # Download button for CSV
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download as CSV",
                        data=csv,
                        file_name="synthetic_data.csv",
                        mime="text/csv"
                    )
                except:
                    st.text("Generated Response:")
                    st.text(generated_data)
            else:
                st.error(f"Error: {result.get('message', 'Unknown error')}")
        else:
            st.error(f"Failed to communicate with backend server {response.json()} ")

            