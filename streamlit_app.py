# Import necessary libraries
import streamlit as st
from transformers import AutoTokenizer, BartForConditionalGeneration

# Set Streamlit page configuration, including title and icon
st.set_page_config(
    page_title='Text Summarization with BART',
    page_icon=':sunglasses:',
    layout='wide'
)

# Create a Markdown section for the application header
st.markdown("""
    <div style="text-align:center;">
        <h1>Text Summarization with BART</h1>
        <P>This Streamlit application allows you to generate a summary of a text using the BART model.</P>
    </div>
""", unsafe_allow_html=True)

# Create a space for content
st.write("")

# Function to load tokenizer from Hugging Face Transformers
@st.cache_resource
def load_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
        return tokenizer
    except Exception as e:
        return None, str(e)

# Load the tokenizer
tokenizer = load_tokenizer()

# Function to load BART model from Hugging Face Transformers
@st.cache_resource
def load_model():
    try:
        model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")
        return model
    except Exception as e:
        return None, str(e)

# Load the BART model
model = load_model()

# Create two columns for layout
col1, col2 = st.columns(2)

# Inside the first column (col1)
with col1:

    # Create a text area for user input
    input_text = st.text_area("Text:", height=200, max_chars=1024)

    # Create an expander for advanced options
    with st.expander("Advanced Options"):
        num_beams = st.slider("Number of Beams:", min_value=1, max_value=10, value=2)
        min_length = st.slider("Minimum Length of Summary:", min_value=1, max_value=256, value=30)
        max_length = st.slider("Maximum Length of Summary:", min_value=30, max_value=512, value=256)

    # Create a space for content
    st.write("")

    # Create a button to generate the summary
    generate_button = st.button('Generate Summary', type="primary", use_container_width=True)

# Inside the second column (col2)
with col2:
    # Create a spinner for generating the summary
    with st.spinner('Generating Summary...'):
        if generate_button:
            if input_text:
                try:
                    # Tokenize and generate the summary
                    inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt", max_length=1024,
                                       add_special_tokens=True)
                    summary_ids = model.generate(inputs.input_ids, num_beams=num_beams, min_length=min_length,
                                                 max_length=max_length, early_stopping=True)
                    generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                except Exception as e:
                    # Handle errors
                    st.error(f"Error generating the summary: {str(e)}")
                    generated_summary = ""
            else:
                # Warn if no input is provided
                st.warning("Please enter some text to summarize.")
                generated_summary = ""
        else:
            generated_summary = ""

    # Display the generated summary in a text area
    st.text_area("Summary:", height=200, max_chars=512, value=generated_summary)

# Create a space for content
st.write("")

# Create instructions for the user
st.markdown("""        

                    **INSTRUCTIONS:**

                    1. Enter the text you want to summarize in the **Text** textbox.
                    2. Adjust the number of beams, minimum length, and maximum length of the summary using the sliders.
                    3. Click the **Generate Summary** button.
                    4. The generated summary will be displayed in the **Summary** textbox.
                    """
)
