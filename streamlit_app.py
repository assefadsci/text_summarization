# Import necessary libraries
import streamlit as st
from transformers import AutoTokenizer, BartForConditionalGeneration

# Set Streamlit page configuration, including title and icon
st.set_page_config(
    page_title='Text Summarizer',
    page_icon='🎁',
    layout='wide'
)

# Create a Markdown section for the application header
st.markdown("""
    <div style="text-align:center;">
        <h1 style="color: #F24949;font-style: sans-serif;font-size: 3em;margin-bottom: 15px;">
            <span style="font-size: 1em; margin-right: 0.2em;">📝</span>
            Text Summarizer
        </h1>
        <P style="color: black;font-style: italic; font-size=1.2em; line-height=1.2em; font-family: Arial;">
            Generate a summary of a text using the BART model with this Streamlit application. Try it now! 🚀
        </P>
    </div>
""", unsafe_allow_html=True)

# Create a space for content
st.write("")

if 'input_text' not in st.session_state:
    st.session_state.input_text = " "

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

    st.subheader(" :red[Enter Your Text Below ] :writing_hand:")

    # Create a text area for user input
    input_text = st.text_area(label= "", height=200, max_chars=4000, value=st.session_state.input_text)

    if input_text:
        st.session_state.input_text = input_text

    # Create an expander for advanced options
    with st.expander(":red[**Advanced Options**]:"):
        num_beams = st.slider("Number of Beams:", min_value=1, max_value=10, value=2)
        min_length = st.slider("Minimum Length of Summary:", min_value=1, max_value=512, value=116)
        max_length = st.slider("Maximum Length of Summary:", min_value=30, max_value=1024, value=710)

    # Create a space for content
    st.write("")

    buttons_col1, buttons_col2 = st.columns(2)
    with buttons_col1:
        generate_button = st.button('Summarize', type="primary", use_container_width=True)
    with buttons_col2:
        clear_button = st.button('Clear', use_container_width=True)

if clear_button:
    # st.session_state.input_text= ""
    output_text = ""

# Inside the second column (col2)
with col2:
    # Create a spinner for generating the summary
    if generate_button:

        # with st.spinner("⚙️ :green[Generating Summary..]"):
        if input_text:

            try:

                # Tokenize and generate the summary
                inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt",
                                   max_length=1024,
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

    st.subheader(":green[Generated Summary] 🎉🎆")
    st.text_area("", height=200, max_chars=512, value=generated_summary)

    if generated_summary:
        st.balloons()

    st.download_button(
        ":red[Download Summary]", data=generated_summary,  file_name="summary.txt", mime="text/plain",
        use_container_width=True)

# Create a space for content
st.write("")

# Create instructions for the user
st.markdown("""        

                    :red[**INSTRUCTIONS**]:

                    1. Enter the text you want to summarize in the :red[**Text**] textbox.
                    2. Adjust the number of beams, minimum length, and maximum length of the summary using the sliders.
                    3. Click the :red[**Summarize**] button.
                    4. The generated summary will be displayed in the :red[**Summary**] textbox.
                    """
)
