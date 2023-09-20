# Text Summarization with BART using Streamlit

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://textsummarization-hqykwe4yz5cftfdcqxte3q.streamlit.app/)

This is a Streamlit application for text summarization using the BART model. It allows you to generate a summary of a text with ease.

## Features

- **User-Friendly Interface**: An intuitive and user-friendly interface for input and options.
- **Advanced Options**: Customize the summarization process with advanced options like the number of beams and length constraints.
- **Real-Time Generation**: Generates summaries in real-time as you adjust the options.
- **Error Handling**: Provides informative error messages for better user experience.

## Usage
1. Enter the text you want to summarize in the Text textbox.
2. Adjust the number of beams, minimum length, and maximum length of the summary using the sliders.
3. Click the Generate Summary button.
4. The generated summary will be displayed in the Summary textbox.

## Getting Started
### Prerequisites
- Before running the application, ensure you have the required Python libraries installed:
  - pip install streamlit transformers

## Running the Application
- To run the Streamlit application, use the following command:
  - streamlit run your_app.py

## Built With
- **Streamlit** - The web framework used for creating interactive web applications in Python.
- **Hugging Face Transformers** - A library for Natural Language Processing (NLP) using state-of-the-art pre-trained models.

## Acknowledgments
- Special thanks to the Hugging Face team for their amazing transformers library
