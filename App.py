# Import the Streamlit library, which is used for creating web applications.
import streamlit as st

# Import T5ForConditionalGeneration and T5Tokenizer from the transformers library.
# T5ForConditionalGeneration is a transformer model for conditional text generation.
# T5Tokenizer is used for tokenization (breaking text into tokens).
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Define a function named summarize_text that takes input text, a model, and a tokenizer as parameters.
def summarize_text(input_text, model, tokenizer):
    # Check if input_text is of type bytes (binary data, possibly from a file).
    if isinstance(input_text, bytes):
        # Decode the bytes to a string using utf-8 encoding.
        input_text = input_text.decode('utf-8')

    # Tokenize and encode the input_text using the T5 model's tokenizer.
    inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate a summary using the T5 model.
    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    # Decode the generated summary and remove special tokens.
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Return the generated summary.
    return summary

# Define the main function for the Streamlit app.
def main():
    
    # Set Streamlit page configuration with a specific title, icon, and layout.
    st.set_page_config(page_title="Text Summarizer App", page_icon="✍️", layout="wide")

    # Display a banner image for the app.
    banner_image = "assets/test.png"
    st.image(banner_image, use_column_width=True)

    # Display the main title and subheader for the app.
    st.title("📚 Text Summarizer App")
    st.subheader("Generate concise summaries for your text!")

    # Specify the path to the fine-tuned summarizer model and load it.
    model_path = "NewSummarizer"
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    # Load the T5 tokenizer from the "t5-small" pre-trained model.
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Allow the user to choose the input option (manual text entry or file upload).
    input_option = st.radio("Choose input option:", ("Enter text manually", "Upload a text file"))

    # Based on the user's choice, obtain the input text.
    if input_option == "Enter text manually":
        # If the user chooses manual entry, use a text area to input text.
        input_text = st.text_area("Enter text to summarize:")
    else:
        # If the user chooses file upload, allow them to upload a text file.
        uploaded_file = st.file_uploader("Choose a text file", type=["txt"])

        if uploaded_file is not None:
            # If a file is uploaded, read its content.
            input_text = uploaded_file.read()
        else:
            # If no file is uploaded, set input_text to an empty string.
            input_text = ""

    # Check if the "Generate Summary" button is clicked.
    if st.button("Generate Summary", key="summarize_button"):
        # Check if there is input text for summarization.
        if input_text:
            # If there is text, display a subheader and use a spinner while generating the summary.
            st.subheader("Generated Summary:")
            with st.spinner("Summarizing..."):
                # Display the generated summary using the summarize_text function.
                st.success(summarize_text(input_text, model, tokenizer))
        else:
            # If no input text, display a warning to enter text.
            st.warning("Please enter text to summarize.")

    # Add a horizontal line for separation.
    st.markdown("<hr>", unsafe_allow_html=True)
    # Display attribution for the app's creators.
    st.markdown("Built by BOUSSEBA Yasser and JA FATIMA Ezzahra")

# Check if the script is executed as the main program and call the main function.
if __name__ == "__main__":
    main()
