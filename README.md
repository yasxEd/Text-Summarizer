# Text Summarizer

A Python application for summarizing text using natural language processing techniques. The app provides a web interface using Streamlit and includes functionality for training and using a T5 transformer model for text summarization.
![Screenshot](https://github.com/user-attachments/assets/2927c82e-341d-4ee6-95f8-f65c83290984)

## Features

- Summarizes long text into shorter, coherent summaries
- Supports manual text entry or text file uploads
- Train and fine-tune the T5 model on custom datasets
- Easy-to-use web interface

## Usage

### Running the Web Application

 `Open your web browser and go to http://localhost:8501`

### Training the Model

1. **Prepare your training, testing, and validation datasets:**
   Ensure your datasets are in CSV format with columns `TEXT` and `SUMMARY`.

2. **Specify the paths to your datasets in the `main()` function in `train.py`.**

3. **Run the training script:**
`python train.py`

4. **The trained model will be saved in the `NewSummarizer` directory.**

## Example

To summarize a text file named `example.txt` and save the summary to `summary.txt`, run:
`python text_summarizer.py example.txt summary.txt`
