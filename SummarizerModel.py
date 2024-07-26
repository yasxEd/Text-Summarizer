# Import necessary libraries
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
import pandas as pd
import chardet

# CustomDataset class to define a custom PyTorch dataset for handling text and summary pairs
class CustomDataset(Dataset):
    def __init__(self, texts, summaries):
        self.texts = texts  # Store input texts
        self.summaries = summaries  # Store target summaries
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")  # Initialize T5 tokenizer

    def __len__(self):
        return len(self.texts)  # Return the number of text-summary pairs in the dataset

    def __getitem__(self, idx):
        input_text = self.texts[idx]  # Get input text for a specific index
        target_summary = self.summaries[idx]  # Get target summary for the same index

        # Tokenize and encode input text and target summary
        inputs = self.tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)
        targets = self.tokenizer.encode(target_summary, return_tensors="pt", max_length=150, truncation=True)

        return {
            "input_ids": inputs[0],  # Return encoded input text
            "labels": targets[0],  # Return encoded target summary
        }
        
# Function to load a CSV dataset and return texts and summaries
def load_csv_dataset(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())  # Detect the character encoding of the file
    encoding = result['encoding']  # Get the detected encoding
        
    df = pd.read_csv(file_path, encoding=encoding)  # Read the CSV file using Pandas
    texts = df['TEXT'].tolist()  # Extract texts from the 'TEXT' column
    summaries = df['SUMMARY'].tolist()  # Extract summaries from the 'SUMMARY' column
    return texts, summaries  # Return the texts and summaries

# Collate function to pad input and label tensors in a batch
def collate_fn(batch):
    batch = [item for item in batch if item["input_ids"].shape[0] > 0 and item["labels"].shape[0] > 0]

    # Get the maximum length of input and label tensors in the batch
    max_input_len = max(item["input_ids"].shape[0] for item in batch)
    max_label_len = max(item["labels"].shape[0] for item in batch)

    # Pad input and label tensors in the batch to the maximum lengths
    padded_inputs = torch.stack([torch.nn.functional.pad(item["input_ids"], (0, max_input_len - item["input_ids"].shape[0])) for item in batch], dim=0)
    padded_labels = torch.stack([torch.nn.functional.pad(item["labels"], (0, max_label_len - item["labels"].shape[0])) for item in batch], dim=0)

    return {
        "input_ids": padded_inputs,  # Return padded input tensors
        "labels": padded_labels,  # Return padded label tensors
    }

# Function to train the summarization model
def train_model(train_dataset, test_dataset, val_dataset, epochs=40):
    model = T5ForConditionalGeneration.from_pretrained("t5-small")  # Load T5 model
    optimizer = AdamW(model.parameters(), lr=5e-5)  # Use AdamW optimizer for training

    # Create data loaders for training, testing, and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available
    model.to(device)  # Move the model to the GPU if available

    for epoch in range(epochs):  # Loop through the specified number of epochs
        model.train()  # Set the model to training mode
        train_losses = []  # List to store training losses
        for batch in train_loader:  # Loop through batches in the training dataset
            input_ids = batch["input_ids"].to(device)  # Move input tensors to the GPU if available
            labels = batch["labels"].to(device)  # Move label tensors to the GPU if available

            optimizer.zero_grad()  # Zero out the gradients
            outputs = model(input_ids, labels=labels)  # Forward pass
            loss = outputs.loss  # Get the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update the model parameters

            train_losses.append(loss.item())  # Append the current loss to the list

        average_train_loss = sum(train_losses) / len(train_losses)  # Compute the average training loss
        print(f"Epoch {epoch + 1}, Training Loss: {average_train_loss}")  # Print the average training loss

        model.eval()  # Set the model to evaluation mode
        test_losses = []  # List to store testing losses

        for test_batch in test_loader:  # Loop through batches in the testing dataset
            test_input_ids = test_batch["input_ids"].to(device)  # Move input tensors to the GPU if available
            test_labels = test_batch["labels"].to(device)  # Move label tensors to the GPU if available

            with torch.no_grad():  # Disable gradient calculation during testing
                test_outputs = model(test_input_ids, labels=test_labels)  # Forward pass
                test_loss = test_outputs.loss  # Get the loss

            test_losses.append(test_loss.item())  # Append the current loss to the list

        average_test_loss = sum(test_losses) / len(test_losses)  # Compute the average testing loss
        print(f"Epoch {epoch + 1}, Testing Loss: {average_test_loss}")  # Print the average testing loss

        val_losses = []  # List to store validation losses
        for val_batch in val_loader:  # Loop through batches in the validation dataset
            val_input_ids = val_batch["input_ids"].to(device)  # Move input tensors to the GPU if available
            val_labels = val_batch["labels"].to(device)  # Move label tensors to the GPU if available

            with torch.no_grad():  # Disable gradient calculation during validation
                val_outputs = model(val_input_ids, labels=val_labels)  # Forward pass
                val_loss = val_outputs.loss  # Get the loss

            val_losses.append(val_loss.item())  # Append the current loss to the list

        average_val_loss = sum(val_losses) / len(val_losses)  # Compute the average validation loss
        print(f"Epoch {epoch + 1}, Validation Loss: {average_val_loss}")  # Print the average validation loss

        # Calculate general and average loss
        general_loss = (average_train_loss + average_test_loss + average_val_loss) / 3
        print(f"Epoch {epoch + 1}, General Loss: {general_loss}")  # Print the general loss

        all_losses = train_losses + test_losses + val_losses
        average_all_loss = sum(all_losses) / len(all_losses)
        print(f"Epoch {epoch + 1}, Average Loss: {average_all_loss}")  # Print the average loss

    return model  # Return the trained model

# Main function to load the datasets, create custom datasets, train the model, and save the trained model
def main():
    train_file_path = "assets/TrainDataset.csv"  # File path for the training dataset
    test_file_path = "assets/TestDataset.csv"  # File path for the testing dataset
    val_file_path = "assets/ValDataset.csv"  # File path for the validation dataset

     # Load texts and summaries from CSV files
    train_texts, train_summaries = load_csv_dataset(train_file_path)
    test_texts, test_summaries = load_csv_dataset(test_file_path)
    val_texts, val_summaries = load_csv_dataset(val_file_path)

    # Create custom datasets using the loaded texts and summaries
    train_dataset = CustomDataset(train_texts, train_summaries)
    test_dataset = CustomDataset(test_texts, test_summaries)
    val_dataset = CustomDataset(val_texts, val_summaries)

    # Train the model using the custom datasets
    trained_model = train_model(train_dataset, test_dataset, val_dataset)

    # Save the trained model in the "NewSummarizer" directory
    trained_model.save_pretrained("NewSummarizer")

# Execute the main function if this script is run
if __name__ == "__main__":
    main()
