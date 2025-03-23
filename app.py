import streamlit as st
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
import torch
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels = ["Non-Hate", "Offensive", "Hate"]

# Factory class for creating model and tokenizer (Factory Design Pattern)
class ModelFactory:
    @staticmethod
    def create_model_and_tokenizer(model_path, tokenizer_path):
        # Verify files exist
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found. Please ensure it exists.")
            raise FileNotFoundError(f"Model file '{model_path}' not found.")
        
        required_tokenizer_files = ["tokenizer_config.json", "vocab.txt"]
        missing_files = [f for f in required_tokenizer_files if not os.path.exists(os.path.join(tokenizer_path, f))]
        if missing_files:
            st.error(f"Missing tokenizer files: {', '.join(missing_files)}. Please upload them.")
            raise FileNotFoundError(f"Missing tokenizer files: {missing_files}")

        # Initialize model with config
        config = BertConfig.from_pretrained("bert-base-uncased", num_hidden_layers=6, num_labels=3)
        model = BertForSequenceClassification(config).to(device)

        # Load state dictionary from .pth file
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)  # Ignore missing/extra keys
        model.eval()

        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

        return model, tokenizer

# Classifier class to handle text classification
class TextClassifier:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def classify(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            pred = logits.argmax(-1).item()
        return labels[pred]

# Streamlit app
def main():
    st.title("Laiba's Hate Speech Detector")  # Attribution to Laiba
    st.write("Enter text below to classify it as Non-Hate, Offensive, or Hate.")
    st.write("Using the Odd-Layer Distilled Model (Test Accuracy: 0.9660)")

    # Paths (absolute for local testing)
    model_path = r"C:\Users\HP\Downloads\nlp_a7-main\nlp_a7-main\student_odd.pth"
    tokenizer_path = r"C:\Users\HP\Downloads\nlp_a7-main\nlp_a7-main"

    # Load model and tokenizer using Factory pattern
    try:
        model, tokenizer = st.cache_resource(ModelFactory.create_model_and_tokenizer)(model_path, tokenizer_path)
        classifier = TextClassifier(model, tokenizer)
    except Exception as e:
        st.error(f"Failed to load model or tokenizer: {str(e)}")
        return

    # User input and classification
    user_input = st.text_area("Your Text", placeholder="Type here...")
    if st.button("Classify"):
        if user_input:
            result = classifier.classify(user_input)
            st.success(f"Prediction: **{result}**")
            st.write(f"Input: \"{user_input}\"")
        else:
            st.warning("Please enter some text to classify.")

    # Try to load performance plot
    try:
        st.subheader("Model Performance")
        plot_path = r"C:\Users\HP\Downloads\nlp_a7-main\nlp_a7-main\download (9).png"
        st.image(plot_path, caption="Training Loss and Eval Accuracy (Laiba's Work)")  # Attribution to Laiba
    except FileNotFoundError:
        st.write("Performance plot not available. Expected file: 'download (9).png'")

if __name__ == "__main__":
    main()
