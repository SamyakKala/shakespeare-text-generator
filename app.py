import streamlit as st
import tensorflow as tf
import numpy as np

# Load the model
model = tf.keras.models.load_model('C:/Users/ok/Desktop/CLG/5TH SEM/FOML/projectfinal/fomlfinal.h5', compile=False)

# Load the character mappings
vocab = sorted(set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.:;!?"))
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = np.array(vocab)

# Define text generation function
def generate_text(model, start_string, num_generate=1000):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = predictions[:, -1, :]
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)

# Streamlit interface
st.title("Text Generation with Tiny Shakespeare Model")
st.write("Enter a starting string to generate Shakespeare-like text.")

# Input text box
start_string = st.text_input("Starting text:", "QUEEN: So, let's end this")

# Button to generate text
if st.button("Generate Text"):
    with st.spinner("Generating text..."):
        generated_text = generate_text(model, start_string)
        st.success("Generated Text:")
        st.write(generated_text)
