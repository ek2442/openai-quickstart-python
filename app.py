import os
import openai
import requests
import matplotlib.pyplot as plt
import tiktoken
import json
import pandas as pd

from flask import Flask, redirect, render_template, request, url_for
from openai.embeddings_utils import get_embedding
app = Flask(__name__)

openai.organization =("org-7EObwYIJXrwN4x0ZSsXRzCqH")
openai.api_key = os.getenv("OPENAI_API_KEY")

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

input_datapath = "uploads/"



def process_input(input_data):
    # Get a chatbot response from OpenAI
    response = openai.Completion.create(
        engine="gpt3.5-turbo",
        prompt=input_data,
        max_tokens=5000,
        n=1,
        stop=None,
        temperature=0.5,
    )
    output_data = response.choices[0].text.strip()

    # Process the output data here
    return output_data.upper()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the input data from the form
        input_data = request.form['input_data']

        # Process the input data
        output_data = process_input(input_data)

        # Pass the input and output data to the template
        return render_template('index.html', input_data=input_data, output_data=output_data)
    else:
        # Render the template without any data
        return render_template('index.html')
    
       # Load and inspect the dataset
    df = pd.read_csv(input_datapath, index_col=0)
    df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
    df = df.dropna()
    df["combined"] = (
        "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
    )
    df_head = df.head(2)

    # Render the template with the dataset and input form
    return render_template('index.html', df_head=df_head)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    file.save(os.path.join(input_datapath, file.filename))
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)