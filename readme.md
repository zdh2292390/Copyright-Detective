# How to Run Copyright Detective

Follow these steps to set up your environment and run the application.

/home/changhu/miniconda3/envs/copyright-detective/bin/python /home/changhu/Copyright-Detective/


## 1. Create a Conda Environment

It is highly recommended to use a Conda environment to manage project dependencies.

Open your terminal and run the following command in the project's root directory to create a new environment named `copyright-detective`:

```bash
conda create --name copyright-detective python=3.9 -y
```

This will create a new Conda environment with Python 3.9.

## 2. Activate the Conda Environment

Before you can install packages or run the app, you need to activate the environment.

```bash
conda activate copyright-detective
```

Your terminal prompt should now show `(copyright-detective)` at the beginning, indicating that the environment is active.

## 3. Install Required Packages

With the virtual environment active, install all the necessary libraries from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

This will install `streamlit`, `openai`, `PyPDF2`, and `rouge-score`.

## 4. Run the Streamlit Application

Now you are ready to start the application. Run the following command:

```bash
streamlit run app.py
```

Streamlit will start a local server and open the application in a new tab in your default web browser.

You can now interact with the "Copyright Detective" tool. Make sure you have a valid OpenAI API key to use the model-based features.

## PDF Analysis Notes

- Chunk size is specified in words. During Whole PDF Analysis, the Generated Text produced for each chunk is enforced to be exactly the same number of words as the selected chunk size. This ensures fair, length-controlled comparisons across chunks.
