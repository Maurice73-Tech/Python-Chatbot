# Digital Business Engineering Information Python-Chatbot

## Introduction

Welcome to the GitHub repository of the Digital Business Engineering (DBE) Information Chatbot. This Python-based chatbot leverages the power of Retrieval-Augmented Generation (RAG) and GPT-4 to provide information about the Master's program in Digital Business Engineering. It extracts and processes data from PDFs and interacts with users, answering their queries about the study program.

## Features

- **PDF Text Extraction**: Extracts text from PDF documents related to the DBE program.
- **Tokenization and Text Splitting**: Utilizes GPT-2 tokenizer to process and split text data.
- **Embedding Generation**: Creates embeddings for text chunks for efficient information retrieval.
- **Question-Answering System**: Leverages a QA retrieval system powered by GPT-4 to answer student queries.
- **Interactive Chat Interface**: Built with Streamlit, offering a user-friendly interface for querying and displaying responses.

## Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your PDF files (related to the DBE program) in the project directory.
2. Run the Streamlit app:
   ```bash
   streamlit run [app.py]
   ```
3. Interact with the chatbot via the Streamlit interface by typing your questions.

## Environment Configuration

- Ensure you have an OpenAI API key and set it as an environment variable in a `.env` file.

## Contributions

Contributions are welcome! Please read the `CONTRIBUTING.md` file for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.

---

*Disclaimer: This chatbot is designed for informational purposes related to the DBE study program. For official information, always refer to the official university resources.*

---

Feel free to star and fork this repository if you find this project interesting or useful. Happy coding! 🚀🤖