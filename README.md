This is the process of building a rag project. 
at first creating a .venv also known as python virtual environments.
Then we have to install uv : pip install uv
Then initialize uv : uv init 
Then sync it : uv sync
Then we have to install some packages : langchain, python-dotenv, streamlit, pypdf2, openai : 
command using : uv add <package name>

To run streamlit using uv : 
1. uv run streamlit run app.py