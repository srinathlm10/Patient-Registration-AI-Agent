import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from git import Repo
import requests
import json

# -------- CONFIG --------
DEFAULT_REPO_URL = "https://github.com/parimienosh/Patient-Registration.git"
CLONE_DIR = "./cloned_repo"

OLLAMA_API_URL = "http://localhost:11434/api/chat"
DB_NAME = "chroma_repo_db"

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=DB_NAME)
collection = chroma_client.get_or_create_collection(name="repo_files")


# -------- Clone GitHub Repo --------
def clone_repo(repo_url):
    if not os.path.exists(CLONE_DIR):
        Repo.clone_from(repo_url, CLONE_DIR)
        st.success("GitHub repo cloned successfully.")
    else:
        st.info("Repo already exists locally. Skipping clone.")


# -------- Read All Files from Repo --------
def read_all_files(repo_path):
    file_data = {}
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith((".java", ".xml", ".txt", ".md", ".json", ".yml")):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        file_data[path] = f.read()
                except Exception as e:
                    st.warning(f"Failed to read {path}: {e}")
    return file_data


# -------- Store Embeddings in ChromaDB --------
def create_chroma_index(files_dict, model):
    for path, content in files_dict.items():
        embedding = model.encode(content).tolist()
        collection.add(ids=[path], embeddings=[embedding], metadatas=[{"path": path}])


# -------- Search Using ChromaDB --------
def search_chroma(query, model):
    query_embedding = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=1)
    if results and results["ids"] and results["ids"][0]:
        return results["ids"][0][0]
    return None


# -------- Ask LLM via Ollama --------
def ask_ollama_llm(file_content, user_question):
    prompt = f"""Here is the file content:

{file_content}

Now answer this:
{user_question}"""
    payload = {"model": "llama3.2", "messages": [{"role": "user", "content": prompt}]}
    response = requests.post(OLLAMA_API_URL, json=payload, stream=True)
    answer = ""
    if response.status_code == 200:
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    json_data = json.loads(line)
                    if "message" in json_data and "content" in json_data["message"]:
                        answer += json_data["message"]["content"]
                except json.JSONDecodeError:
                    pass
    return answer


# -------- AI Agent (Conversational) --------
def ai_agent_interaction(files_dict):
    st.title("AI Code Assistant 🤖")

    # Introduction and Guidance
    st.write("Welcome! I am your AI Code Assistant. You can ask me about your project repository.")

    menu = st.radio("Choose a task", ["Search Files", "Review Code", "Generate Test Cases", "Generate Code Summary"])

    if menu == "Search Files":
        query = st.text_input("Search for something in the repository (e.g., 'PatientController')")
        if query and st.button("🔍 Search"):
            model = SentenceTransformer('all-MiniLM-L6-v2')
            best_match = search_chroma(query, model)
            if best_match:
                st.session_state["selected_file"] = best_match
                st.session_state["selected_content"] = files_dict[best_match]
                st.write(f"Found the best match in: `{best_match}`")
                st.write(f"File content:\n{files_dict[best_match][:1000]}")

    elif menu == "Review Code":
        selected_file = st.selectbox("Select a file to review:", list(files_dict.keys()))
        if selected_file:
            st.write(f"Reviewing code in `{selected_file}`...")
            code_review = ask_ollama_llm(files_dict[selected_file], "Analyze and provide a detailed review.")
            st.write("Code Review Suggestions:")
            st.write(code_review)

    elif menu == "Generate Test Cases":
        selected_file = st.selectbox("Select a file to generate test cases:", list(files_dict.keys()))
        if selected_file:
            st.write(f"Generating test cases for `{selected_file}`...")
            test_case_gen = ask_ollama_llm(files_dict[selected_file], "Generate JUnit test cases for this code.")
            st.write("Suggested JUnit Tests:")
            st.write(test_case_gen)

    elif menu == "Generate Code Summary":
        selected_file = st.selectbox("Select a file to generate code summary:", list(files_dict.keys()))
        if selected_file:
            st.write(f"Generating code summary for `{selected_file}`...")
            code_summary = ask_ollama_llm(files_dict[selected_file], "Provide a high-level summary of this code.")
            st.write("Code Summary:")
            st.write(code_summary)


# -------- Streamlit UI --------
st.set_page_config(page_title="AI Code Assistant", layout="wide")

repo_url_input = st.text_input("Enter GitHub Repo URL", DEFAULT_REPO_URL)
if st.button("🚀 Clone Repo"):
    clone_repo(repo_url_input)

# Load the repo and files
if os.path.exists(CLONE_DIR):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    files_dict = read_all_files(CLONE_DIR)
    if not collection.count():
        create_chroma_index(files_dict, model)

    ai_agent_interaction(files_dict)

else:
    st.warning("⚠️ Please clone the repository first!")
