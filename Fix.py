import nbformat

notebook_path = 'MinorModel.ipynb'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)

    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(notebook, f)
    print("Notebook repaired successfully.")

except Exception as e:
    print(f"Error repairing notebook: {e}")
