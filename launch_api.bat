python -m venv api                   &:: Creates the Python Virtual Environment to run API in 
call api\Scripts\activate            &:: Switches into Virtual Environment to execute commands
pip install uv                       &:: Install uv package manager to speed up next command
uv pip install -r requirements.txt   &:: Installs dependencies with uv package manager
uvicorn api:app --reload --port=8888 &:: Launch the FastAPI on port 8888
