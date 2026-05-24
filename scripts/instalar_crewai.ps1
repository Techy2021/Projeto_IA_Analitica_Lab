D:
cd D:\Projeto_IA_Analitica_Lab
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install crewai crewai-tools litellm requests python-dotenv

python scripts\diagnostico_crewai.py
