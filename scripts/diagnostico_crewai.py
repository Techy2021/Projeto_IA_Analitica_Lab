import importlib.util
import sys
import traceback


print("Python executável:", sys.executable)
print("Versão Python:", sys.version)

for pacote in ["crewai", "crewai_tools", "litellm", "requests", "dotenv"]:
    spec = importlib.util.find_spec(pacote)
    print(pacote, "OK" if spec else "NÃO ENCONTRADO")

try:
    import crewai

    print("CrewAI importado com sucesso")
    print("CrewAI:", crewai)
except Exception:
    print("Erro ao importar CrewAI")
    traceback.print_exc()
