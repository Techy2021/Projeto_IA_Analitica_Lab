import os
import sys
import traceback
from pathlib import Path

import requests
from dotenv import load_dotenv


PROJETO_DIR = Path(__file__).resolve().parents[1]
if str(PROJETO_DIR) not in sys.path:
    sys.path.insert(0, str(PROJETO_DIR))


def main() -> int:
    load_dotenv(PROJETO_DIR / ".env")

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    modelo = os.getenv("OLLAMA_MODEL", "gemma3:4b")

    print("Python executável:", sys.executable)
    print("Ollama:", base_url)
    print("Modelo:", modelo)

    try:
        resposta = requests.get(f"{base_url}/api/tags", timeout=10)
        resposta.raise_for_status()
        payload = resposta.json()
        modelos = [
            item.get("name") or item.get("model")
            for item in payload.get("models", [])
        ]
        modelos = [str(item) for item in modelos if item]
        print("Modelos Ollama encontrados:", modelos)

        if modelo not in modelos:
            raise RuntimeError(f"Modelo Ollama não encontrado: {modelo}")

        from crewai import Agent, Crew, LLM, Process, Task

        llm = LLM(model=f"ollama/{modelo}", base_url=base_url)
        agente = Agent(
            role="Especialista em nutrição animal",
            goal="Responder perguntas simples sobre farelo de soja em português.",
            backstory=(
                "Você interpreta conceitos laboratoriais e nutricionais de forma "
                "clara e objetiva."
            ),
            llm=llm,
            verbose=False,
        )
        tarefa = Task(
            description="Responda em português: o que é proteína no farelo de soja?",
            expected_output="Uma explicação curta em português.",
            agent=agente,
        )
        crew = Crew(
            agents=[agente],
            tasks=[tarefa],
            process=Process.sequential,
            verbose=False,
        )

        resultado = crew.kickoff()
        print("Resposta CrewAI:")
        print(resultado)
        return 0
    except Exception:
        print("Erro ao testar CrewAI com Ollama")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
