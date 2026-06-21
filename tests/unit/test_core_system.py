import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai.agentes.ai_responder import responder_pergunta_teste
from ai.modeling.predict import (
    carregar_metadata_modelo,
    carregar_modelo_salvo,
    modelo_treinado_existe,
)
from ai.rag import router


class ModeloPreditivoTests(unittest.TestCase):
    def test_flaml_automl_pode_ser_importado(self):
        from flaml import AutoML

        self.assertTrue(callable(AutoML))

    def test_modelo_e_metadados_estao_disponiveis(self):
        self.assertTrue(modelo_treinado_existe())
        modelo = carregar_modelo_salvo()
        self.assertTrue(callable(getattr(modelo, "predict", None)))

    def test_metadata_possui_contrato_minimo(self):
        metadata = carregar_metadata_modelo()
        self.assertIsInstance(metadata, dict)
        self.assertIn("alvo", metadata)
        self.assertIn("tipo_problema", metadata)
        self.assertIsInstance(metadata.get("colunas_usadas"), list)
        self.assertTrue(metadata["colunas_usadas"])


class RespostasIATests(unittest.TestCase):
    def test_metricas_sao_exibidas_sem_ollama(self):
        resultado = responder_pergunta_teste("Quais sao as metricas do modelo?")
        resposta = resultado["resposta"].lower()
        self.assertIn("metricas", resposta)
        self.assertTrue(any(nome in resposta for nome in ("mae", "rmse", "r2")))

    def test_chat_responde_pergunta_conceitual_local(self):
        resultado = responder_pergunta_teste(
            "O modelo pode liberar automaticamente um lote?"
        )
        self.assertIn("nao", resultado["resposta"].lower())
        self.assertIn("analista", resultado["resposta"].lower())
        self.assertIn("criterios oficiais", resultado["resposta"].lower())

    def test_pergunta_numerica_usa_duckdb(self):
        resultado = responder_pergunta_teste(
            "Quando o farelo tem proteina 46%, qual e a media de fibras?"
        )
        contexto = json.loads(resultado["contexto"])
        self.assertEqual(contexto["status"], "ok")
        self.assertEqual(contexto["fonte"], "duckdb")
        self.assertEqual(contexto["variavel_alvo"], "farelo_fibras_pct")
        self.assertGreater(contexto["total_amostras"], 0)

    def test_pergunta_numerica_fora_do_escopo_informa_limite(self):
        resultado = responder_pergunta_teste(
            "Qual e a media de cinzas quando a proteina do farelo e 46%?"
        )
        self.assertIn("nao suportada", resultado["resposta"].lower())


class RastreabilidadeTests(unittest.TestCase):
    def test_consulta_numerica_registra_log_jsonl(self):
        with tempfile.TemporaryDirectory() as pasta:
            log_path = Path(pasta) / "router.jsonl"
            with patch.object(router, "NUMERIC_LOG_PATH", log_path):
                router.responder_pergunta_numerica(
                    "Quando o farelo tem proteina 46%, qual e a media de fibras?"
                )

            linhas = log_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(linhas), 1)
            registro = json.loads(linhas[0])
            self.assertEqual(registro["tipo"], "numerica")
            self.assertIn("tempo_consulta_ms", registro)


if __name__ == "__main__":
    unittest.main()
