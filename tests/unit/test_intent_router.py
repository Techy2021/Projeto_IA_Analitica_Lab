import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai.agentes.ai_responder import responder_pergunta_teste
from ai.intent_router import identificar_intencao, registrar_intencao
from ai.numeric_query_engine import executar_consulta_numerica


class IntentRouterTests(unittest.TestCase):
    def test_media_fibras_com_proteina_proxima(self):
        pergunta = "Qual a media de fibras quando a proteina e 46%?"
        rota = identificar_intencao(pergunta)
        self.assertEqual(rota["intencao"], "consulta_numerica")
        self.assertEqual(rota["entidades"]["operacao"], "media")
        self.assertEqual(
            rota["entidades"]["variavel_alvo"],
            "farelo_fibras_pct",
        )
        self.assertEqual(
            rota["entidades"]["filtro_variavel"],
            "farelo_proteina_pct",
        )
        self.assertEqual(rota["entidades"]["filtro_valor"], 46.0)

        resultado = executar_consulta_numerica(rota)
        self.assertEqual(resultado["fonte"], "duckdb")
        self.assertGreater(resultado["total_amostras"], 0)
        self.assertAlmostEqual(resultado["tolerancia"], 0.2)

    def test_maior_valor_umidade(self):
        rota = identificar_intencao("Qual o maior valor de umidade?")
        self.assertEqual(rota["intencao"], "consulta_numerica")
        self.assertEqual(rota["entidades"]["operacao"], "maximo")
        resultado = executar_consulta_numerica(rota)
        self.assertEqual(resultado["resultado"], 13.86)
        self.assertEqual(resultado["total_amostras"], 500)

    def test_contagem_urease_acima_de_limite(self):
        rota = identificar_intencao(
            "Quantas amostras tem urease acima de 0,20?"
        )
        self.assertEqual(rota["entidades"]["operacao"], "contagem")
        self.assertEqual(rota["entidades"]["filtro_operador"], "maior_que")
        resultado = executar_consulta_numerica(rota)
        self.assertEqual(resultado["resultado"], resultado["total_amostras"])
        self.assertGreater(resultado["total_amostras"], 0)

    def test_metricas_modelo(self):
        rota = identificar_intencao("Qual e o R2 do modelo?")
        self.assertEqual(rota["intencao"], "metricas_modelo")
        resposta = responder_pergunta_teste("Qual e o R2 do modelo?")
        self.assertIn("r2", resposta["resposta"].lower())

    def test_colunas_treinamento(self):
        rota = identificar_intencao(
            "Quais colunas foram usadas no treinamento?"
        )
        self.assertEqual(rota["intencao"], "colunas_treinamento")
        resposta = responder_pergunta_teste(
            "Quais colunas foram usadas no treinamento?"
        )
        self.assertIn("farelo_umidade_pct", resposta["resposta"])

    def test_limitacoes_modelo(self):
        rota = identificar_intencao("Posso confiar nesse modelo?")
        self.assertEqual(rota["intencao"], "limitacoes_modelo")
        resposta = responder_pergunta_teste("Posso confiar nesse modelo?")
        self.assertIn("apoio", resposta["resposta"].lower())
        self.assertIn("revisao humana", resposta["resposta"].lower())

    def test_fora_do_escopo(self):
        rota = identificar_intencao("Qual a previsao do tempo?")
        self.assertEqual(rota["intencao"], "fora_escopo")
        resposta = responder_pergunta_teste("Qual a previsao do tempo?")
        self.assertIn("dados laboratoriais", resposta["resposta"].lower())

    def test_interpretacao_laboratorial(self):
        rota = identificar_intencao("O que significa urease alta?")
        self.assertEqual(rota["intencao"], "interpretacao_laboratorial")

    def test_predicao_de_amostra(self):
        rota = identificar_intencao(
            "Preveja a qualidade de uma amostra com umidade 12,5%, "
            "proteina 46% e fibras 4,8%."
        )
        self.assertEqual(rota["intencao"], "predicao_amostra")
        self.assertIn(
            "farelo_umidade_pct",
            rota["entidades"]["valores_informados"],
        )

    def test_log_de_intencao(self):
        rota = identificar_intencao("Qual o maior valor de umidade?")
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "intent.jsonl"
            with patch("ai.intent_router.INTENT_LOG_PATH", caminho):
                registrar_intencao(
                    "Qual o maior valor de umidade?",
                    rota,
                    ferramenta="numeric_query_engine",
                    tempo_execucao_ms=12,
                    status="ok",
                )
            registro = json.loads(caminho.read_text(encoding="utf-8"))
            self.assertEqual(registro["intencao"], "consulta_numerica")
            self.assertEqual(registro["ferramenta"], "numeric_query_engine")


if __name__ == "__main__":
    unittest.main()
