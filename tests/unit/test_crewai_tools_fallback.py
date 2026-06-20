import json
import unittest
from unittest.mock import patch

import requests

from ai.agentes.crewai_tools_lab import (
    consultar_dados_laboratorio_func,
    obter_amostra_media_func,
    obter_metadata_modelo_func,
    prever_farelo_soja_func,
)
from ai.modeling.predict import adaptar_aliases_entrada


class CrewAIToolsFallbackTests(unittest.TestCase):
    @patch.dict("os.environ", {}, clear=True)
    @patch("ai.agentes.crewai_tools_lab.load_dotenv")
    @patch("ai.agentes.crewai_tools_lab.requests.get")
    def test_sem_api_configurada_nao_faz_requisicao(
        self,
        mock_get,
        _mock_load_dotenv,
    ):
        resultado = json.loads(obter_metadata_modelo_func())

        mock_get.assert_not_called()
        self.assertEqual(resultado["fonte"], "metadata_local")

    @patch("ai.agentes.crewai_tools_lab.requests.get")
    def test_metadata_usa_arquivo_local_quando_api_esta_offline(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError()

        resultado = json.loads(obter_metadata_modelo_func())

        self.assertEqual(resultado["fonte"], "metadata_local")
        self.assertTrue(resultado["colunas_usadas"])
        self.assertIn("metricas", resultado)

    @patch("ai.agentes.crewai_tools_lab.requests.get")
    def test_amostra_usa_mediana_do_modelo_quando_api_esta_offline(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError()

        resultado = json.loads(obter_amostra_media_func())

        self.assertEqual(resultado["fonte"], "medianas_modelo_salvo")
        self.assertEqual(set(resultado["dados"]), set(resultado["colunas_usadas"]))

    @patch("ai.agentes.crewai_tools_lab.requests.post")
    @patch("ai.agentes.crewai_tools_lab.gerar_previsao_detalhada")
    def test_previsao_usa_modelo_local_quando_api_esta_offline(
        self,
        mock_previsao,
        mock_post,
    ):
        mock_post.side_effect = requests.exceptions.ConnectionError()
        mock_previsao.return_value = {
            "previsao": 46.1,
            "alvo": "farelo_proteina_pct",
            "tarefa": "regression",
            "colunas_preenchidas": [],
            "colunas_extras_ignoradas": [],
        }

        resultado = json.loads(prever_farelo_soja_func(umidade_pct=12.5))

        self.assertEqual(resultado["fonte"], "modelo_local")
        self.assertEqual(resultado["previsao"], 46.1)

    def test_aliases_da_interface_respeitam_colunas_do_modelo(self):
        colunas_modelo = [
            "farelo_umidade_pct",
            "farelo_oleo_pct",
            "farelo_fibras_pct",
            "farelo_materia_mineral_pct",
            "farelo_urease_delta_ph",
            "farelo_solubilidade_koh_pct",
        ]
        dados = {
            "umidade_pct": 12.5,
            "proteina_pct": 46.2,
            "extrato_etereo_pct": 1.8,
            "fibras_pct": 4.7,
            "materia_mineral_pct": 4.5,
            "urease_uph": 0.12,
            "solubilidade_pct": 82.0,
        }

        adaptados = adaptar_aliases_entrada(dados, colunas_modelo)

        self.assertEqual(adaptados["farelo_umidade_pct"], 12.5)
        self.assertEqual(adaptados["farelo_oleo_pct"], 1.8)
        self.assertEqual(adaptados["farelo_fibras_pct"], 4.7)
        self.assertEqual(adaptados["farelo_materia_mineral_pct"], 4.5)
        self.assertEqual(adaptados["farelo_urease_delta_ph"], 0.12)
        self.assertEqual(adaptados["farelo_solubilidade_koh_pct"], 82.0)
        self.assertNotIn("farelo_proteina_pct", adaptados)
        self.assertNotIn("umidade_pct", adaptados)
        self.assertIn("proteina_pct", adaptados)

    @patch("ai.agentes.crewai_tools_lab.requests.post")
    @patch("ai.agentes.crewai_tools_lab.consultar_sql")
    def test_consulta_sem_api_ou_duckdb_retorna_limitacao_amigavel(
        self,
        mock_consultar,
        mock_post,
    ):
        mock_post.side_effect = requests.exceptions.ConnectionError()
        mock_consultar.side_effect = RuntimeError("dataset_lab não disponível")

        resultado = json.loads(
            consultar_dados_laboratorio_func("SELECT * FROM dataset_lab")
        )

        self.assertIn("Recurso indisponível neste ambiente", resultado["erro"])
        self.assertNotIn("Inicie com", resultado["erro"])


if __name__ == "__main__":
    unittest.main()
