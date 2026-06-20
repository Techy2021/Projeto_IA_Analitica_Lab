import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai.llm_provider import (
    MENSAGEM_CHAVE_OPENAI_AUSENTE,
    MENSAGEM_FALHA_SEGURA,
    MENSAGEM_OLLAMA_INDISPONIVEL,
    gerar_resposta_llm,
    obter_configuracao_llm,
    sanitizar_texto,
    testar_conexao_llm,
    validar_configuracao,
)


class LLMProviderTests(unittest.TestCase):
    def test_seleciona_ollama_por_variavel_de_ambiente(self):
        ambiente = {
            "LLM_PROVIDER": "ollama",
            "OLLAMA_MODEL": "gemma3:4b",
            "OLLAMA_BASE_URL": "http://ollama:11434",
        }
        with patch.dict(os.environ, ambiente, clear=True):
            config = obter_configuracao_llm()
        self.assertEqual(config.provedor, "ollama")
        self.assertEqual(config.modelo, "gemma3:4b")
        self.assertEqual(config.base_url, "http://ollama:11434")

    def test_seleciona_openai_por_variavel_de_ambiente(self):
        ambiente = {
            "LLM_PROVIDER": "openai",
            "OPENAI_MODEL": "modelo-online",
            "OPENAI_API_KEY": "chave-de-teste",
        }
        with patch.dict(os.environ, ambiente, clear=True):
            config = obter_configuracao_llm()
        self.assertEqual(config.provedor, "openai")
        self.assertEqual(config.modelo, "modelo-online")
        self.assertTrue(config.api_key)

    def test_openai_sem_chave_retorna_mensagem_segura(self):
        ambiente = {"LLM_PROVIDER": "openai", "OPENAI_MODEL": "modelo-online"}
        with patch.dict(os.environ, ambiente, clear=True):
            resposta = gerar_resposta_llm("teste")
        self.assertEqual(resposta, MENSAGEM_CHAVE_OPENAI_AUSENTE)

    def test_falha_do_provedor_retorna_fallback_seguro(self):
        ambiente = {
            "LLM_PROVIDER": "openai",
            "OPENAI_MODEL": "modelo-online",
            "OPENAI_API_KEY": "chave-de-teste",
        }
        with patch.dict(os.environ, ambiente, clear=True):
            with patch("litellm.completion", side_effect=RuntimeError("falhou")):
                resposta = gerar_resposta_llm("teste")
        self.assertEqual(resposta, MENSAGEM_FALHA_SEGURA)

    def test_falha_do_ollama_retorna_mensagem_especifica(self):
        ambiente = {
            "LLM_PROVIDER": "ollama",
            "OLLAMA_MODEL": "gemma3:1b",
            "OLLAMA_BASE_URL": "http://localhost:11434",
        }
        with patch.dict(os.environ, ambiente, clear=True):
            with patch("litellm.completion", side_effect=ConnectionError("offline")):
                resposta = gerar_resposta_llm("teste")
        self.assertEqual(resposta, MENSAGEM_OLLAMA_INDISPONIVEL)

    def test_provedor_invalido_nao_faz_fallback_silencioso(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "desconhecido"}, clear=True):
            config = obter_configuracao_llm()
            erro = validar_configuracao(config)
            resposta = gerar_resposta_llm("teste")
        self.assertEqual(config.provedor, "desconhecido")
        self.assertIn("Provedor de IA inválido", erro)
        self.assertEqual(resposta, erro)

    def test_conexao_nao_expoe_chave(self):
        chave = "sk-chave-conexao-super-secreta-123456"
        overrides = {
            "LLM_PROVIDER": "openai",
            "OPENAI_MODEL": "modelo-online",
            "OPENAI_API_KEY": chave,
        }
        resposta_mock = {
            "choices": [{"message": {"content": "OK"}}]
        }
        with patch("litellm.completion", return_value=resposta_mock):
            resultado = testar_conexao_llm(overrides)
        self.assertEqual(resultado["status"], "ok")
        self.assertTrue(resultado["chave_configurada"])
        self.assertNotIn("api_key", resultado)
        self.assertNotIn(chave, json.dumps(resultado))

    def test_chave_nao_aparece_em_texto_sanitizado_ou_log(self):
        chave = "sk-chave-super-secreta-123456"
        config = obter_configuracao_llm(
            {
                "LLM_PROVIDER": "openai",
                "OPENAI_MODEL": "modelo-online",
                "OPENAI_API_KEY": chave,
            }
        )
        texto = sanitizar_texto(f"api_key={chave}; Authorization Bearer {chave}", config)
        self.assertNotIn(chave, texto)

        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "teste.log"
            logger = logging.getLogger("teste_llm_provider")
            logger.handlers.clear()
            handler = logging.FileHandler(caminho, encoding="utf-8")
            logger.addHandler(handler)
            logger.error("%s", texto)
            handler.close()
            self.assertNotIn(chave, caminho.read_text(encoding="utf-8"))

    def test_configuracao_publica_nunca_contem_api_key(self):
        config = obter_configuracao_llm(
            {
                "LLM_PROVIDER": "openai",
                "OPENAI_MODEL": "modelo-online",
                "OPENAI_API_KEY": "segredo",
            }
        )
        serializado = json.dumps(config.publico())
        self.assertNotIn("segredo", serializado)
        self.assertNotIn("api_key", config.publico())

    def test_trace_de_agente_remove_chave(self):
        from ai.agentes import crewai_agents_lab

        chave = "sk-chave-log-super-secreta-123456"
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "trace.jsonl"
            with patch.object(crewai_agents_lab, "TRACE_PATH", caminho):
                crewai_agents_lab._registrar_trace(
                    {"status": "erro", "erro": f"api_key={chave}"}
                )
            conteudo = caminho.read_text(encoding="utf-8")
        self.assertNotIn(chave, conteudo)
        self.assertIn("[REDACTED]", conteudo)

    def test_chat_usa_litellm_direto_sem_crewai(self):
        from ai.agentes import crewai_agents_lab

        roteamento = {
            "intencao": "interpretacao_laboratorial",
            "confianca": 0.9,
            "entidades": {},
        }
        with (
            patch.object(crewai_agents_lab, "CREWAI_DISPONIVEL", False),
            patch.object(crewai_agents_lab, "CREWAI_TOOLS_DISPONIVEL", False),
            patch.object(
                crewai_agents_lab,
                "identificar_intencao",
                return_value=roteamento,
            ),
            patch.object(
                crewai_agents_lab,
                "_gerar_contexto_ferramentas",
                return_value=("contexto seguro", ["metadata_modelo"]),
            ),
            patch.object(
                crewai_agents_lab,
                "gerar_resposta_llm",
                return_value="Resposta direta de produção.",
            ),
            patch.object(crewai_agents_lab, "_registrar_trace"),
            patch.object(crewai_agents_lab, "registrar_intencao"),
        ):
            resultado = crewai_agents_lab.executar_crew_lab(
                "Explique este resultado laboratorial.",
                llm_config={
                    "LLM_PROVIDER": "openai",
                    "OPENAI_MODEL": "modelo-online",
                    "OPENAI_API_KEY": "segredo-de-teste",
                },
            )

        self.assertEqual(resultado["status"], "ok")
        self.assertEqual(resultado["modo_ferramentas"], "llm_direto_deploy")
        self.assertEqual(resultado["resposta"], "Resposta direta de produção.")


if __name__ == "__main__":
    unittest.main()
