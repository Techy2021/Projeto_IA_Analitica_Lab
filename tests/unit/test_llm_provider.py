import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai.llm_provider import (
    MENSAGEM_CHAVE_GEMINI_AUSENTE,
    MENSAGEM_CHAVE_OPENAI_AUSENTE,
    MENSAGEM_FALHA_SEGURA,
    MENSAGEM_OLLAMA_INDISPONIVEL,
    MENSAGEM_RESPOSTA_VAZIA,
    MENSAGEM_TIMEOUT,
    gerar_resposta_llm,
    obter_configuracao_llm,
    sanitizar_texto,
    testar_conexao_llm as executar_teste_conexao_llm,
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

    def test_alias_qwen3_vl_usa_variante_instruct_para_chat(self):
        ambiente = {
            "LLM_PROVIDER": "ollama",
            "OLLAMA_MODEL": "qwen3-vl:4b",
            "OLLAMA_BASE_URL": "http://localhost:11434",
        }
        with patch.dict(os.environ, ambiente, clear=True):
            config = obter_configuracao_llm()

        self.assertEqual(config.modelo, "qwen3-vl:4b-instruct")

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

    def test_seleciona_gemini_por_variavel_de_ambiente(self):
        ambiente = {
            "LLM_PROVIDER": "gemini",
            "GEMINI_MODEL": "gemini-2.5-flash",
            "GEMINI_API_KEY": "chave-gemini-de-teste",
        }
        with patch.dict(os.environ, ambiente, clear=True):
            config = obter_configuracao_llm()

        self.assertEqual(config.provedor, "gemini")
        self.assertEqual(config.modelo, "gemini-2.5-flash")
        self.assertTrue(config.api_key)

    def test_configuracao_da_sessao_tem_prioridade_sobre_ambiente(self):
        ambiente = {
            "LLM_PROVIDER": "openai",
            "OPENAI_MODEL": "modelo-do-ambiente",
            "OPENAI_API_KEY": "chave-do-ambiente",
        }
        sessao = {
            "LLM_PROVIDER": "openai",
            "OPENAI_MODEL": "modelo-da-sessao",
            "OPENAI_API_KEY": "chave-da-sessao",
        }
        with patch.dict(os.environ, ambiente, clear=True):
            config = obter_configuracao_llm(sessao)

        self.assertEqual(config.modelo, "modelo-da-sessao")
        self.assertEqual(config.api_key, "chave-da-sessao")

    def test_campo_de_sessao_vazio_usa_ambiente_como_fallback(self):
        ambiente = {
            "LLM_PROVIDER": "gemini",
            "GEMINI_MODEL": "gemini-2.5-flash",
            "GEMINI_API_KEY": "chave-do-ambiente",
        }
        sessao = {
            "LLM_PROVIDER": "gemini",
            "GEMINI_MODEL": "gemini-2.5-flash",
            "GEMINI_API_KEY": "",
        }
        with patch.dict(os.environ, ambiente, clear=True):
            config = obter_configuracao_llm(sessao)

        self.assertEqual(config.api_key, "chave-do-ambiente")

    def test_openai_sem_chave_retorna_mensagem_segura(self):
        ambiente = {"LLM_PROVIDER": "openai", "OPENAI_MODEL": "modelo-online"}
        with patch.dict(os.environ, ambiente, clear=True):
            resposta = gerar_resposta_llm("teste")
        self.assertEqual(resposta, MENSAGEM_CHAVE_OPENAI_AUSENTE)

    def test_gemini_sem_chave_retorna_mensagem_amigavel(self):
        ambiente = {
            "LLM_PROVIDER": "gemini",
            "GEMINI_MODEL": "gemini-2.5-flash",
        }
        with patch.dict(os.environ, ambiente, clear=True):
            resposta = gerar_resposta_llm("teste")

        self.assertEqual(resposta, MENSAGEM_CHAVE_GEMINI_AUSENTE)
        self.assertEqual(resposta, "Chave GEMINI_API_KEY não configurada.")

    def test_gemini_usa_prefixo_e_chave_no_litellm(self):
        ambiente = {
            "LLM_PROVIDER": "gemini",
            "GEMINI_MODEL": "gemini-2.5-flash",
            "GEMINI_API_KEY": "chave-gemini-de-teste",
        }
        resposta_mock = {"choices": [{"message": {"content": "Gemini ativo."}}]}
        with patch.dict(os.environ, ambiente, clear=True):
            with patch("litellm.completion", return_value=resposta_mock) as completion:
                resposta = gerar_resposta_llm("teste")

        self.assertEqual(resposta, "Gemini ativo.")
        self.assertEqual(
            completion.call_args.kwargs["model"],
            "gemini/gemini-2.5-flash",
        )
        self.assertEqual(
            completion.call_args.kwargs["api_key"],
            "chave-gemini-de-teste",
        )

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

    def test_timeout_do_provedor_retorna_mensagem_especifica(self):
        ambiente = {
            "LLM_PROVIDER": "gemini",
            "GEMINI_MODEL": "gemini-2.5-flash",
            "GEMINI_API_KEY": "chave-gemini-de-teste",
        }
        with patch.dict(os.environ, ambiente, clear=True):
            with patch(
                "litellm.completion",
                side_effect=TimeoutError("request timed out"),
            ):
                resposta = gerar_resposta_llm("teste")

        self.assertEqual(resposta, MENSAGEM_TIMEOUT)

    def test_resposta_vazia_do_provedor_retorna_orientacao_completa(self):
        ambiente = {
            "LLM_PROVIDER": "gemini",
            "GEMINI_MODEL": "gemini-2.5-flash",
            "GEMINI_API_KEY": "chave-gemini-de-teste",
        }
        resposta_mock = {"choices": [{"message": {"content": ""}}]}
        with patch.dict(os.environ, ambiente, clear=True):
            with patch("litellm.completion", return_value=resposta_mock):
                resposta = gerar_resposta_llm("teste")

        self.assertEqual(resposta, MENSAGEM_RESPOSTA_VAZIA)
        self.assertIn("modelo, chave", resposta)

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

    def test_ollama_desativa_thinking_para_retornar_conteudo(self):
        ambiente = {
            "LLM_PROVIDER": "ollama",
            "OLLAMA_MODEL": "qwen3-vl:4b-thinking",
            "OLLAMA_BASE_URL": "http://localhost:11434",
        }
        resposta_mock = {"choices": [{"message": {"content": "Funcionou."}}]}
        with patch.dict(os.environ, ambiente, clear=True):
            with patch("litellm.completion", return_value=resposta_mock) as completion:
                resposta = gerar_resposta_llm("teste")

        self.assertEqual(resposta, "Funcionou.")
        self.assertFalse(completion.call_args.kwargs["think"])
        self.assertEqual(completion.call_args.kwargs["num_retries"], 0)
        self.assertEqual(completion.call_args.kwargs["max_tokens"], 1200)
        self.assertEqual(completion.call_args.kwargs["timeout"], 300)

    def test_ollama_thinking_vazio_recomenda_variante_instruct(self):
        ambiente = {
            "LLM_PROVIDER": "ollama",
            "OLLAMA_MODEL": "qwen3-vl:4b-thinking",
            "OLLAMA_BASE_URL": "http://localhost:11434",
        }
        resposta_mock = {"choices": [{"message": {"content": ""}}]}
        with patch.dict(os.environ, ambiente, clear=True):
            with patch("litellm.completion", return_value=resposta_mock):
                resposta = gerar_resposta_llm("teste")

        self.assertIn("qwen3-vl:4b-instruct", resposta)

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
            resultado = executar_teste_conexao_llm(overrides)
        self.assertEqual(resultado["status"], "ok")
        self.assertTrue(resultado["chave_configurada"])
        self.assertNotIn("api_key", resultado)
        self.assertNotIn(chave, json.dumps(resultado))

    def test_conexao_ollama_valida_servico_e_modelo_sem_gerar_texto(self):
        overrides = {
            "LLM_PROVIDER": "ollama",
            "OLLAMA_MODEL": "qwen3-vl:4b",
            "OLLAMA_BASE_URL": "http://localhost:11434",
        }
        resposta_http = unittest.mock.Mock()
        resposta_http.json.return_value = {
            "models": [{"name": "qwen3-vl:4b-instruct"}]
        }

        with (
            patch("ai.llm_provider.requests.get", return_value=resposta_http),
            patch("litellm.completion") as completion,
        ):
            resultado = executar_teste_conexao_llm(overrides)

        self.assertEqual(resultado["status"], "ok")
        resposta_http.raise_for_status.assert_called_once()
        completion.assert_not_called()

    def test_conexao_ollama_informa_modelo_ausente(self):
        overrides = {
            "LLM_PROVIDER": "ollama",
            "OLLAMA_MODEL": "qwen3-vl:4b",
            "OLLAMA_BASE_URL": "http://localhost:11434",
        }
        resposta_http = unittest.mock.Mock()
        resposta_http.json.return_value = {"models": [{"name": "gemma3:1b"}]}

        with patch("ai.llm_provider.requests.get", return_value=resposta_http):
            resultado = executar_teste_conexao_llm(overrides)

        self.assertEqual(resultado["status"], "erro")
        self.assertIn("ollama pull qwen3-vl:4b-instruct", resultado["mensagem"])

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

    def test_chave_gemini_nao_aparece_em_texto_sanitizado(self):
        chave = "chave-gemini-super-secreta"
        config = obter_configuracao_llm(
            {
                "LLM_PROVIDER": "gemini",
                "GEMINI_MODEL": "gemini-2.5-flash",
                "GEMINI_API_KEY": chave,
            }
        )

        texto = sanitizar_texto(f"api_key={chave}", config)

        self.assertNotIn(chave, texto)

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
                return_value=("contexto seguro", ["metadata_modelo"], None),
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

    def test_chat_continua_sem_rag_e_retorna_aviso(self):
        from ai.agentes import crewai_agents_lab
        from ai.rag.index import MENSAGEM_RAG_INDISPONIVEL

        roteamento = {
            "intencao": "interpretacao_laboratorial",
            "confianca": 0.9,
            "entidades": {},
        }
        with (
            patch.object(
                crewai_agents_lab,
                "identificar_intencao",
                return_value=roteamento,
            ),
            patch.object(
                crewai_agents_lab,
                "verificar_disponibilidade_rag",
                return_value={
                    "disponivel": False,
                    "motivo": "chromadb_ausente",
                    "mensagem": MENSAGEM_RAG_INDISPONIVEL,
                },
            ),
            patch.object(
                crewai_agents_lab,
                "gerar_resposta_llm",
                return_value="funcionando",
            ),
            patch.object(crewai_agents_lab, "_registrar_trace"),
            patch.object(crewai_agents_lab, "registrar_intencao"),
        ):
            resultado = crewai_agents_lab.executar_crew_lab(
                "Responda apenas: funcionando.",
                llm_config={
                    "LLM_PROVIDER": "gemini",
                    "GEMINI_MODEL": "gemini-2.5-flash",
                    "GEMINI_API_KEY": "segredo-de-teste",
                    "ASSISTANT_DIRECT_MODE": "true",
                },
            )

        self.assertEqual(resultado["status"], "ok")
        self.assertEqual(resultado["resposta"], "funcionando")
        self.assertEqual(resultado["aviso_rag"], MENSAGEM_RAG_INDISPONIVEL)


if __name__ == "__main__":
    unittest.main()
