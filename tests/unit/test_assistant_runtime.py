import time
import unittest

from ai.assistant_runtime import (
    MENSAGEM_ERRO_ASSISTENTE,
    MENSAGEM_RESPOSTA_VAZIA_ASSISTENTE,
    MENSAGEM_TIMEOUT_ASSISTENTE,
    executar_assistente_com_timeout,
    preparar_resposta_assistente,
)


class AssistantRuntimeTests(unittest.TestCase):
    def test_resposta_bem_sucedida_e_renderizavel(self):
        resposta, sucesso = preparar_resposta_assistente(
            {"status": "ok", "resposta": "funcionando"}
        )

        self.assertTrue(sucesso)
        self.assertEqual(resposta, "funcionando")

    def test_resposta_vazia_retorna_mensagem_amigavel(self):
        resposta, sucesso = preparar_resposta_assistente(
            {"status": "ok", "resposta": "   "}
        )

        self.assertFalse(sucesso)
        self.assertEqual(resposta, MENSAGEM_RESPOSTA_VAZIA_ASSISTENTE)

    def test_status_de_erro_nao_expoe_detalhe_tecnico(self):
        resposta, sucesso = preparar_resposta_assistente(
            {"status": "erro", "resposta": "401 chave inválida"}
        )

        self.assertFalse(sucesso)
        self.assertEqual(resposta, MENSAGEM_ERRO_ASSISTENTE)
        self.assertNotIn("401", resposta)

    def test_execucao_tem_timeout_global(self):
        def executor_lento(pergunta, llm_config):
            time.sleep(0.2)
            return {"status": "ok", "resposta": pergunta}

        with self.assertRaisesRegex(TimeoutError, "demorou mais"):
            executar_assistente_com_timeout(
                "funcionando",
                {"LLM_PROVIDER": "gemini"},
                timeout_segundos=0.05,
                executor_func=executor_lento,
            )

    def test_timeout_configurado_e_repassado_ao_provedor(self):
        configuracao_recebida = {}

        def executor_imediato(pergunta, llm_config):
            configuracao_recebida.update(llm_config)
            return {"status": "ok", "resposta": pergunta}

        resultado = executar_assistente_com_timeout(
            "funcionando",
            {"LLM_PROVIDER": "gemini"},
            timeout_segundos=60,
            executor_func=executor_imediato,
        )

        self.assertEqual(resultado["resposta"], "funcionando")
        self.assertEqual(configuracao_recebida["LLM_TIMEOUT"], "60")
        self.assertEqual(configuracao_recebida["ASSISTANT_DIRECT_MODE"], "true")
        self.assertIn("60 segundos", MENSAGEM_TIMEOUT_ASSISTENTE)

    def test_erro_nativo_e_convertido_em_falha_controlada(self):
        class ErroNativo(BaseException):
            pass

        def executor_com_panic(pergunta, llm_config):
            raise ErroNativo("falha nativa")

        with self.assertRaisesRegex(RuntimeError, "Não foi possível concluir"):
            executar_assistente_com_timeout(
                "funcionando",
                {"LLM_PROVIDER": "gemini"},
                timeout_segundos=60,
                executor_func=executor_com_panic,
            )


if __name__ == "__main__":
    unittest.main()
