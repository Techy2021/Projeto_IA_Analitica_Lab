import unittest
from unittest.mock import patch

from ai.rag import index, retriever


MANLAB013 = "MANLAB013 - MANUAL DO LABORATÓRIO GUARAPUAVA.docx"


class FakeCollection:
    def __init__(self):
        self.ids = ["manlab-1", "ativo-1"]
        self.metadatas = [
            {"source_file": MANLAB013, "source_path": "knowledge_base/MANLAB013.docx"},
            {"source_file": "ativo.txt", "source_path": "data/knowledge_base/ativo.txt"},
        ]

    def get(self, include=None):
        return {"ids": list(self.ids), "metadatas": list(self.metadatas)}

    def delete(self, ids=None, where=None):
        remover = set(ids or [])
        mantidos = [
            (identificador, metadata)
            for identificador, metadata in zip(self.ids, self.metadatas)
            if identificador not in remover
        ]
        self.ids = [item[0] for item in mantidos]
        self.metadatas = [item[1] for item in mantidos]

    def count(self):
        return len(self.ids)

    def query(self, **kwargs):
        documentos = ["conteudo ativo"] * len(self.ids)
        return {
            "documents": [documentos],
            "metadatas": [list(self.metadatas)],
            "distances": [[0.1] * len(self.ids)],
        }


class GerenciamentoDocumentosRagTests(unittest.TestCase):
    def test_excluir_documento_rag_remove_todos_os_chunks_da_fonte(self):
        collection = FakeCollection()
        with (
            patch.object(index, "obter_collection", return_value=collection),
            patch.object(index, "_localizar_arquivo_por_nome", return_value=None),
        ):
            resultado = index.excluir_documento_rag(MANLAB013)

        self.assertEqual(resultado["chunks_removidos"], 1)
        fontes_restantes = [
            metadata["source_file"] for metadata in collection.metadatas
        ]
        self.assertNotIn(MANLAB013, fontes_restantes)
        self.assertIn("ativo.txt", fontes_restantes)

    def test_listar_documentos_rag_nao_lista_fonte_excluida(self):
        collection = FakeCollection()
        collection.delete(ids=["manlab-1"])
        with patch.object(index, "obter_collection", return_value=collection):
            fontes = index.listar_documentos_rag()

        self.assertEqual([item["fonte"] for item in fontes], ["ativo.txt"])
        self.assertNotIn(MANLAB013, [item["fonte"] for item in fontes])

    def test_exclusao_tolera_nome_sem_acento(self):
        collection = FakeCollection()
        nome_sem_acento = "MANLAB013 - MANUAL DO LABORATORIO GUARAPUAVA.docx"
        with (
            patch.object(index, "obter_collection", return_value=collection),
            patch.object(index, "_localizar_arquivo_por_nome", return_value=None),
        ):
            resultado = index.excluir_documento_rag(nome_sem_acento)

        self.assertEqual(resultado["chunks_removidos"], 1)
        self.assertNotIn("manlab-1", collection.ids)

    def test_busca_rag_nao_retorna_manlab013_apos_exclusao(self):
        collection = FakeCollection()
        collection.delete(ids=["manlab-1"])
        with (
            patch.object(retriever, "obter_collection", return_value=collection),
            patch.object(retriever, "gerar_embedding", return_value=[0.1, 0.2]),
        ):
            resultados = retriever.buscar_trechos("consulta laboratorial", k=4)

        fontes = [item["fonte"] for item in resultados]
        self.assertNotIn(MANLAB013, fontes)
        self.assertEqual(fontes, ["ativo.txt"])


if __name__ == "__main__":
    unittest.main()
