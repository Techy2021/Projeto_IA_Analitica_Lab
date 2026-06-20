import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from database import conexao
from database.consultas import tabela_existe


class TabelaExisteTests(unittest.TestCase):
    def test_retorna_false_quando_dataset_lab_nao_existe(self):
        with tempfile.TemporaryDirectory() as pasta:
            banco = Path(pasta) / "deploy.duckdb"
            with patch.object(conexao, "DB_PATH", banco):
                self.assertFalse(tabela_existe("dataset_lab"))

    def test_retorna_true_quando_dataset_lab_existe(self):
        with tempfile.TemporaryDirectory() as pasta:
            banco = Path(pasta) / "local.duckdb"
            with patch.object(conexao, "DB_PATH", banco):
                con = conexao.get_connection()
                try:
                    con.execute("CREATE TABLE dataset_lab (valor INTEGER)")
                finally:
                    con.close()

                self.assertTrue(tabela_existe("dataset_lab"))


if __name__ == "__main__":
    unittest.main()
