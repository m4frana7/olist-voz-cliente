"""Testes locais com API simulada. Não usam chave real nem realizam geração paga."""
import io
import json
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from base_cloud import ErroServico, responder, chamar_groq


class TesteAPI(unittest.TestCase):
    def setUp(self):
        self.fontes = [{'document_id': 'review-1', 'text': 'Produto com defeito.',
                        'metadata': {'review_score': 1}}]

    def test_sem_chave_nao_envia(self):
        with patch('base_cloud.urllib.request.urlopen') as requisicao:
            with self.assertRaises(ErroServico):
                responder('Há defeitos?', self.fontes, '')
            requisicao.assert_not_called()

    def test_sem_fontes_nao_envia(self):
        with patch('base_cloud.chamar_groq') as api:
            resposta, uso = responder('Há defeitos?', [], 'teste')
            self.assertFalse(resposta['suficiente'])
            self.assertEqual(uso['chamadas'], 0)
            api.assert_not_called()

    def test_limite_nao_faz_retry_ou_fallback(self):
        erro = urllib.error.HTTPError('https://api.groq.com', 429, 'limit', {}, io.BytesIO(b''))
        with patch('base_cloud.urllib.request.urlopen', side_effect=erro) as op:
            with self.assertRaises(ErroServico) as c:
                chamar_groq('CHAVE_TESTE', 'openai/gpt-oss-20b', [], {})
            self.assertIn('cota', str(c.exception))
            self.assertNotIn('CHAVE_TESTE', str(c.exception))
            self.assertEqual(op.call_count, 1)

    def test_id_inventado_nao_publicado(self):
        falso = {'suficiente': True, 'motivo': '', 'achados': [
            {'document_id': 'inventado', 'afirmacao': 'Há defeito.', 'trecho': 'Produto com defeito.'}]}
        with patch('base_cloud.chamar_groq', return_value=(falso, {})) as api:
            r, uso = responder('Há defeitos?', self.fontes, 'teste')
            self.assertFalse(r['suficiente'])
            self.assertEqual(api.call_count, 1)

    def test_quebrado_nao_acrescentado(self):
        inicial = {'suficiente': True, 'motivo': '', 'achados': [
            {'document_id': 'review-1', 'afirmacao': 'Produto quebrado.', 'trecho': 'Produto com defeito.'}]}
        auditoria = {'verificacoes': [{'indice': 0, 'relevante': True, 'sustentado': True, 'justificativa': 'Citação literal.'}]}
        with patch('base_cloud.chamar_groq', side_effect=[(inicial, {}), (auditoria, {})]):
            r, uso = responder('Há defeitos?', self.fontes, 'teste')
            self.assertTrue(r['suficiente'])
            self.assertNotIn('quebrado', r['achados'][0]['afirmacao'])
            self.assertEqual(uso['chamadas'], 2)


class TesteInterface(unittest.TestCase):
    def test_abre_sem_secrets(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(Path(__file__).parent / 'streamlit_app.py')).run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(at.checkbox[0].disabled)

    def test_busca_sem_api(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        st.cache_resource.clear()
        fonte = {'document_id': 'teste', 'text': 'Entrega rápida.', 'similaridade': .8,
                 'metadata': {'review_score': 5, 'customer_state': 'SP', 'categorias': []}}
        with patch('base_cloud.Busca') as busca, patch('base_cloud.chamar_groq') as api:
            busca.return_value.buscar.return_value = [fonte]
            at = AppTest.from_file(str(Path(__file__).parent / 'streamlit_app.py')).run()
            at.text_area[0].set_value('Há elogios à entrega?')
            at.button[0].click().run()
            self.assertEqual(len(at.exception), 0)
            self.assertEqual(len(at.expander), 1)
            self.assertIn('Modo de busca', '\n'.join(x.value for x in at.info))
            api.assert_not_called()
        st.cache_resource.clear()


if __name__ == '__main__':
    unittest.main()
