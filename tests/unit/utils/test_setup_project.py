import unittest
import os
import sys
from io import StringIO
from unittest.mock import patch, MagicMock
from exceptions.config_exceptions import MissingConfigKeyError
from utils.setup_project import is_colab, get_secret, initialize


class TestSetupProject(unittest.TestCase):
    
    def test_is_colab_false(self):
        """Test che is_colab ritorna False in ambiente locale"""
        self.assertFalse(is_colab())
    
    def test_is_colab_mocked(self):
        """Test che la funzione is_colab può essere mockata"""
        with patch('sys.modules', {'google.colab': MagicMock()}):
            # La funzione is_colab controlla sys.modules internamente
            # Questo test verifica solo che la funzione non crashi con il mock
            result = is_colab()
            self.assertIsInstance(result, bool)
    
    def test_get_secret_from_env(self):
        """Test recupero secret da variabile d'ambiente"""
        os.environ['TEST_KEY'] = 'test_value'
        result = get_secret('TEST_KEY')
        self.assertEqual(result, 'test_value')
        del os.environ['TEST_KEY']
    
    def test_get_secret_missing(self):
        """Test secret non trovato"""
        with self.assertRaises(MissingConfigKeyError):
            get_secret('NONEXISTENT_KEY')
    
    @patch('torch.cuda.is_available', return_value=False)
    @patch('utils.setup_project.is_colab', return_value=False)
    def test_initialize_basic(self, mock_is_colab, mock_cuda):
        """Test inizializzazione base del progetto"""
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        try:
            # Cattura output per evitare errori di encoding
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                device = initialize(temp_dir)
            finally:
                sys.stdout = old_stdout
            self.assertIsNotNone(device)
            self.assertEqual(str(device), 'cpu')
        finally:
            shutil.rmtree(temp_dir)
