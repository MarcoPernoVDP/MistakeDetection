import unittest
import os
import tempfile
import shutil
import zipfile
import sys
from io import StringIO
from utils.setup_dataset import setup_dataset


class TestSetupDataset(unittest.TestCase):
    
    def setUp(self):
        """Crea directory temporanee per i test"""
        self.temp_dir = tempfile.mkdtemp()
        self.source_dir = os.path.join(self.temp_dir, "source")
        self.project_dir = os.path.join(self.temp_dir, "project")
        os.makedirs(self.source_dir)
        os.makedirs(self.project_dir)
    
    def tearDown(self):
        """Rimuove directory temporanee"""
        shutil.rmtree(self.temp_dir)
    
    def test_setup_with_folder(self):
        """Test copia di una cartella"""
        test_folder = os.path.join(self.source_dir, "test_folder")
        os.makedirs(test_folder)
        with open(os.path.join(test_folder, "file.txt"), "w") as f:
            f.write("test")
        
        # Cattura output per evitare errori di encoding
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            setup_dataset(self.source_dir, self.project_dir)
        finally:
            sys.stdout = old_stdout
        
        dest_folder = os.path.join(self.project_dir, "data", "test_folder")
        self.assertTrue(os.path.exists(dest_folder))
        self.assertTrue(os.path.exists(os.path.join(dest_folder, "file.txt")))
    
    def test_setup_with_zip(self):
        """Test estrazione di un file zip"""
        zip_path = os.path.join(self.source_dir, "test.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("test_file.txt", "content")
        
        # Cattura output per evitare errori di encoding
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            setup_dataset(self.source_dir, self.project_dir)
        finally:
            sys.stdout = old_stdout
        
        self.assertTrue(os.path.exists(os.path.join(self.project_dir, "data", "test_file.txt")))
    
    def test_setup_empty_source(self):
        """Test con cartella sorgente vuota"""
        # Cattura output per evitare errori di encoding
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            setup_dataset(self.source_dir, self.project_dir)
        finally:
            sys.stdout = old_stdout
        # Non dovrebbe crashare
        self.assertTrue(os.path.exists(os.path.join(self.project_dir, "data")))
    
    def test_setup_nonexistent_source(self):
        """Test con cartella sorgente inesistente"""
        fake_source = os.path.join(self.temp_dir, "nonexistent")
        # Cattura output per evitare errori di encoding
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            setup_dataset(fake_source, self.project_dir)
        finally:
            sys.stdout = old_stdout
        # Non dovrebbe crashare
        self.assertTrue(os.path.exists(os.path.join(self.project_dir, "data")))
