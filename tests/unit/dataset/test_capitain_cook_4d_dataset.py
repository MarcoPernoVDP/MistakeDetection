import unittest
import os
import sys
from io import StringIO
import tempfile
import shutil
import json
import numpy as np
import torch
from dataset.capitain_cook_4d_mlp_dataset import CaptainCook4DMLP_Dataset, DatasetSource


class TestCaptainCook4DDataset(unittest.TestCase):
    
    def setUp(self):
        """Crea un mini dataset di test"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Crea struttura directory
        self.data_dir = os.path.join(self.temp_dir, "data")
        self.omnivore_dir = os.path.join(self.data_dir, "omnivore")
        self.annotations_dir = os.path.join(self.data_dir, "annotation_json")
        
        os.makedirs(self.omnivore_dir)
        os.makedirs(self.annotations_dir)
        
        # Crea file npz di test
        features = np.random.rand(10, 1024).astype(np.float32)
        np.savez(os.path.join(self.omnivore_dir, "1_10_360p.mp4_1s_1s.npz"), features=features)
        
        # Crea annotazioni di test
        annotations = {
            "1_10": {
                "steps": [
                    {"start_time": 2, "end_time": 5, "has_errors": 1},
                    {"start_time": 7, "end_time": 8, "has_errors": 0}
                ]
            }
        }
        with open(os.path.join(self.annotations_dir, "complete_step_annotations.json"), "w") as f:
            json.dump(annotations, f)
    
    def tearDown(self):
        """Rimuove directory temporanee"""
        shutil.rmtree(self.temp_dir)
    
    def test_dataset_loading(self):
        """Test caricamento base del dataset"""
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            dataset = CaptainCook4DMLP_Dataset(DatasetSource.OMNIVORE, self.temp_dir)
        finally:
            sys.stdout = old_stdout
        self.assertIsNotNone(dataset)
        self.assertGreater(len(dataset), 0)
    
    def test_dataset_shape(self):
        """Test shape del dataset"""
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            dataset = CaptainCook4DMLP_Dataset(DatasetSource.OMNIVORE, self.temp_dir)
        finally:
            sys.stdout = old_stdout
        shape = dataset.shape()
        self.assertEqual(len(shape), 2)
        self.assertEqual(shape[1], 1024)
    
    def test_dataset_getitem(self):
        """Test accesso agli item del dataset"""
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            dataset = CaptainCook4DMLP_Dataset(DatasetSource.OMNIVORE, self.temp_dir)
        finally:
            sys.stdout = old_stdout
        x, y = dataset[0]
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(x.shape[0], 1024)
    
    def test_labels_generation(self):
        """Test generazione corretta delle label"""
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            dataset = CaptainCook4DMLP_Dataset(DatasetSource.OMNIVORE, self.temp_dir)
        finally:
            sys.stdout = old_stdout
        unique_labels = torch.unique(dataset.y)
        self.assertTrue(len(unique_labels) > 0)
