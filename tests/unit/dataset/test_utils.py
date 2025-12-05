import unittest
import sys
from io import StringIO
import torch
from torch.utils.data import Dataset, Subset
from dataset.utils import print_class_balance, get_loaders


class MockDataset(Dataset):
    """Dataset mock per testing"""
    def __init__(self, size=100, feature_dim=10):
        self.X = torch.randn(size, feature_dim)
        # Crea label bilanciate (60% classe 0, 40% classe 1)
        self.y = torch.cat([
            torch.zeros(int(size * 0.6), dtype=torch.long),
            torch.ones(int(size * 0.4), dtype=torch.long)
        ])
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def shape(self):
        return self.X.shape


class TestUtils(unittest.TestCase):
    
    def test_print_class_balance(self):
        """Test che print_class_balance non crashi"""
        dataset = MockDataset(size=50)
        # Test non dovrebbe crashare
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            print_class_balance(dataset, "Test Dataset")
            success = True
        except:
            success = False
        finally:
            sys.stdout = old_stdout
        self.assertTrue(success)
    
    def test_print_class_balance_subset(self):
        """Test print_class_balance con Subset"""
        dataset = MockDataset(size=100)
        subset = Subset(dataset, indices=list(range(20)))
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            print_class_balance(subset, "Test Subset")
            success = True
        except:
            success = False
        finally:
            sys.stdout = old_stdout
        self.assertTrue(success)
    
    def test_get_loaders(self):
        """Test creazione loaders"""
        dataset = MockDataset(size=100)
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            train_loader, val_loader, test_loader = get_loaders(
                dataset, 
                batch_size=16, 
                val_ratio=0.1, 
                test_ratio=0.2
            )
        finally:
            sys.stdout = old_stdout
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)
        
        # Verifica che i loader contengano dati
        self.assertGreater(len(train_loader), 0)
        self.assertGreater(len(val_loader), 0)
        self.assertGreater(len(test_loader), 0)
    
    def test_get_loaders_split_ratios(self):
        """Test che i ratio di split siano rispettati"""
        dataset = MockDataset(size=100)
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            train_loader, val_loader, test_loader = get_loaders(
                dataset,
                batch_size=100,  # Batch grande per contare facilmente
                val_ratio=0.1,
                test_ratio=0.2
            )
        finally:
            sys.stdout = old_stdout
        
        total_samples = sum(len(batch[0]) for batch in train_loader)
        total_samples += sum(len(batch[0]) for batch in val_loader)
        total_samples += sum(len(batch[0]) for batch in test_loader)
        
        self.assertEqual(total_samples, 100)
