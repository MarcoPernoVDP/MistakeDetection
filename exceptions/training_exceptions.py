"""
Eccezioni relative al training e alla validazione dei modelli.
"""

from .base import MistakeDetectionError


class TrainingError(MistakeDetectionError):
    """Eccezione base per errori durante il training."""
    pass


class CheckpointError(TrainingError):
    """Sollevata quando c'è un problema con i checkpoint."""
    
    def __init__(self, checkpoint_path: str, operation: str, reason: str = None):
        message = f"Errore durante {operation} del checkpoint"
        details = {"checkpoint_path": checkpoint_path}
        if reason:
            details["reason"] = reason
        super().__init__(message, details)


class ValidationError(TrainingError):
    """Sollevata quando c'è un errore durante la validazione."""
    
    def __init__(self, epoch: int = None, reason: str = None):
        message = "Errore durante la validazione"
        details = {}
        if epoch is not None:
            details["epoch"] = epoch
        if reason:
            details["reason"] = reason
        super().__init__(message, details)


class MetricCalculationError(TrainingError):
    """Sollevata quando il calcolo di una metrica fallisce."""
    
    def __init__(self, metric_name: str, reason: str = None):
        message = f"Errore nel calcolo della metrica '{metric_name}'"
        details = {}
        if reason:
            details["reason"] = reason
        super().__init__(message, details)


class EarlyStoppingError(TrainingError):
    """Sollevata quando l'early stopping viene attivato in modo anomalo."""
    
    def __init__(self, reason: str, patience: int = None):
        message = "Early stopping attivato in modo anomalo"
        details = {"reason": reason}
        if patience is not None:
            details["patience"] = patience
        super().__init__(message, details)


class OptimizerError(TrainingError):
    """Sollevata quando c'è un problema con l'optimizer."""
    
    def __init__(self, optimizer_name: str, reason: str):
        message = f"Errore con optimizer '{optimizer_name}'"
        details = {"reason": reason}
        super().__init__(message, details)
