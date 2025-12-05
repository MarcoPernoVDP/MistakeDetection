"""
Eccezioni relative ai modelli di Machine Learning.
"""

from .base import MistakeDetectionError


class ModelError(MistakeDetectionError):
    """Eccezione base per errori nei modelli."""
    pass


class InvalidModelConfigError(ModelError):
    """Sollevata quando la configurazione del modello non è valida."""
    
    def __init__(self, config_key: str, reason: str):
        message = f"Configurazione modello non valida per '{config_key}'"
        details = {"reason": reason}
        super().__init__(message, details)


class ModelLoadError(ModelError):
    """Sollevata quando il caricamento di un modello fallisce."""
    
    def __init__(self, checkpoint_path: str, original_error: str = None):
        message = f"Impossibile caricare il modello da '{checkpoint_path}'"
        details = {}
        if original_error:
            details["original_error"] = str(original_error)
        super().__init__(message, details)


class ModelSaveError(ModelError):
    """Sollevata quando il salvataggio di un modello fallisce."""
    
    def __init__(self, save_path: str, original_error: str = None):
        message = f"Impossibile salvare il modello in '{save_path}'"
        details = {}
        if original_error:
            details["original_error"] = str(original_error)
        super().__init__(message, details)


class InvalidInputShapeError(ModelError):
    """Sollevata quando l'input ha una forma non compatibile con il modello."""
    
    def __init__(self, expected_shape: tuple, actual_shape: tuple):
        message = "Forma dell'input non compatibile con il modello"
        details = {
            "expected_shape": expected_shape,
            "actual_shape": actual_shape
        }
        super().__init__(message, details)


class ModelNotTrainedError(ModelError):
    """Sollevata quando si tenta di usare un modello non ancora addestrato."""
    
    def __init__(self, model_name: str = None):
        message = "Modello non ancora addestrato"
        details = {"model_name": model_name} if model_name else {}
        super().__init__(message, details)
