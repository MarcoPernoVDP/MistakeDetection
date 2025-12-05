"""
Eccezioni relative al caricamento e gestione dei dataset.
"""

from .base import MistakeDetectionError


class DatasetError(MistakeDetectionError):
    """Eccezione base per errori nel dataset."""
    pass


class AnnotationNotFoundError(DatasetError):
    """Sollevata quando un file di annotazioni non viene trovato."""
    
    def __init__(self, annotation_file: str, searched_path: str):
        message = f"File di annotazioni '{annotation_file}' non trovato"
        details = {"searched_path": searched_path}
        super().__init__(message, details)


class FeatureFileNotFoundError(DatasetError):
    """Sollevata quando un file di features (.npz) non viene trovato."""
    
    def __init__(self, feature_file: str, searched_path: str = None):
        message = f"File di features '{feature_file}' non trovato"
        details = {"searched_path": searched_path} if searched_path else {}
        super().__init__(message, details)


class InvalidDatasetSourceError(DatasetError):
    """Sollevata quando viene specificata una sorgente dataset non valida."""
    
    def __init__(self, source: str, valid_sources: list):
        message = f"Sorgente dataset '{source}' non valida"
        details = {"valid_sources": valid_sources}
        super().__init__(message, details)


class MismatchedDataShapeError(DatasetError):
    """Sollevata quando le dimensioni dei dati non corrispondono."""
    
    def __init__(self, expected_shape: tuple, actual_shape: tuple, context: str = ""):
        message = f"Dimensioni dati non corrispondenti{': ' + context if context else ''}"
        details = {
            "expected_shape": expected_shape,
            "actual_shape": actual_shape
        }
        super().__init__(message, details)


class EmptyDatasetError(DatasetError):
    """Sollevata quando il dataset risulta vuoto dopo il caricamento."""
    
    def __init__(self, reason: str = ""):
        message = "Dataset vuoto dopo il caricamento"
        details = {"reason": reason} if reason else {}
        super().__init__(message, details)


class CorruptedFeatureFileError(DatasetError):
    """Sollevata quando un file .npz è corrotto o non leggibile."""
    
    def __init__(self, file_path: str, original_error: str):
        message = f"File di features corrotto: {file_path}"
        details = {"original_error": str(original_error)}
        super().__init__(message, details)


class MissingAnnotationKeyError(DatasetError):
    """Sollevata quando manca una chiave nelle annotazioni JSON."""
    
    def __init__(self, missing_key: str, file_name: str):
        message = f"Chiave '{missing_key}' mancante nelle annotazioni"
        details = {"file_name": file_name}
        super().__init__(message, details)
