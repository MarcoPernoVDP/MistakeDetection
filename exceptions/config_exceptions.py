"""
Eccezioni relative alla configurazione del progetto.
"""

from .base import MistakeDetectionError


class ConfigError(MistakeDetectionError):
    """Eccezione base per errori di configurazione."""
    pass


class InvalidConfigValueError(ConfigError):
    """Sollevata quando un valore di configurazione non è valido."""
    
    def __init__(self, config_key: str, invalid_value, valid_values: list = None):
        message = f"Valore non valido per '{config_key}': {invalid_value}"
        details = {}
        if valid_values:
            details["valid_values"] = valid_values
        super().__init__(message, details)


class MissingConfigKeyError(ConfigError):
    """Sollevata quando manca una chiave di configurazione richiesta."""
    
    def __init__(self, missing_key: str, config_file: str = None):
        message = f"Chiave di configurazione mancante: '{missing_key}'"
        details = {}
        if config_file:
            details["config_file"] = config_file
        super().__init__(message, details)


class ConfigFileNotFoundError(ConfigError):
    """Sollevata quando un file di configurazione non viene trovato."""
    
    def __init__(self, config_file: str, searched_path: str = None):
        message = f"File di configurazione non trovato: '{config_file}'"
        details = {}
        if searched_path:
            details["searched_path"] = searched_path
        super().__init__(message, details)
