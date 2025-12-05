"""
Base exception class for the project.
"""


class MistakeDetectionError(Exception):
    """
    Eccezione base per tutto il progetto MistakeDetection.
    
    Tutte le altre eccezioni custom devono ereditare da questa classe.
    Permette di catturare tutte le eccezioni del progetto con un singolo except.
    
    Example:
        try:
            # codice del progetto
            pass
        except MistakeDetectionError as e:
            # gestisce qualsiasi errore del progetto
            print(f"Errore nel progetto: {e}")
    """
    
    def __init__(self, message: str, details: dict = None):
        """
        Args:
            message: Messaggio descrittivo dell'errore
            details: Dizionario opzionale con informazioni aggiuntive
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        base_msg = self.message
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{base_msg} | Details: {details_str}"
        return base_msg
