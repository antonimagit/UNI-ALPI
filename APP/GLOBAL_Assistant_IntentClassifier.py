import json
import re
import os
from datetime import datetime

# Nome del file di configurazione - UNICO PUNTO DOVE È DEFINITO
CONFIG_FILE = "GLOBAL_Assistant_Classifier_Config.json"

class IntentClassifier:
    def __init__(self, log_func=None):        
        self.log = log_func if log_func else self._default_log
        self.config = self.load_config()  # Carica config UNA SOLA VOLTA
        self.last_matched_pattern = None 
        
    def _default_log(self, message):        
        """Default logging se non viene passata una funzione di log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [INTENT] {message}")
    
    def load_config(self):        
        """Carica la configurazione dal file JSON - DEVE ESISTERE"""
        try:
            if not os.path.exists(CONFIG_FILE):
                error_msg = f"CRITICAL ERROR: Config file {CONFIG_FILE} not found!"
                self.log(error_msg)
                raise FileNotFoundError(error_msg)
            
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.log(f"Intent classifier config loaded successfully from {CONFIG_FILE}")
                return config
                
        except json.JSONDecodeError as e:
            error_msg = f"CRITICAL ERROR: Invalid JSON in {CONFIG_FILE}: {e}"
            self.log(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"CRITICAL ERROR loading config: {e}"
            self.log(error_msg)
            raise

    def classify_query_intent(self, user_query, conversation_history=None):        
        """
        Classifica l'intent della query dell'utente
        Ritorna: 'FOLLOW_UP' o 'NEW_QUERY'
        """
        query_lower = user_query.lower().strip()
        
        # Se non c'è storia conversazionale e è richiesta per follow-up
        if (self.config["settings"]["require_conversation_history_for_followup"] 
            and not conversation_history):
            if self.config["settings"]["enable_logging"]:
                self.log("NEW_QUERY: no conversation history required for follow-up")
            return 'NEW_QUERY'
        
        # Controlla pattern di FOLLOW_UP
        for category, patterns in self.config["followup_patterns"].items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    if self.config["settings"]["enable_logging"]:
                        self.log(f"FOLLOW_UP detected with pattern: {pattern} (category: {category})")
                        self.last_matched_pattern = {"category": category, "pattern": pattern}
                    return 'FOLLOW_UP'
        
        # Default: NEW_QUERY
        if self.config["settings"]["enable_logging"]:
            self.log("NEW_QUERY: no follow-up patterns matched")
        return 'NEW_QUERY'
    
    def get_reference_turn(self, conversation_history):
        """Ottiene l'ultimo turn della conversazione per il caching"""
        if not conversation_history:
            return None
        return conversation_history[-1]
    
    def add_pattern(self, category, pattern):
        """Aggiunge un nuovo pattern a runtime (solo in memoria, non su file)"""
        if category not in self.config["followup_patterns"]:
            self.config["followup_patterns"][category] = []
        
        self.config["followup_patterns"][category].append(pattern)
        self.log(f"Added pattern '{pattern}' to category '{category}' (memory only)")
    
    def get_stats(self):
        """Ritorna statistiche sui pattern configurati"""
        total_patterns = sum(len(patterns) for patterns in self.config["followup_patterns"].values())
        categories = list(self.config["followup_patterns"].keys())
        
        return {
            "total_followup_patterns": total_patterns,
            "categories": categories,
            "settings": self.config["settings"],
            "new_query_patterns": len(self.config.get("new_query_indicators", []))
        }

# Funzione di utilità per integrazione facile
def create_classifier(log_func=None):
    """Factory function per creare un classificatore"""   
    return IntentClassifier(log_func)