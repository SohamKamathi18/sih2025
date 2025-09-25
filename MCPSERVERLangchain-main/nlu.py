import json
import re
import spacy
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

class OceanographicNLU:
    def __init__(self):
        """Initialize the NLU module with models and dictionaries"""
        
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize intent classifier (using a general sentiment classifier as placeholder)
        # In production, you'd fine-tune DistilBERT on oceanographic intents
        try:
                self.intent_classifier = pipeline(
                    "zero-shot-classification",
                    model="distilbert-base-uncased"
                )
        except Exception as e:
            print(f"Warning: Could not load intent classifier: {e}")
            self.intent_classifier = None
        
        # Intent categories for oceanographic queries
        self.intent_labels = [
            "data_discovery",
            "geospatial", 
            "temporal",
            "scientific_analysis",
            "visualization",
            "metadata_inquiry",
            "hybrid_query"
        ]
        
        # Domain-specific dictionaries
        self.parameter_synonyms = {
            # Temperature variations
            "temp": "temperature",
            "water temperature": "temperature",
            "sea temperature": "temperature",
            "ocean temperature": "temperature",
            
            # Salinity variations
            "salt": "salinity",
            "saltiness": "salinity",
            "salty water": "salinity",
            "practical salinity": "salinity",
            "psal": "salinity",
            
            # Oxygen variations
            "o2": "oxygen",
            "dissolved oxygen": "oxygen",
            "oxygen content": "oxygen",
            
            # Pressure/Depth variations
            "depth": "pressure",
            "water depth": "pressure",
            "pressure": "pressure",
            "pres": "pressure",
            
            # BGC parameters
            "ph": "ph",
            "acidity": "ph",
            "nitrate": "nitrate",
            "no3": "nitrate",
            "chlorophyll": "chlorophyll",
            "chl": "chlorophyll",
            "backscattering": "backscattering",
            "turbidity": "backscattering"
        }
        
        self.geographic_regions = {
            # Oceans
            "indian ocean": "Indian Ocean",
            "pacific ocean": "Pacific Ocean",
            "atlantic ocean": "Atlantic Ocean",
            "arctic ocean": "Arctic Ocean",
            "southern ocean": "Southern Ocean",
            
            # Seas and Bays
            "bay of bengal": "Bay of Bengal",
            "arabian sea": "Arabian Sea",
            "mediterranean sea": "Mediterranean Sea",
            "red sea": "Red Sea",
            "south china sea": "South China Sea",
            
            # Coastal regions
            "east coast": "East Coast",
            "west coast": "West Coast",
            "north coast": "North Coast",
            "south coast": "South Coast",
            
            # Countries (for deployment context)
            "india": "India",
            "australia": "Australia",
            "usa": "USA",
            "europe": "Europe"
        }
        
        self.float_types = {
            "bgc": "BGC",
            "biogeochemical": "BGC",
            "bio-argo": "BGC",
            "core": "Core",
            "deep": "Deep",
            "deep argo": "Deep"
        }
        
        # Time patterns
        self.time_patterns = {
            r'\b(\d{4})\b': 'year',
            r'\b(\d{4})-(\d{4})\b': 'year_range',
            r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b': 'date',
            r'\blast\s+(\d+)\s+(year|month|day)s?\b': 'relative_time',
            r'\bpast\s+(\d+)\s+(year|month|day)s?\b': 'relative_time',
            r'\brecent\b': 'recent',
            r'\blatest\b': 'latest',
            r'\bcurrent\b': 'current',
            r'\b(summer|winter|spring|autumn|fall)\b': 'season',
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b': 'month'
        }

    def detect_intent(self, text: str) -> str:
        """Detect the primary intent of the query"""
        if not self.intent_classifier:
            # Fallback rule-based intent detection
            text_lower = text.lower()
            
            if any(word in text_lower for word in ['plot', 'show', 'display', 'visualize', 'chart', 'graph']):
                return "visualization"
            elif any(word in text_lower for word in ['where', 'location', 'region', 'near', 'around', 'latitude', 'longitude']):
                return "geospatial"
            elif any(word in text_lower for word in ['when', 'time', 'date', 'year', 'month', 'recent', 'latest']):
                return "temporal"
            elif any(word in text_lower for word in ['what', 'how many', 'count', 'list', 'find']):
                return "data_discovery"
            elif any(word in text_lower for word in ['metadata', 'info', 'information', 'details', 'sensors']):
                return "metadata_inquiry"
            elif any(word in text_lower for word in ['compare', 'analyze', 'correlation', 'trend', 'pattern']):
                return "scientific_analysis"
            else:
                return "hybrid_query"
        
        try:
            result = self.intent_classifier(text, self.intent_labels)
            return result['labels'][0]
        except:
            return "hybrid_query"

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract named entities and domain-specific terms"""
        entities = {
            "parameters": [],
            "regions": [],
            "float_types": [],
            "float_ids": [],
            "temporal": [],
            "numeric": []
        }
        
        text_lower = text.lower()
        
        # Extract parameters using synonyms
        for synonym, standard in self.parameter_synonyms.items():
            if synonym in text_lower:
                if standard not in entities["parameters"]:
                    entities["parameters"].append(standard)
        
        # Extract geographic regions
        for region_variant, standard in self.geographic_regions.items():
            if region_variant in text_lower:
                if standard not in entities["regions"]:
                    entities["regions"].append(standard)
        
        # Extract float types
        for float_variant, standard in self.float_types.items():
            if float_variant in text_lower:
                if standard not in entities["float_types"]:
                    entities["float_types"].append(standard)
        
        # Extract float IDs (typically 7-digit numbers)
        float_id_pattern = r'\b\d{7}\b'
        float_ids = re.findall(float_id_pattern, text)
        entities["float_ids"] = float_ids
        
        # Extract temporal information
        for pattern, time_type in self.time_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities["temporal"].append({
                    "type": time_type,
                    "value": matches,
                    "raw_match": re.search(pattern, text, re.IGNORECASE).group()
                })
        
        # Extract numeric values
        numeric_pattern = r'\b\d+(?:\.\d+)?\b'
        numeric_matches = re.findall(numeric_pattern, text)
        entities["numeric"] = [float(n) for n in numeric_matches if float(n) not in [float(fid) for fid in float_ids]]
        
        # Use spaCy NER if available
        if self.nlp:
            doc = self.nlp(text)
            spacy_entities = []
            for ent in doc.ents:
                spacy_entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
            entities["spacy_entities"] = spacy_entities
        
        return entities

    def normalize_parameters(self, parameters: List[str]) -> List[str]:
        """Normalize parameter names to standard database columns"""
        normalized = []
        for param in parameters:
            # Direct mapping
            if param in self.parameter_synonyms.values():
                normalized.append(param)
            else:
                # Find closest match (simple approach)
                for synonym, standard in self.parameter_synonyms.items():
                    if param.lower() in synonym or synonym in param.lower():
                        if standard not in normalized:
                            normalized.append(standard)
                        break
                else:
                    # Keep original if no mapping found
                    normalized.append(param)
        
        return normalized

    def detect_ambiguity(self, entities: Dict[str, Any], text: str) -> List[str]:
        """Detect potential ambiguities in the query"""
        uncertainty_flags = []
        
        # Check for ambiguous regions
        if len(entities["regions"]) > 1:
            uncertainty_flags.append("multiple_regions_specified")
        
        # Check for vague geographic terms
        vague_terms = ["coast", "near", "around", "close to", "vicinity"]
        if any(term in text.lower() for term in vague_terms) and not entities["regions"]:
            uncertainty_flags.append("vague_geographic_reference")
        
        # Check for ambiguous time references
        relative_time_count = sum(1 for t in entities["temporal"] if t["type"] in ["relative_time", "recent", "latest"])
        if relative_time_count > 1:
            uncertainty_flags.append("conflicting_time_references")
        
        # Check for missing essential parameters
        if not entities["parameters"] and any(word in text.lower() for word in ["plot", "show", "analyze"]):
            uncertainty_flags.append("missing_parameter_specification")
        
        # Check for float ID conflicts
        if len(entities["float_ids"]) > 3:
            uncertainty_flags.append("too_many_float_ids")
        
        return uncertainty_flags

    def build_query_frame(self, text: str) -> Dict[str, Any]:
        """Build the complete QueryFrame from the input text"""
        
        # Step 1: Intent Detection
        intent = self.detect_intent(text)
        
        # Step 2: Entity Extraction
        entities = self.extract_entities(text)
        
        # Step 3: Parameter Normalization
        normalized_parameters = self.normalize_parameters(entities["parameters"])
        entities["parameters"] = normalized_parameters
        
        # Step 4: Ambiguity Detection
        uncertainty_flags = self.detect_ambiguity(entities, text)
        
        # Build the final QueryFrame
        query_frame = {
            "intent": intent,
            "entities": entities,
            "raw_text": text,
            "uncertainty_flags": uncertainty_flags,
            "confidence_score": self._calculate_confidence(entities, uncertainty_flags),
            "processing_timestamp": datetime.now().isoformat(),
            "suggested_clarifications": self._generate_clarifications(uncertainty_flags, entities, text)
        }
        
        return query_frame

    def _calculate_confidence(self, entities: Dict[str, Any], uncertainty_flags: List[str]) -> float:
        """Calculate confidence score based on entity extraction and ambiguity"""
        base_score = 0.8
        
        # Boost confidence for well-defined entities
        if entities["parameters"]:
            base_score += 0.1
        if entities["regions"]:
            base_score += 0.1
        if entities["temporal"]:
            base_score += 0.05
        
        # Reduce confidence for uncertainty flags
        confidence_penalty = len(uncertainty_flags) * 0.15
        
        final_score = max(0.1, min(1.0, base_score - confidence_penalty))
        return round(final_score, 2)

    def _generate_clarifications(self, uncertainty_flags: List[str], entities: Dict[str, Any], text: str) -> List[str]:
        """Generate clarification questions for ambiguous queries"""
        clarifications = []
        
        if "multiple_regions_specified" in uncertainty_flags:
            regions = ", ".join(entities["regions"])
            clarifications.append(f"Multiple regions detected: {regions}. Which specific region are you interested in?")
        
        if "vague_geographic_reference" in uncertainty_flags:
            clarifications.append("Could you specify which coast or region you're referring to?")
        
        if "missing_parameter_specification" in uncertainty_flags:
            clarifications.append("Which oceanographic parameter would you like to analyze? (e.g., temperature, salinity, oxygen)")
        
        if "conflicting_time_references" in uncertainty_flags:
            clarifications.append("Multiple time periods mentioned. Could you clarify the specific time range?")
        
        return clarifications

    def process_query(self, query: str) -> str:
        """Main method to process a query and return JSON output"""
        query_frame = self.build_query_frame(query)
        return json.dumps(query_frame, indent=2, ensure_ascii=False)


# Example usage and testing
if __name__ == "__main__":
    # Initialize the NLU module
    nlu = OceanographicNLU()
    
    # Test queries
    test_queries = [
        "Show me the latest salinity profiles for all Indian BGC floats near the equator",
        "Plot temperature data from float 6901481 in Bay of Bengal",
        "Find oxygen levels in Arabian Sea during summer 2022",
        "How many active floats are deployed by India?",
        "Compare salinity between 2020 and 2023 near the coast",
        "What sensors are on BGC floats?",
        "Show me deep floats in Indian Ocean with recent data"
    ]
    
    print("=== Oceanographic NLU Module Test Results ===\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        print("="*50)
        result = nlu.process_query(query)
        print(result)
        print("\n" + "="*80 + "\n")


# Additional utility function for batch processing
def batch_process_queries(queries: List[str], nlu_instance: OceanographicNLU) -> List[Dict[str, Any]]:
    """Process multiple queries and return list of QueryFrames"""
    results = []
    for query in queries:
        query_frame = nlu_instance.build_query_frame(query)
        results.append(query_frame)
    return results