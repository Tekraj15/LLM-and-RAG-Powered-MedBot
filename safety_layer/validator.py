"""
Enhanced Medical Response Validator with Source Traceability
Implements multi-layer safety validation for medical AI responses.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation severity levels"""
    SAFE = "safe"
    WARNING = "warning" 
    UNSAFE = "unsafe"
    EMERGENCY = "emergency"

@dataclass
class ValidationResult:
    """Result of safety validation with detailed feedback"""
    level: ValidationLevel
    is_safe: bool
    modified_response: str
    warnings: List[str]
    sources_verified: bool
    confidence_score: float
    recommendations: List[str]

class MedicalResponseValidator:
    """
    Advanced validator for medical chatbot responses with source verification
    """
    
    def __init__(self):
        """Initialize the validator with safety patterns and rules"""
        self.load_safety_patterns()
        self.load_medical_disclaimers()
        logger.info("MedicalResponseValidator initialized")
    
    def load_safety_patterns(self) -> None:
        """Load patterns for detecting unsafe medical advice"""
        
        # Critical unsafe patterns (immediate rejection)
        self.critical_unsafe_patterns = [
            r'\b(stop taking|discontinue|quit)\s+(medication|medicine|pills?)\b',
            r'\b(take more|increase dose|double dose)\b',
            r'\b(you have|diagnosed with|definitely have)\b',
            r'\b(cure|will fix|guaranteed to work)\b',
            r'\b(never see a doctor|don\'t need medical care)\b',
            r'\b(instead of seeing doctor|replace medical treatment)\b'
        ]
        
        # Warning patterns (require modification)
        self.warning_patterns = [
            r'\b(should take|recommended dose|try taking)\b',
            r'\b(probably|likely|might be|could be)\s+\w+\s+(disease|condition|syndrome)\b',
            r'\b(home remedy|natural cure|alternative to medicine)\b',
            r'\b(self-treat|treat yourself|manage on your own)\b',
            r'\b(medical advice|diagnosis|treatment plan)\b'
        ]
        
        # Emergency indicators
        self.emergency_indicators = [
            r'\b(call 911|emergency room|immediate medical attention)\b',
            r'\b(life-threatening|critical condition|urgent care)\b',
            r'\b(heart attack|stroke|severe allergic reaction)\b'
        ]
        
        # Required disclaimers keywords
        self.required_disclaimers = [
            "consult", "healthcare", "professional", "doctor", "physician"
        ]
    
    def load_medical_disclaimers(self) -> None:
        """Load appropriate medical disclaimers"""
        
        self.disclaimers = {
            "general": "This information is for educational purposes only and should not replace professional medical advice. Please consult a healthcare provider for personalized guidance.",
            
            "medication": "Medication information provided is general in nature. Always follow your doctor's instructions and consult a pharmacist or physician before making any changes to your medication regimen.",
            
            "symptom": "Symptom information is provided for general awareness only. For accurate diagnosis and treatment, please consult a qualified healthcare professional.",
            
            "emergency": "If you're experiencing a medical emergency, please call emergency services immediately (911 in US, 999 in UK, 112 in EU) or go to the nearest emergency room.",
            
            "chronic": "Management of chronic conditions requires ongoing medical supervision. This information supplements but does not replace your healthcare team's guidance.",
            
            "mental_health": "Mental health support information is provided for general guidance. For professional mental health care, please consult a licensed therapist or counselor. In crisis situations, contact emergency services or a crisis helpline."
        }
    
    def validate_response(self, response: str, sources: Optional[List[Dict[str, Any]]] = None, 
                         query_type: Optional[str] = None) -> ValidationResult:
        """
        Comprehensive validation of medical response
        
        Args:
            response: The AI-generated response to validate
            sources: List of sources used in generating the response
            query_type: Type of medical query (for context-specific validation)
            
        Returns:
            ValidationResult with safety assessment and modifications
        """
        try:
            response_lower = response.lower()
            warnings = []
            recommendations = []
            
            # Check for critical unsafe patterns
            if self._contains_critical_unsafe_content(response_lower):
                return ValidationResult(
                    level=ValidationLevel.UNSAFE,
                    is_safe=False,
                    modified_response=self._get_safe_fallback_response(),
                    warnings=["Response contained potentially harmful medical advice"],
                    sources_verified=False,
                    confidence_score=0.0,
                    recommendations=["Human medical professional consultation required"]
                )
            
            # Check for emergency indicators
            if self._contains_emergency_indicators(response_lower):
                emergency_response = self._enhance_emergency_response(response)
                return ValidationResult(
                    level=ValidationLevel.EMERGENCY,
                    is_safe=True,
                    modified_response=emergency_response,
                    warnings=["Emergency situation detected"],
                    sources_verified=True,
                    confidence_score=1.0,
                    recommendations=["Immediate medical attention required"]
                )
            
            # Check for warning patterns and modify if needed
            modified_response = response
            warning_level = ValidationLevel.SAFE
            
            if self._contains_warning_patterns(response_lower):
                modified_response = self._add_safety_qualifiers(response)
                warning_level = ValidationLevel.WARNING
                warnings.append("Response required safety modifications")
                recommendations.append("Added medical disclaimers and safety qualifiers")
            
            # Verify source attribution
            sources_verified = self._verify_sources(sources) if sources else False
            if sources and not sources_verified:
                warnings.append("Source verification incomplete")
                recommendations.append("Verify source credibility and recency")
            
            # Check for required disclaimers
            has_disclaimer = self._has_appropriate_disclaimer(response_lower)
            if not has_disclaimer:
                modified_response = self._add_disclaimer(modified_response, query_type)
                recommendations.append("Added appropriate medical disclaimer")
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                modified_response, sources, warning_level
            )
            
            return ValidationResult(
                level=warning_level,
                is_safe=True,
                modified_response=modified_response,
                warnings=warnings,
                sources_verified=sources_verified,
                confidence_score=confidence_score,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error in validation: {str(e)}")
            return ValidationResult(
                level=ValidationLevel.UNSAFE,
                is_safe=False,
                modified_response=self._get_safe_fallback_response(),
                warnings=[f"Validation error: {str(e)}"],
                sources_verified=False,
                confidence_score=0.0,
                recommendations=["Manual review required"]
            )
    
    def _contains_critical_unsafe_content(self, response: str) -> bool:
        """Check for critically unsafe medical advice"""
        return any(re.search(pattern, response, re.IGNORECASE) 
                  for pattern in self.critical_unsafe_patterns)
    
    def _contains_warning_patterns(self, response: str) -> bool:
        """Check for patterns that require warnings"""
        return any(re.search(pattern, response, re.IGNORECASE) 
                  for pattern in self.warning_patterns)
    
    def _contains_emergency_indicators(self, response: str) -> bool:
        """Check for emergency medical situations"""
        return any(re.search(pattern, response, re.IGNORECASE) 
                  for pattern in self.emergency_indicators)
    
    def _has_appropriate_disclaimer(self, response: str) -> bool:
        """Check if response contains appropriate medical disclaimers"""
        disclaimer_count = sum(1 for keyword in self.required_disclaimers 
                             if keyword in response)
        return disclaimer_count >= 2  # Require at least 2 disclaimer keywords
    
    def _add_safety_qualifiers(self, response: str) -> str:
        """Add safety qualifiers to potentially risky statements"""
        
        # Add qualifiers for medical advice
        response = re.sub(
            r'\b(you should|recommended|try)\b',
            r'you might consider discussing with your doctor',
            response,
            flags=re.IGNORECASE
        )
        
        # Add qualifiers for diagnostic suggestions
        response = re.sub(
            r'\b(you have|diagnosed with|condition is)\b',
            r'symptoms may suggest - consult a doctor for proper diagnosis',
            response,
            flags=re.IGNORECASE
        )
        
        return response
    
    def _add_disclaimer(self, response: str, query_type: Optional[str] = None) -> str:
        """Add appropriate disclaimer based on query type"""
        
        disclaimer_type = "general"
        if query_type:
            query_type_lower = query_type.lower()
            if "medication" in query_type_lower or "drug" in query_type_lower:
                disclaimer_type = "medication"
            elif "symptom" in query_type_lower:
                disclaimer_type = "symptom"
            elif "emergency" in query_type_lower:
                disclaimer_type = "emergency"
            elif "chronic" in query_type_lower:
                disclaimer_type = "chronic"
            elif "mental" in query_type_lower:
                disclaimer_type = "mental_health"
        
        disclaimer = self.disclaimers.get(disclaimer_type, self.disclaimers["general"])
        
        return f"{response}\n\n⚠️ **Medical Disclaimer:** {disclaimer}"
    
    def _enhance_emergency_response(self, response: str) -> str:
        """Enhance emergency responses with priority messaging"""
        
        emergency_prefix = "🚨 **MEDICAL EMERGENCY** 🚨\n\n"
        emergency_suffix = "\n\n**IMMEDIATE ACTIONS:**\n• Call emergency services now (911/999/112)\n• Do not delay seeking professional medical help\n• Follow emergency responder instructions\n• Stay calm and seek immediate assistance"
        
        return f"{emergency_prefix}{response}{emergency_suffix}"
    
    def _verify_sources(self, sources: List[Dict[str, Any]]) -> bool:
        """Verify the credibility and recency of sources"""
        if not sources:
            return False
        
        credible_sources = ["internal_kb", "CDC", "WHO", "DrugBank", "MedlinePlus", "Emergency Protocol"]
        verified_count = 0
        
        for source in sources:
            source_name = source.get("source", "").lower()
            confidence = source.get("confidence", 0.0)
            
            # Check if source is from credible list
            if any(credible in source_name for credible in credible_sources):
                verified_count += 1
            
            # Check confidence threshold
            if confidence < 0.7:
                return False
        
        # Require at least 50% of sources to be verified
        return verified_count >= len(sources) * 0.5
    
    def _calculate_confidence_score(self, response: str, sources: Optional[List[Dict[str, Any]]], 
                                  validation_level: ValidationLevel) -> float:
        """Calculate overall confidence score for the response"""
        
        base_score = 0.8
        
        # Adjust based on validation level
        level_adjustments = {
            ValidationLevel.SAFE: 0.0,
            ValidationLevel.WARNING: -0.1,
            ValidationLevel.UNSAFE: -0.8,
            ValidationLevel.EMERGENCY: 0.2
        }
        
        score = base_score + level_adjustments.get(validation_level, -0.2)
        
        # Adjust based on source quality
        if sources:
            avg_source_confidence = sum(s.get("confidence", 0.5) for s in sources) / len(sources)
            score = (score + avg_source_confidence) / 2
        else:
            score -= 0.1  # Penalty for no sources
        
        # Adjust based on disclaimer presence
        if "medical disclaimer" in response.lower():
            score += 0.05
        
        return max(0.0, min(1.0, score))  # Clamp between 0 and 1
    
    def _get_safe_fallback_response(self) -> str:
        """Return a safe fallback response for unsafe content"""
        return ("I'm unable to provide specific medical advice for your query. "
                "For your safety and to get accurate information, please consult "
                "a qualified healthcare professional who can properly assess your "
                "individual situation and provide personalized guidance.\n\n"
                f"⚠️ **Medical Disclaimer:** {self.disclaimers['general']}")

# Backward compatibility
def validate_response(response: str) -> str:
    """Legacy function for backward compatibility"""
    validator = MedicalResponseValidator()
    result = validator.validate_response(response)
    return result.modified_response

# Example usage
if __name__ == "__main__":
    validator = MedicalResponseValidator()
    
    # Test various response types
    test_responses = [
        "You should stop taking your medication immediately.",  # Unsafe
        "You might have diabetes. Consult your doctor for proper testing.",  # Warning
        "If you're having chest pain, call 911 immediately.",  # Emergency
        "Ibuprofen can help with pain relief. Always follow dosage instructions."  # Safe
    ]
    
    for response in test_responses:
        result = validator.validate_response(response)
        print(f"Response: {response[:50]}...")
        print(f"Level: {result.level.value}")
        print(f"Safe: {result.is_safe}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print("---")