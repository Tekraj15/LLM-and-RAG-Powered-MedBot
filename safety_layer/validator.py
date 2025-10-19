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
        self._load_safety_patterns()
        self._load_medical_disclaimers()
        logger.info("MedicalResponseValidator initialized")

    def _load_safety_patterns(self) -> None:
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
        # Required disclaimer keywords
        self.required_disclaimers = ["consult", "healthcare", "professional", "doctor", "physician"]

    def _load_medical_disclaimers(self) -> None:
        """Load appropriate medical disclaimers"""
        self.disclaimers = {
            "general": "This information is for educational purposes only and should not replace professional medical advice. Please consult a healthcare provider for personalized guidance.",
            "medication": "Medication information is general. Follow your doctor's instructions and consult a pharmacist or physician before changes.",
            "symptom": "Symptom information is for awareness only. Consult a qualified healthcare professional for diagnosis and treatment.",
            "emergency": "If experiencing a medical emergency, call emergency services immediately (911/999/112) or go to the nearest emergency room.",
            "chronic": "Chronic condition management requires medical supervision. This supplements, but does not replace, your healthcare team's guidance.",
            "mental_health": "Mental health support is general guidance. Consult a licensed therapist or counselor. In crisis, contact emergency services or a helpline."
        }

    def validate_response(self, response: str, sources: Optional[List[Dict[str, Any]]] = None, 
                         query_type: Optional[str] = None) -> ValidationResult:
        """
        Comprehensive validation of medical response
        """
        try:
            response_lower = response.lower()
            warnings = []
            recommendations = []

            # Check for critical unsafe content
            if self._contains_critical_unsafe_content(response_lower):
                return ValidationResult(
                    level=ValidationLevel.UNSAFE,
                    is_safe=False,
                    modified_response=self._get_safe_fallback_response(),
                    warnings=["Potentially harmful medical advice detected"],
                    sources_verified=False,
                    confidence_score=0.0,
                    recommendations=["Consult a medical professional immediately"]
                )

            # Check for emergency indicators
            if self._contains_emergency_indicators(response_lower):
                enhanced_response = self._enhance_emergency_response(response)
                return ValidationResult(
                    level=ValidationLevel.EMERGENCY,
                    is_safe=True,
                    modified_response=enhanced_response,
                    warnings=["Emergency situation detected"],
                    sources_verified=True,
                    confidence_score=1.0,
                    recommendations=["Seek immediate medical attention"]
                )

            # Check for warning patterns
            modified_response = response
            validation_level = ValidationLevel.SAFE
            if self._contains_warning_patterns(response_lower):
                modified_response = self._add_safety_qualifiers(response)
                validation_level = ValidationLevel.WARNING
                warnings.append("Safety modifications applied")
                recommendations.append("Added qualifiers and disclaimers")

            # Verify sources
            sources_verified = self._verify_sources(sources) if sources else False
            if sources and not sources_verified:
                warnings.append("Incomplete source verification")
                recommendations.append("Verify source credibility")

            # Add disclaimer if missing
            has_disclaimer = self._has_appropriate_disclaimer(response_lower)
            if not has_disclaimer:
                modified_response = self._add_disclaimer(modified_response, query_type)
                recommendations.append("Added medical disclaimer")

            # Calculate confidence
            confidence_score = self._calculate_confidence_score(modified_response, sources, validation_level)

            return ValidationResult(
                level=validation_level,
                is_safe=True,
                modified_response=modified_response,
                warnings=warnings,
                sources_verified=sources_verified,
                confidence_score=confidence_score,
                recommendations=recommendations
            )
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return ValidationResult(
                level=ValidationLevel.UNSAFE,
                is_safe=False,
                modified_response=self._get_safe_fallback_response(),
                warnings=[f"Validation failed: {str(e)}"],
                sources_verified=False,
                confidence_score=0.0,
                recommendations=["Manual review required"]
            )

    def _contains_critical_unsafe_content(self, response: str) -> bool:
        """Check for critically unsafe medical advice"""
        return any(re.search(pattern, response, re.IGNORECASE) for pattern in self.critical_unsafe_patterns)

    def _contains_warning_patterns(self, response: str) -> bool:
        """Check for patterns requiring warnings"""
        return any(re.search(pattern, response, re.IGNORECASE) for pattern in self.warning_patterns)

    def _contains_emergency_indicators(self, response: str) -> bool:
        """Check for emergency medical situations"""
        return any(re.search(pattern, response, re.IGNORECASE) for pattern in self.emergency_indicators)

    def _has_appropriate_disclaimer(self, response: str) -> bool:
        """Check if response contains appropriate disclaimers"""
        return sum(1 for keyword in self.required_disclaimers if keyword in response) >= 2

    def _add_safety_qualifiers(self, response: str) -> str:
        """Add safety qualifiers to risky statements"""
        response = re.sub(
            r'\b(you should|recommended|try)\b',
            r'you might consider discussing with your doctor',
            response,
            flags=re.IGNORECASE
        )
        response = re.sub(
            r'\b(you have|diagnosed with|condition is)\b',
            r'symptoms may suggest - consult a doctor for diagnosis',
            response,
            flags=re.IGNORECASE
        )
        return response

    def _add_disclaimer(self, response: str, query_type: Optional[str] = None) -> str:
        """Add appropriate disclaimer based on query type"""
        disclaimer_type = "general"
        if query_type:
            query_type_lower = query_type.lower()
            if any(key in query_type_lower for key in ["medication", "drug"]):
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
        return f"{response}\n\n**Medical Disclaimer!:** {disclaimer}"

    def _enhance_emergency_response(self, response: str) -> str:
        """Enhance emergency responses with priority messaging"""
        return (f"**MEDICAL EMERGENCY!** \n\n{response}\n\n"
                "**IMMEDIATE ACTIONS:**\n• Call emergency services now (911/999/112)\n"
                "• Do not delay seeking professional help\n• Follow responder instructions")

    def _verify_sources(self, sources: List[Dict[str, Any]]) -> bool:
        """Verify source credibility and recency"""
        if not sources:
            return False
        credible_sources = {"internal_kb", "cdc", "who", "drugbank", "medlineplus", "emergency protocol"}
        verified = sum(1 for s in sources if any(cs in s.get("source", "").lower() for cs in credible_sources) and s.get("confidence", 0.0) >= 0.7)
        return verified >= len(sources) * 0.5

    def _calculate_confidence_score(self, response: str, sources: Optional[List[Dict[str, Any]]], 
                                  validation_level: ValidationLevel) -> float:
        """Calculate confidence score"""
        base_score = 0.8
        adjustments = {
            ValidationLevel.SAFE: 0.0,
            ValidationLevel.WARNING: -0.1,
            ValidationLevel.UNSAFE: -0.8,
            ValidationLevel.EMERGENCY: 0.2
        }
        score = base_score + adjustments.get(validation_level, -0.2)
        if sources:
            avg_confidence = sum(s.get("confidence", 0.5) for s in sources) / len(sources)
            score = (score + avg_confidence) / 2
        else:
            score -= 0.1
        if "medical disclaimer" in response.lower():
            score += 0.05
        return max(0.0, min(1.0, score))

    def _get_safe_fallback_response(self) -> str:
        """Return a safe fallback response"""
        return ("I'm unable to provide specific medical advice. For your safety, consult a "
                "qualified healthcare professional for personalized guidance.\n\n"
                f"**Medical Disclaimer!:** {self.disclaimers['general']}")

if __name__ == "__main__":
    validator = MedicalResponseValidator()
    test_responses = [
        "Stop taking your medication immediately.",  # Unsafe
        "You might have diabetes. Consult your doctor.",  # Warning
        "Call 911 if you have chest pain.",  # Emergency
        "Ibuprofen helps pain. Follow dosage instructions."  # Safe
    ]
    for response in test_responses:
        result = validator.validate_response(response)
        print(f"Response: {response[:50]}...")
        print(f"Level: {result.level.value}, Confidence: {result.confidence_score:.2f}")
        print(f"Modified: {result.modified_response[:50]}...")
        print("---")