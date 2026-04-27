import os
import json
from typing import Dict, List, Any, Optional
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

class SemanticAnalyzer:
    """
    Uses Gemini LLM to perform high-fidelity semantic data analysis.
    This helps in identifying sensitive columns that traditional 
    regex-based inference might miss.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if HAS_GEMINI and self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None

    def analyze_columns(self, headers: List[str], sample_data: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Calls Gemini to analyze the semantic meaning of columns.
        """
        if not self.model:
            return {}

        # Prepare a prompt with headers and a few samples
        sample_json = json.dumps(sample_data[:3], indent=2)
        
        prompt = f"""
        Analyze the following CSV columns and their sample data. 
        Categorize each column into one of the following types: 
        'age', 'year', 'monetary', 'numeric', 'count', 'boolean', 'id', or 'string'.
        
        Headers: {headers}
        Sample Data: {sample_json}
        
        Return ONLY a JSON object mapping column names to categories. Do not include any markdown formatting like ```json or other text.
        """

        try:
            response = self.model.generate_content(prompt)
            # Clean up response text if it contains markdown code blocks
            text = response.text.strip()
            if text.startswith("```json"):
                text = text.replace("```json", "").replace("```", "").strip()
            elif text.startswith("```"):
                text = text.replace("```", "").strip()
                
            result = json.loads(text)
            return result
        except Exception as e:
            print(f"AI Analysis Error: {e}")
            return {}

    def detect_bias_context(self, headers: List[str], sample_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Uses Gemini to identify sensitive attributes, target variables, and potential proxies.
        """
        if not self.model:
            return {}

        sample_json = json.dumps(sample_data[:5], indent=2)
        
        prompt = f"""
        Analyze the following dataset metadata and sample data for fairness and bias concerns.
        
        Headers: {headers}
        Sample Data: {sample_json}
        
        Your task is to:
        1. Identify 'sensitive_attributes' (e.g., race, gender, age, religion).
        2. Identify the likely 'target_variable' (the column a model would predict).
        3. Identify 'proxy_attributes' (columns that might leak sensitive info like ZIP code, school, etc.).
        4. Provide 'fairness_risk_level' (Low, Medium, High).
        5. Suggest 'mitigation_strategies'.
        
        Return ONLY a JSON object with these keys. No other text. Do not include any markdown formatting like ```json.
        """

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            if text.startswith("```json"):
                text = text.replace("```json", "").replace("```", "").strip()
            elif text.startswith("```"):
                text = text.replace("```", "").strip()
                
            return json.loads(text)
        except Exception as e:
            print(f"AI Bias Detection Error: {e}")
            return {}

    def generate_insights(self, score: float, status: str, findings: List[str], impacts: Dict[str, float]) -> str:
        """
        Generates AI-driven insights for the diagnostic report.
        """
        if not self.model:
            return ""

        prompt = f"""
        You are a data scientist interpreting a dataset diagnostic report.
        
        Health Score: {score} ({status})
        Top Findings: {findings}
        Top Predictors: {impacts}
        
        Provide 2-3 brief 'Dataset Insights' for a non-technical manager. 
        Explain what these results mean for their business or analysis.
        Avoid jargon. Keep it to 3 bullets max.
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"AI Insights Error: {e}")
            return ""

