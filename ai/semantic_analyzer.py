import os
import json
from typing import Dict, List, Any, Optional
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

class SemanticAnalyzer:
    """
    Uses LLMs to perform high-fidelity semantic data analysis.
    This helps in identifying sensitive columns that traditional 
    regex-based inference might miss.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if (HAS_OPENAI and self.api_key) else None

    def analyze_columns(self, headers: List[str], sample_data: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Calls OpenAI to analyze the semantic meaning of columns.
        """
        if not self.client:
            return {}

        # Prepare a prompt with headers and a few samples
        sample_json = json.dumps(sample_data[:3], indent=2)
        
        prompt = f"""
        Analyze the following CSV columns and their sample data. 
        Categorize each column into one of the following types: 
        'age', 'year', 'monetary', 'numeric', 'count', 'boolean', 'id', or 'string'.
        
        Headers: {headers}
        Sample Data: {sample_json}
        
        Return ONLY a JSON object mapping column names to categories.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are a data privacy expert."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" }
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print(f"AI Analysis Error: {e}")
            return {}
