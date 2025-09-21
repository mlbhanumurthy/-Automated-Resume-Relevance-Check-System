import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI

class SemanticAnalyzer:
    """
    Semantic analysis using TF-IDF and OpenAI for deep understanding
    """
    
    def __init__(self):
        # Initialize TF-IDF vectorizer for basic semantic analysis
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
        else:
            self.openai_client = None

    def calculate_semantic_similarity(self, resume_text, job_description):
        """
        Calculate semantic similarity between resume and job description using TF-IDF
        
        Args:
            resume_text (str): Resume text content
            job_description (str): Job description text
            
        Returns:
            dict: Semantic similarity analysis results
        """
        try:
            # Create TF-IDF vectors
            texts = [resume_text, job_description]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Break down text into sections for detailed analysis
            resume_sections = self._extract_sections(resume_text)
            
            section_similarities = {}
            for section_name, section_text in resume_sections.items():
                if section_text.strip():
                    try:
                        section_texts = [section_text, job_description]
                        section_tfidf = self.vectorizer.fit_transform(section_texts)
                        section_similarity = cosine_similarity(section_tfidf[0:1], section_tfidf[1:2])[0][0]
                        section_similarities[section_name] = float(section_similarity)
                    except:
                        section_similarities[section_name] = 0.0
            
            return {
                'similarity_score': float(similarity_score),
                'section_similarities': section_similarities,
                'interpretation': self._interpret_similarity_score(similarity_score)
            }
            
        except Exception as e:
            return {
                'similarity_score': 0.0,
                'section_similarities': {},
                'interpretation': f'Error calculating similarity: {str(e)}',
                'error': str(e)
            }

    def get_llm_analysis(self, resume_text, job_description):
        """
        Get detailed LLM analysis of resume vs job description match
        
        Args:
            resume_text (str): Resume text content
            job_description (str): Job description text
            
        Returns:
            dict: LLM analysis results
        """
        try:
            if not self.openai_client:
                return {
                    'strengths': ['OpenAI API key not configured'],
                    'improvements': ['LLM analysis requires OpenAI API key'],
                    'assessment': 'LLM analysis unavailable - OpenAI API key not provided',
                    'recommended_score': '50',
                    'key_gaps': ['Analysis unavailable'],
                    'unique_value': ['Analysis unavailable']
                }
                
            prompt = f"""
            You are an expert HR analyst and resume reviewer. Analyze the following resume against the job description and provide a comprehensive evaluation.

            JOB DESCRIPTION:
            {job_description[:2000]}

            RESUME:
            {resume_text[:2000]}

            Please provide your analysis in JSON format with the following structure:
            {{
                "strengths": ["list of candidate's key strengths that match the job"],
                "improvements": ["list of areas where candidate could improve or is lacking"],
                "assessment": "overall assessment paragraph explaining the fit",
                "recommended_score": "score from 0-100 representing overall match quality",
                "key_gaps": ["critical skills or experiences missing from the resume"],
                "unique_value": ["unique aspects of the candidate that add value"]
            }}

            Focus on both technical qualifications and soft skills. Be specific and actionable in your feedback.
            """

            # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
            # do not change this unless explicitly requested by the user
            response = self.openai_client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert HR analyst. Provide detailed, actionable resume analysis in valid JSON format."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if content:
                analysis = json.loads(content)
                return analysis
            else:
                raise ValueError("Empty response from OpenAI")
            
        except Exception as e:
            return {
                'strengths': ['Unable to perform LLM analysis'],
                'improvements': ['LLM analysis unavailable'],
                'assessment': f'Error in LLM analysis: {str(e)}',
                'recommended_score': '50',
                'key_gaps': ['Analysis unavailable'],
                'unique_value': ['Analysis unavailable'],
                'error': str(e)
            }

    def _extract_sections(self, text):
        """
        Extract different sections from resume/job description text
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary with section names as keys and content as values
        """
        sections = {
            'skills': '',
            'experience': '',
            'education': '',
            'summary': '',
            'other': ''
        }
        
        lines = text.split('\n')
        current_section = 'other'
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Identify section headers
            if any(keyword in line_lower for keyword in ['skill', 'technical', 'competenc']):
                current_section = 'skills'
            elif any(keyword in line_lower for keyword in ['experience', 'work', 'employment', 'career']):
                current_section = 'experience'
            elif any(keyword in line_lower for keyword in ['education', 'qualification', 'degree', 'academic']):
                current_section = 'education'
            elif any(keyword in line_lower for keyword in ['summary', 'objective', 'profile', 'about']):
                current_section = 'summary'
            
            sections[current_section] += line + ' '
        
        return sections

    def _interpret_similarity_score(self, score):
        """
        Interpret the semantic similarity score
        
        Args:
            score (float): Similarity score between 0 and 1
            
        Returns:
            str: Human-readable interpretation
        """
        if score >= 0.8:
            return "Excellent semantic match - very strong alignment between resume and job requirements"
        elif score >= 0.6:
            return "Good semantic match - solid alignment with some room for improvement"
        elif score >= 0.4:
            return "Moderate semantic match - some alignment but significant gaps exist"
        elif score >= 0.2:
            return "Weak semantic match - limited alignment between resume and job requirements"
        else:
            return "Poor semantic match - little to no alignment between resume and job requirements"

    def get_embedding_similarity_matrix(self, texts):
        """
        Get similarity matrix for multiple texts using TF-IDF
        
        Args:
            texts (list): List of text strings
            
        Returns:
            numpy.ndarray: Similarity matrix
        """
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            return similarity_matrix
        except Exception as e:
            return np.zeros((len(texts), len(texts)))
