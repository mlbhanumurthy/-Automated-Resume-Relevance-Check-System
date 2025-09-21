import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class HybridScoringEngine:
    """
    Hybrid scoring engine that combines rule-based matching with statistical analysis
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
        # Common technical skills and keywords
        self.technical_skills = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'go', 'rust', 'swift', 'kotlin'],
            'web_development': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'bootstrap'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle', 'sql server', 'sqlite'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins', 'gitlab'],
            'data_science': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'matplotlib', 'jupyter'],
            'mobile': ['android', 'ios', 'react native', 'flutter', 'xamarin'],
            'devops': ['git', 'github', 'bitbucket', 'linux', 'bash', 'ansible', 'puppet', 'chef'],
            'soft_skills': ['leadership', 'communication', 'teamwork', 'problem solving', 'analytical', 'creative']
        }
        
        # Education keywords
        self.education_keywords = [
            'bachelor', 'master', 'phd', 'degree', 'university', 'college', 'graduation',
            'computer science', 'software engineering', 'information technology', 'engineering'
        ]
        
        # Experience keywords
        self.experience_keywords = [
            'years', 'experience', 'worked', 'developed', 'managed', 'led', 'created',
            'implemented', 'designed', 'built', 'maintained', 'deployed'
        ]

    def calculate_hard_score(self, resume_text, job_description):
        """
        Calculate hard score based on keyword matching and rule-based analysis
        
        Args:
            resume_text (str): Resume text content
            job_description (str): Job description text
            
        Returns:
            dict: Hard score analysis results
        """
        resume_text = resume_text.lower()
        job_description = job_description.lower()
        
        # Extract skills from job description
        job_skills = self._extract_skills(job_description)
        resume_skills = self._extract_skills(resume_text)
        
        # Find matched skills
        matched_skills = list(set(job_skills) & set(resume_skills))
        
        # Calculate skill match ratio
        skill_match_score = len(matched_skills) / len(job_skills) if job_skills else 0
        
        # Extract important keywords from job description
        job_keywords = self._extract_keywords(job_description)
        resume_keywords = self._extract_keywords(resume_text)
        
        # Find matched keywords
        matched_keywords = list(set(job_keywords) & set(resume_keywords))
        missing_keywords = list(set(job_keywords) - set(resume_keywords))
        
        # Calculate keyword match score
        keyword_match_score = len(matched_keywords) / len(job_keywords) if job_keywords else 0
        
        # Education matching
        education_score = self._calculate_education_score(resume_text, job_description)
        
        # Experience matching
        experience_score = self._calculate_experience_score(resume_text, job_description)
        
        # Calculate overall hard score (weighted average)
        hard_score = (
            skill_match_score * 0.4 +
            keyword_match_score * 0.3 +
            education_score * 0.2 +
            experience_score * 0.1
        )
        
        return {
            'score': hard_score,
            'matched_skills': matched_skills,
            'missing_keywords': missing_keywords[:15],  # Limit to top 15 missing
            'skill_match_score': skill_match_score,
            'keyword_match_score': keyword_match_score,
            'education_score': education_score,
            'experience_score': experience_score
        }

    def _extract_skills(self, text):
        """Extract technical skills from text"""
        skills = []
        
        for category, skill_list in self.technical_skills.items():
            for skill in skill_list:
                if skill.lower() in text.lower():
                    skills.append(skill)
        
        return list(set(skills))

    def _extract_keywords(self, text):
        """Extract important keywords from text using TF-IDF"""
        try:
            # Tokenize and clean text
            tokens = word_tokenize(text.lower())
            
            # Filter out stopwords and non-alphabetic tokens
            keywords = [token for token in tokens 
                       if token.isalpha() and token not in self.stop_words and len(token) > 2]
            
            # Create document for TF-IDF
            doc_text = ' '.join(keywords)
            
            # Use TF-IDF to find important terms
            vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform([doc_text])
            
            # Get feature names (keywords)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get TF-IDF scores
            scores = tfidf_matrix.toarray()[0]
            
            # Sort keywords by score
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top keywords
            return [keyword for keyword, score in keyword_scores[:50] if score > 0]
            
        except Exception as e:
            # Fallback to simple keyword extraction
            tokens = word_tokenize(text.lower())
            return [token for token in tokens 
                   if token.isalpha() and token not in self.stop_words and len(token) > 2][:50]

    def _calculate_education_score(self, resume_text, job_description):
        """Calculate education matching score"""
        resume_education_matches = 0
        job_education_requirements = 0
        
        for keyword in self.education_keywords:
            if keyword in job_description:
                job_education_requirements += 1
                if keyword in resume_text:
                    resume_education_matches += 1
        
        return resume_education_matches / job_education_requirements if job_education_requirements > 0 else 0.5

    def _calculate_experience_score(self, resume_text, job_description):
        """Calculate experience matching score"""
        # Extract years of experience mentioned
        resume_years = self._extract_years_experience(resume_text)
        job_years = self._extract_years_experience(job_description)
        
        experience_keywords_match = 0
        job_experience_keywords = 0
        
        for keyword in self.experience_keywords:
            if keyword in job_description:
                job_experience_keywords += 1
                if keyword in resume_text:
                    experience_keywords_match += 1
        
        keyword_score = experience_keywords_match / job_experience_keywords if job_experience_keywords > 0 else 0.5
        
        # Years experience score
        if resume_years >= job_years:
            years_score = 1.0
        elif resume_years >= job_years * 0.7:
            years_score = 0.8
        elif resume_years >= job_years * 0.5:
            years_score = 0.6
        else:
            years_score = 0.3
        
        return (keyword_score + years_score) / 2

    def _extract_years_experience(self, text):
        """Extract years of experience from text"""
        # Look for patterns like "5 years", "3+ years", "2-4 years"
        patterns = [
            r'(\d+)\+?\s*years?\s*of\s*experience',
            r'(\d+)\+?\s*years?\s*experience',
            r'(\d+)\+?\s*years?',
            r'(\d+)-(\d+)\s*years?'
        ]
        
        years = []
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                if isinstance(match, tuple):
                    # Range pattern (e.g., "2-4 years")
                    years.extend([int(x) for x in match if x.isdigit()])
                else:
                    # Single number pattern
                    if match.isdigit():
                        years.append(int(match))
        
        return max(years) if years else 0
