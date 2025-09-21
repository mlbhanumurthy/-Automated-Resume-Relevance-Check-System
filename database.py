import json
import os
from datetime import datetime
import uuid
import psycopg2
from psycopg2.extras import RealDictCursor
import streamlit as st

class Database:
    """
    PostgreSQL database for storing job descriptions and evaluations
    """
    
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")
        if not self.database_url:
            raise Exception("DATABASE_URL environment variable not found")
        
        # Initialize database tables
        self._initialize_database()
    
    def _get_connection(self):
        """Get database connection"""
        try:
            return psycopg2.connect(self.database_url)
        except Exception as e:
            st.error(f"Database connection error: {str(e)}")
            return None
    
    def _initialize_database(self):
        """Initialize database tables if they don't exist"""
        try:
            conn = self._get_connection()
            if not conn:
                return
            
            cursor = conn.cursor()
            
            # Create job_descriptions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS job_descriptions (
                    id VARCHAR(36) PRIMARY KEY,
                    title VARCHAR(255) NOT NULL,
                    company VARCHAR(255) NOT NULL,
                    location VARCHAR(255),
                    employment_type VARCHAR(100),
                    description TEXT NOT NULL,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create evaluations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id VARCHAR(36) PRIMARY KEY,
                    candidate_name VARCHAR(255) NOT NULL,
                    candidate_email VARCHAR(255),
                    job_id VARCHAR(36) REFERENCES job_descriptions(id) ON DELETE CASCADE,
                    job_title VARCHAR(255),
                    company VARCHAR(255),
                    hard_score FLOAT,
                    semantic_score FLOAT,
                    final_score FLOAT,
                    verdict VARCHAR(100),
                    matched_skills TEXT,
                    missing_keywords TEXT,
                    llm_analysis TEXT,
                    resume_text TEXT,
                    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            st.error(f"Database initialization error: {str(e)}")
    
    def save_job_description(self, job_data):
        """
        Save a job description to the database
        
        Args:
            job_data (dict): Job description data
        """
        try:
            conn = self._get_connection()
            if not conn:
                raise Exception("Database connection failed")
            
            cursor = conn.cursor()
            
            # Add unique ID
            job_id = str(uuid.uuid4())
            
            cursor.execute("""
                INSERT INTO job_descriptions 
                (id, title, company, location, employment_type, description, created_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                job_id,
                job_data['title'],
                job_data['company'],
                job_data.get('location', ''),
                job_data.get('employment_type', ''),
                job_data['description'],
                datetime.now()
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            raise Exception(f"Error saving job description: {str(e)}")
    
    def get_job_descriptions(self):
        """
        Get all job descriptions from the database
        
        Returns:
            list: List of job descriptions
        """
        try:
            conn = self._get_connection()
            if not conn:
                return []
            
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT * FROM job_descriptions 
                ORDER BY created_date DESC
            """)
            
            jobs = [dict(row) for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            
            return jobs
            
        except Exception as e:
            st.error(f"Error fetching job descriptions: {str(e)}")
            return []
    
    def get_job_description(self, job_id):
        """
        Get a specific job description by ID
        
        Args:
            job_id (str): Job ID
            
        Returns:
            dict: Job description data
        """
        try:
            conn = self._get_connection()
            if not conn:
                return None
            
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM job_descriptions WHERE id = %s", (job_id,))
            
            job = cursor.fetchone()
            cursor.close()
            conn.close()
            
            return dict(job) if job else None
            
        except Exception as e:
            st.error(f"Error fetching job description: {str(e)}")
            return None
    
    def delete_job_description(self, job_id):
        """
        Delete a job description from the database
        
        Args:
            job_id (str): Job ID to delete
        """
        try:
            conn = self._get_connection()
            if not conn:
                raise Exception("Database connection failed")
            
            cursor = conn.cursor()
            cursor.execute("DELETE FROM job_descriptions WHERE id = %s", (job_id,))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            raise Exception(f"Error deleting job description: {str(e)}")
    
    def save_evaluation(self, evaluation_data):
        """
        Save an evaluation result to the database
        
        Args:
            evaluation_data (dict): Evaluation result data
        """
        try:
            conn = self._get_connection()
            if not conn:
                raise Exception("Database connection failed")
            
            cursor = conn.cursor()
            
            # Add unique ID
            evaluation_id = str(uuid.uuid4())
            
            cursor.execute("""
                INSERT INTO evaluations 
                (id, candidate_name, candidate_email, job_id, job_title, company, 
                 hard_score, semantic_score, final_score, verdict, matched_skills, 
                 missing_keywords, llm_analysis, resume_text, evaluation_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                evaluation_id,
                evaluation_data['candidate_name'],
                evaluation_data.get('candidate_email', ''),
                evaluation_data['job_id'],
                evaluation_data['job_title'],
                evaluation_data['company'],
                evaluation_data['hard_score'],
                evaluation_data['semantic_score'],
                evaluation_data['final_score'],
                evaluation_data['verdict'],
                json.dumps(evaluation_data.get('matched_skills', [])),
                json.dumps(evaluation_data.get('missing_keywords', [])),
                json.dumps(evaluation_data.get('llm_analysis', {})),
                evaluation_data.get('resume_text', ''),
                datetime.now()
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            raise Exception(f"Error saving evaluation: {str(e)}")
    
    def get_evaluations(self):
        """
        Get all evaluations from the database
        
        Returns:
            list: List of evaluations
        """
        try:
            conn = self._get_connection()
            if not conn:
                return []
            
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT * FROM evaluations 
                ORDER BY evaluation_date DESC
            """)
            
            evaluations = []
            for row in cursor.fetchall():
                eval_data = dict(row)
                # Parse JSON fields
                try:
                    eval_data['matched_skills'] = json.loads(eval_data['matched_skills']) if eval_data['matched_skills'] else []
                    eval_data['missing_keywords'] = json.loads(eval_data['missing_keywords']) if eval_data['missing_keywords'] else []
                    eval_data['llm_analysis'] = json.loads(eval_data['llm_analysis']) if eval_data['llm_analysis'] else {}
                except json.JSONDecodeError:
                    eval_data['matched_skills'] = []
                    eval_data['missing_keywords'] = []
                    eval_data['llm_analysis'] = {}
                
                evaluations.append(eval_data)
            
            cursor.close()
            conn.close()
            
            return evaluations
            
        except Exception as e:
            st.error(f"Error fetching evaluations: {str(e)}")
            return []
    
    def get_evaluations_by_job(self, job_id):
        """
        Get evaluations for a specific job
        
        Args:
            job_id (str): Job ID
            
        Returns:
            list: List of evaluations for the job
        """
        try:
            conn = self._get_connection()
            if not conn:
                return []
            
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM evaluations WHERE job_id = %s ORDER BY evaluation_date DESC", (job_id,))
            
            evaluations = []
            for row in cursor.fetchall():
                eval_data = dict(row)
                # Parse JSON fields
                try:
                    eval_data['matched_skills'] = json.loads(eval_data['matched_skills']) if eval_data['matched_skills'] else []
                    eval_data['missing_keywords'] = json.loads(eval_data['missing_keywords']) if eval_data['missing_keywords'] else []
                    eval_data['llm_analysis'] = json.loads(eval_data['llm_analysis']) if eval_data['llm_analysis'] else {}
                except json.JSONDecodeError:
                    eval_data['matched_skills'] = []
                    eval_data['missing_keywords'] = []
                    eval_data['llm_analysis'] = {}
                
                evaluations.append(eval_data)
            
            cursor.close()
            conn.close()
            
            return evaluations
            
        except Exception as e:
            st.error(f"Error fetching evaluations by job: {str(e)}")
            return []
    
    def get_evaluations_by_candidate(self, candidate_name):
        """
        Get evaluations for a specific candidate
        
        Args:
            candidate_name (str): Candidate name
            
        Returns:
            list: List of evaluations for the candidate
        """
        try:
            conn = self._get_connection()
            if not conn:
                return []
            
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM evaluations WHERE candidate_name = %s ORDER BY evaluation_date DESC", (candidate_name,))
            
            evaluations = []
            for row in cursor.fetchall():
                eval_data = dict(row)
                # Parse JSON fields
                try:
                    eval_data['matched_skills'] = json.loads(eval_data['matched_skills']) if eval_data['matched_skills'] else []
                    eval_data['missing_keywords'] = json.loads(eval_data['missing_keywords']) if eval_data['missing_keywords'] else []
                    eval_data['llm_analysis'] = json.loads(eval_data['llm_analysis']) if eval_data['llm_analysis'] else {}
                except json.JSONDecodeError:
                    eval_data['matched_skills'] = []
                    eval_data['missing_keywords'] = []
                    eval_data['llm_analysis'] = {}
                
                evaluations.append(eval_data)
            
            cursor.close()
            conn.close()
            
            return evaluations
            
        except Exception as e:
            st.error(f"Error fetching evaluations by candidate: {str(e)}")
            return []
    
    def delete_evaluation(self, evaluation_id):
        """
        Delete an evaluation from the database
        
        Args:
            evaluation_id (str): Evaluation ID to delete
        """
        try:
            conn = self._get_connection()
            if not conn:
                raise Exception("Database connection failed")
            
            cursor = conn.cursor()
            cursor.execute("DELETE FROM evaluations WHERE id = %s", (evaluation_id,))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            raise Exception(f"Error deleting evaluation: {str(e)}")
    
    def get_statistics(self):
        """
        Get database statistics
        
        Returns:
            dict: Statistics about jobs and evaluations
        """
        try:
            conn = self._get_connection()
            if not conn:
                return {
                    'total_jobs': 0,
                    'total_evaluations': 0,
                    'average_score': 0,
                    'verdict_distribution': {}
                }
            
            cursor = conn.cursor()
            
            # Get total jobs
            cursor.execute("SELECT COUNT(*) FROM job_descriptions")
            jobs_result = cursor.fetchone()
            total_jobs = jobs_result[0] if jobs_result else 0
            
            # Get total evaluations
            cursor.execute("SELECT COUNT(*) FROM evaluations")
            evals_result = cursor.fetchone()
            total_evaluations = evals_result[0] if evals_result else 0
            
            # Get average score
            cursor.execute("SELECT AVG(final_score) FROM evaluations")
            avg_result = cursor.fetchone()
            avg_result = avg_result[0] if avg_result else None
            avg_score = float(avg_result) if avg_result else 0
            
            # Get verdict distribution
            cursor.execute("SELECT verdict, COUNT(*) FROM evaluations GROUP BY verdict")
            verdict_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            cursor.close()
            conn.close()
            
            return {
                'total_jobs': total_jobs,
                'total_evaluations': total_evaluations,
                'average_score': avg_score,
                'verdict_distribution': verdict_counts
            }
            
        except Exception as e:
            return {
                'total_jobs': 0,
                'total_evaluations': 0,
                'average_score': 0,
                'verdict_distribution': {},
                'error': str(e)
            }