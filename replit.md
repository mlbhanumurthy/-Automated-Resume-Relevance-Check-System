# Overview

The AI Resume Evaluation Engine is a Streamlit-based web application that provides automated resume analysis and scoring. The system combines rule-based matching with semantic analysis to evaluate resumes against job descriptions, offering both individual resume evaluation and batch processing capabilities. It features a placement dashboard for tracking evaluation results and uses hybrid scoring methodologies to provide comprehensive resume assessments.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit Framework**: Web-based user interface with multiple pages for different functionalities
- **Custom Styling**: Times New Roman font family applied across the application for professional presentation
- **Multi-page Navigation**: Sidebar navigation supporting Resume Evaluation, Batch Processing, and Placement Dashboard pages
- **Component Caching**: Streamlit's `@st.cache_resource` decorator used for efficient component initialization

## Backend Architecture
- **Modular Design**: Separated utility modules for different concerns (text extraction, scoring, semantic analysis, database operations)
- **Hybrid Scoring Engine**: Combines rule-based keyword matching with statistical TF-IDF analysis for comprehensive evaluation
- **Text Processing Pipeline**: Multi-stage text extraction supporting PDF and DOCX formats using PyPDF2 and python-docx libraries

## Data Storage Solutions
- **PostgreSQL Database**: Primary data store with dedicated tables for job descriptions and evaluations
- **JSON Fallback**: Local JSON files for storing evaluations and job descriptions when database is unavailable
- **Database Schema**: Structured tables with UUID primary keys, timestamps, and comprehensive metadata fields

## Scoring Methodology
- **Rule-based Matching**: Predefined skill categories including programming languages, web development, databases, cloud technologies, and soft skills
- **Semantic Analysis**: TF-IDF vectorization with cosine similarity calculations for content matching
- **Section-based Evaluation**: Resume parsing into logical sections (experience, education, skills) for targeted analysis

## Authentication and Authorization
- **Environment-based Configuration**: Database connection and API keys managed through environment variables
- **No Built-in Authentication**: Application relies on deployment environment for access control

# External Dependencies

## Core Libraries
- **Streamlit**: Web application framework for the user interface
- **pandas**: Data manipulation and analysis for evaluation results
- **scikit-learn**: Machine learning utilities for TF-IDF vectorization and similarity calculations
- **NLTK**: Natural language processing for tokenization and stopword removal

## Document Processing
- **PyPDF2**: PDF text extraction capabilities
- **python-docx**: Microsoft Word document text extraction

## Database Integration
- **psycopg2**: PostgreSQL database adapter with connection pooling and real dictionary cursor support

## Optional AI Enhancement
- **OpenAI API**: Semantic analysis enhancement through GPT models (configured via OPENAI_API_KEY environment variable)

## System Dependencies
- **PostgreSQL**: Production database requiring DATABASE_URL environment variable
- **NLTK Data**: Automatic download of punkt tokenizer and stopwords corpus during initialization

The application is designed with fallback mechanisms, allowing operation without external AI services while maintaining core functionality through statistical methods and rule-based analysis.