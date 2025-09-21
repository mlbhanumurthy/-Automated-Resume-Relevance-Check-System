import streamlit as st
import pandas as pd
import numpy as np
import PyPDF2

def extract_text_from_file(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

from datetime import datetime
import json
from utils.text_extractor import extract_text_from_file
from utils.scoring_engine import HybridScoringEngine
from utils.semantic_analyzer import SemanticAnalyzer
from utils.database import Database

# Initialize components
@st.cache_resource
def init_components():
    scoring_engine = HybridScoringEngine()
    semantic_analyzer = SemanticAnalyzer()
    database = Database()
    return scoring_engine, semantic_analyzer, database

def main():
    st.set_page_config(
        page_title="AI Resume Evaluation Engine",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply Times New Roman font styling
    st.markdown("""
    <style>
    .main .block-container {
        font-family: 'Times New Roman', Times, serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Times New Roman', Times, serif;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎯 AI-Powered Resume Evaluation Engine")
    st.markdown("*Hybrid rule-based and semantic analysis for resume scoring*")
    
    # Initialize components
    scoring_engine, semantic_analyzer, database = init_components()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Page", [
        "Resume Evaluation", 
        "Batch Resume Processing",
        "Placement Dashboard", 
        "Job Description Management"
    ])
    
    if page == "Resume Evaluation":
        resume_evaluation_page(scoring_engine, semantic_analyzer, database)
    elif page == "Batch Resume Processing":
        batch_processing_page(scoring_engine, semantic_analyzer, database)
    elif page == "Placement Dashboard":
        placement_dashboard_page(database)
    elif page == "Job Description Management":
        job_description_management_page(database)

def resume_evaluation_page(scoring_engine, semantic_analyzer, database):
    st.header("📄 Resume Evaluation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Resume")
        resume_file = st.file_uploader(
            "Upload Resume (PDF/DOCX)", 
            type=['pdf', 'docx'],
            help="Upload a PDF or DOCX resume file for evaluation"
        )
        
        candidate_name = st.text_input("Candidate Name", placeholder="Enter candidate's name")
        candidate_email = st.text_input("Candidate Email", placeholder="candidate@email.com")
    
    with col2:
        st.subheader("Select Job Description")
        job_descriptions = database.get_job_descriptions()
        
        if job_descriptions:
            job_options = {f"{job['title']} - {job['company']}": job['id'] for job in job_descriptions}
            selected_job = st.selectbox("Choose Job Description", options=list(job_options.keys()))
            job_id = job_options[selected_job] if selected_job else None
        else:
            st.warning("No job descriptions available. Please add job descriptions first.")
            job_id = None
    
    if st.button("🚀 Evaluate Resume", type="primary", use_container_width=True):
        if resume_file and job_id and candidate_name:
            with st.spinner("Analyzing resume... This may take a moment."):
                try:
                    # Extract text from resume
                    resume_text = extract_text_from_file(resume_file)
                    
                    if not resume_text.strip():
                        st.error("Could not extract text from the resume. Please check the file format.")
                        return
                    
                    # Get job description
                    job_desc = database.get_job_description(job_id)
                    
                    # Perform hybrid scoring
                    hard_score_result = scoring_engine.calculate_hard_score(resume_text, job_desc['description'])
                    soft_score_result = semantic_analyzer.calculate_semantic_similarity(resume_text, job_desc['description'])
                    
                    # Get LLM analysis
                    llm_analysis = semantic_analyzer.get_llm_analysis(resume_text, job_desc['description'])
                    
                    # Calculate final score
                    final_score = (hard_score_result['score'] * 0.4 + soft_score_result['similarity_score'] * 0.6) * 100
                    
                    # Determine verdict
                    if final_score >= 80:
                        verdict = "Excellent Match"
                        verdict_color = "green"
                    elif final_score >= 60:
                        verdict = "Good Match"
                        verdict_color = "orange"
                    elif final_score >= 40:
                        verdict = "Fair Match"
                        verdict_color = "orange"
                    else:
                        verdict = "Poor Match"
                        verdict_color = "red"
                    
                    # Display results
                    st.success("✅ Resume evaluation completed!")
                    
                    # Score display
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Hard Match Score", f"{hard_score_result['score']*100:.1f}%")
                    with col2:
                        st.metric("Semantic Score", f"{soft_score_result['similarity_score']*100:.1f}%")
                    with col3:
                        st.metric("Final Score", f"{final_score:.1f}%")
                    
                    # Verdict
                    st.markdown(f"### Verdict: :{verdict_color}[{verdict}]")
                    
                    # Detailed analysis
                    st.subheader("📊 Detailed Analysis")
                    
                    # Hard match details
                    with st.expander("🔍 Hard Match Analysis", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Matched Skills:**")
                            if hard_score_result['matched_skills']:
                                for skill in hard_score_result['matched_skills']:
                                    st.write(f"✅ {skill}")
                            else:
                                st.write("No direct skill matches found")
                        
                        with col2:
                            st.write("**Missing Keywords:**")
                            if hard_score_result['missing_keywords']:
                                for keyword in hard_score_result['missing_keywords'][:10]:  # Limit to 10
                                    st.write(f"❌ {keyword}")
                            else:
                                st.write("No missing keywords identified")
                    
                    # LLM analysis
                    with st.expander("🤖 AI Semantic Analysis", expanded=True):
                        if llm_analysis:
                            st.write("**Strengths:**")
                            for strength in llm_analysis.get('strengths', []):
                                st.write(f"✅ {strength}")
                            
                            st.write("**Areas for Improvement:**")
                            for improvement in llm_analysis.get('improvements', []):
                                st.write(f"📈 {improvement}")
                            
                            st.write("**Overall Assessment:**")
                            st.write(llm_analysis.get('assessment', 'No assessment available'))
                    
                    # Save evaluation
                    evaluation_data = {
                        'candidate_name': candidate_name,
                        'candidate_email': candidate_email,
                        'job_id': job_id,
                        'job_title': job_desc['title'],
                        'company': job_desc['company'],
                        'hard_score': hard_score_result['score'] * 100,
                        'semantic_score': soft_score_result['similarity_score'] * 100,
                        'final_score': final_score,
                        'verdict': verdict,
                        'matched_skills': hard_score_result['matched_skills'],
                        'missing_keywords': hard_score_result['missing_keywords'],
                        'llm_analysis': llm_analysis,
                        'evaluation_date': datetime.now().isoformat(),
                        'resume_text': resume_text[:1000]  # Store first 1000 chars for reference
                    }
                    
                    database.save_evaluation(evaluation_data)
                    st.success("💾 Evaluation saved to database!")
                    
                except Exception as e:
                    st.error(f"❌ Error during evaluation: {str(e)}")
        else:
            st.error("Please provide all required information: resume file, candidate name, and job selection.")

def batch_processing_page(scoring_engine, semantic_analyzer, database):
    st.header("🚀 Batch Resume Processing")
    st.markdown("*Upload multiple resumes for efficient batch evaluation*")
    
    # Job description selection
    job_descriptions = database.get_job_descriptions()
    
    if not job_descriptions:
        st.warning("⚠️ No job descriptions available. Please add job descriptions first.")
        st.info("Go to 'Job Description Management' to add job descriptions before batch processing.")
        return
    
    job_options = {f"{job['title']} - {job['company']}": job['id'] for job in job_descriptions}
    selected_job = st.selectbox("📋 Select Job Description for Batch Evaluation", 
                               options=list(job_options.keys()),
                               help="All uploaded resumes will be evaluated against this job description")
    job_id = job_options[selected_job] if selected_job else None
    
    # Multi-file upload
    st.subheader("📄 Upload Multiple Resumes")
    resume_files = st.file_uploader(
        "Select Resume Files (PDF/DOCX)", 
        type=['pdf', 'docx'],
        accept_multiple_files=True,
        help="You can select multiple resume files at once for batch processing"
    )
    
    if resume_files and job_id:
        st.success(f"✅ {len(resume_files)} resume(s) uploaded successfully!")
        
        # Show file list
        with st.expander("📋 Uploaded Files", expanded=True):
            for i, file in enumerate(resume_files, 1):
                st.write(f"{i}. {file.name} ({file.size} bytes)")
        
        # Batch processing options
        col1, col2 = st.columns(2)
        with col1:
            include_candidate_names = st.checkbox(
                "Extract candidate names from filenames", 
                value=True,
                help="Use filename (without extension) as candidate name"
            )
        with col2:
            save_to_database = st.checkbox(
                "Save results to database", 
                value=True,
                help="Store evaluation results in database for future reference"
            )
        
        # Processing options
        col3, col4 = st.columns(2)
        with col3:
            fast_mode = st.checkbox(
                "Fast Mode (Skip AI Analysis)",
                value=False,
                help="Skip LLM analysis for faster batch processing"
            )
        with col4:
            max_batch_size = st.number_input(
                "Max batch size",
                min_value=1,
                max_value=50,
                value=20,
                help="Maximum number of resumes to process in one batch"
            )
        
        # Process batch button
        if st.button("🚀 Process Batch", type="primary", use_container_width=True):
            # Check batch size limit
            if len(resume_files) > max_batch_size:
                st.error(f"❌ Batch size ({len(resume_files)}) exceeds maximum limit ({max_batch_size}). Please reduce the number of files.")
                return
            
            process_batch_resumes(resume_files, job_id, scoring_engine, semantic_analyzer, 
                                database, include_candidate_names, save_to_database, fast_mode)
    
    elif resume_files:
        st.info("Please select a job description to proceed with batch processing.")
    else:
        st.info("Please upload resume files to start batch processing.")

def process_batch_resumes(resume_files, job_id, scoring_engine, semantic_analyzer, 
                         database, include_candidate_names, save_to_database, fast_mode=False):
    """Process multiple resumes in batch"""
    
    # Get job description
    job_desc = database.get_job_description(job_id)
    if not job_desc:
        st.error("❌ Selected job description not found!")
        return
    
    st.subheader("🔄 Processing Batch...")
    
    # Initialize results storage
    batch_results = []
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process each resume
    for i, resume_file in enumerate(resume_files):
        try:
            # Update progress  
            progress_bar.progress(i / len(resume_files))
            status_text.text(f"Processing {resume_file.name}... ({i+1}/{len(resume_files)})")
            
            # Extract candidate name from filename
            if include_candidate_names:
                candidate_name = resume_file.name.rsplit('.', 1)[0]  # Remove extension
            else:
                candidate_name = f"Candidate_{i+1}"
            
            # Extract text from resume
            resume_text = extract_text_from_file(resume_file)
            
            if not resume_text.strip():
                st.warning(f"⚠️ Could not extract text from {resume_file.name}")
                batch_results.append({
                    'candidate_name': candidate_name,
                    'filename': resume_file.name,
                    'status': 'Failed - No text extracted',
                    'error': 'Could not extract text from resume'
                })
                continue
            
            # Perform hybrid scoring
            hard_score_result = scoring_engine.calculate_hard_score(resume_text, job_desc['description'])
            soft_score_result = semantic_analyzer.calculate_semantic_similarity(resume_text, job_desc['description'])
            
            # Calculate final score first (needed for LLM analysis)
            final_score = (hard_score_result['score'] * 0.4 + soft_score_result['similarity_score'] * 0.6) * 100
            
            # Get LLM analysis (skip in fast mode)
            if fast_mode:
                llm_analysis = {
                    'strengths': ['Fast mode - detailed analysis skipped'],
                    'improvements': ['Run without fast mode for detailed analysis'],
                    'assessment': 'Fast mode evaluation - detailed AI analysis skipped for speed',
                    'recommended_score': str(int(final_score)),
                    'key_gaps': ['Analysis skipped'],
                    'unique_value': ['Analysis skipped']
                }
            else:
                try:
                    llm_analysis = semantic_analyzer.get_llm_analysis(resume_text, job_desc['description'])
                except Exception as e:
                    st.warning(f"⚠️ LLM analysis failed for {resume_file.name}: {str(e)}")
                    llm_analysis = {
                        'strengths': ['LLM analysis failed'],
                        'improvements': ['Analysis unavailable due to error'],
                        'assessment': f'LLM analysis error: {str(e)}',
                        'recommended_score': str(int(final_score)),
                        'key_gaps': ['Analysis failed'],
                        'unique_value': ['Analysis failed']
                    }
            
            # Determine verdict
            if final_score >= 80:
                verdict = "Excellent Match"
            elif final_score >= 60:
                verdict = "Good Match"
            elif final_score >= 40:
                verdict = "Fair Match"
            else:
                verdict = "Poor Match"
            
            # Store result
            result = {
                'candidate_name': candidate_name,
                'filename': resume_file.name,
                'job_title': job_desc['title'],
                'company': job_desc['company'],
                'hard_score': hard_score_result['score'] * 100,
                'semantic_score': soft_score_result['similarity_score'] * 100,
                'final_score': final_score,
                'verdict': verdict,
                'matched_skills': hard_score_result['matched_skills'],
                'missing_keywords': hard_score_result['missing_keywords'],
                'llm_analysis': llm_analysis,
                'status': 'Success'
            }
            
            batch_results.append(result)
            
            # Save to database if requested
            if save_to_database:
                evaluation_data = {
                    'candidate_name': candidate_name,
                    'candidate_email': '',  # Not available in batch processing
                    'job_id': job_id,
                    'job_title': job_desc['title'],
                    'company': job_desc['company'],
                    'hard_score': hard_score_result['score'] * 100,
                    'semantic_score': soft_score_result['similarity_score'] * 100,
                    'final_score': final_score,
                    'verdict': verdict,
                    'matched_skills': hard_score_result['matched_skills'],
                    'missing_keywords': hard_score_result['missing_keywords'],
                    'llm_analysis': llm_analysis,
                    'evaluation_date': datetime.now().isoformat(),
                    'resume_text': resume_text[:1000]  # Store first 1000 chars for reference
                }
                try:
                    database.save_evaluation(evaluation_data)
                except Exception as e:
                    st.warning(f"⚠️ Failed to save evaluation for {candidate_name}: {str(e)}")
        
        except Exception as e:
            batch_results.append({
                'candidate_name': candidate_name if 'candidate_name' in locals() else f"Candidate_{i+1}",
                'filename': resume_file.name,
                'status': 'Failed',
                'error': str(e)
            })
    
    # Complete progress
    progress_bar.progress(1.0)
    status_text.text("✅ Batch processing completed!")
    
    # Display results
    display_batch_results(batch_results, job_desc, save_to_database, fast_mode)

def display_batch_results(batch_results, job_desc, saved_to_db, fast_mode=False):
    """Display batch processing results"""
    
    st.success(f"🎉 Batch processing completed for {len(batch_results)} resume(s)!")
    
    # Calculate summary statistics
    successful_results = [r for r in batch_results if r['status'] == 'Success']
    failed_results = [r for r in batch_results if r['status'] != 'Success']
    
    if successful_results:
        avg_score = sum([r['final_score'] for r in successful_results]) / len(successful_results)
        excellent_matches = len([r for r in successful_results if r['verdict'] == 'Excellent Match'])
        good_matches = len([r for r in successful_results if r['verdict'] == 'Good Match'])
    else:
        avg_score = 0
        excellent_matches = 0
        good_matches = 0
    
    # Display summary
    st.subheader("📊 Batch Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Processed", len(batch_results))
    with col2:
        st.metric("Successful", len(successful_results))
    with col3:
        st.metric("Average Score", f"{avg_score:.1f}%")
    with col4:
        st.metric("Excellent Matches", excellent_matches)
    
    if failed_results:
        with st.expander(f"❌ Failed Processing ({len(failed_results)} files)", expanded=False):
            for result in failed_results:
                st.error(f"**{result['filename']}**: {result.get('error', 'Unknown error')}")
    
    # Display successful results
    if successful_results:
        st.subheader("📋 Evaluation Results")
        
        # Create results DataFrame
        df_data = []
        for result in successful_results:
            df_data.append({
                'Candidate Name': result['candidate_name'],
                'Filename': result['filename'],
                'Final Score': f"{result['final_score']:.1f}%",
                'Verdict': result['verdict'],
                'Hard Score': f"{result['hard_score']:.1f}%",
                'Semantic Score': f"{result['semantic_score']:.1f}%",
            })
        
        results_df = pd.DataFrame(df_data)
        
        # Color code verdicts
        def color_verdict(val):
            if val == 'Excellent Match':
                return 'background-color: #d4edda'
            elif val == 'Good Match':
                return 'background-color: #fff3cd'
            elif val == 'Fair Match':
                return 'background-color: #f8d7da'
            else:
                return 'background-color: #f5c6cb'
        
        # Add verdict colors as a helper column for display
        results_df['Verdict_Color'] = results_df['Verdict'].apply(lambda x: 
            '🟢' if x == 'Excellent Match' else 
            '🟡' if x == 'Good Match' else 
            '🟠' if x == 'Fair Match' else '🔴'
        )
        
        # Reorder columns for better display
        display_columns = ['Verdict_Color', 'Candidate Name', 'Filename', 'Final Score', 'Verdict', 'Hard Score', 'Semantic Score']
        results_df = results_df[display_columns]
        
        st.dataframe(results_df, use_container_width=True, height=400)
        
        # Detailed results
        with st.expander("🔍 Detailed Analysis", expanded=False):
            selected_candidate = st.selectbox("Select candidate for detailed analysis", 
                                            options=[r['candidate_name'] for r in successful_results])
            
            if selected_candidate:
                candidate_result = next(r for r in successful_results if r['candidate_name'] == selected_candidate)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Final Score:** {candidate_result['final_score']:.1f}%")
                    st.write(f"**Verdict:** {candidate_result['verdict']}")
                    st.write(f"**Job Applied:** {candidate_result['job_title']}")
                    st.write(f"**Company:** {candidate_result['company']}")
                
                with col2:
                    st.write("**Top Matched Skills:**")
                    if candidate_result['matched_skills']:
                        for skill in candidate_result['matched_skills'][:5]:
                            st.write(f"✅ {skill}")
                    else:
                        st.write("No direct skill matches found")
                
                # LLM Analysis
                if candidate_result.get('llm_analysis') and isinstance(candidate_result['llm_analysis'], dict):
                    llm_data = candidate_result['llm_analysis']
                    st.write("**AI Assessment:**")
                    st.write(llm_data.get('assessment', 'No assessment available'))
    
    if saved_to_db:
        st.info("💾 All successful evaluations have been saved to the database and are available in the Placement Dashboard.")
    
    # Export options
    if successful_results:
        st.subheader("📥 Export Results")
        
        col1, col2 = st.columns(2)
        with col1:
            # CSV download
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="📄 Download as CSV",
                data=csv_data,
                file_name=f"batch_evaluation_results_{job_desc['title'].replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # JSON download with detailed results
            json_data = json.dumps(successful_results, indent=2, default=str)
            st.download_button(
                label="📋 Download Detailed JSON",
                data=json_data,
                file_name=f"batch_evaluation_detailed_{job_desc['title'].replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True
            )

def placement_dashboard_page(database):
    st.header("📊 Placement Dashboard")
    
    # Advanced Filters Section
    with st.expander("🔍 Advanced Filters", expanded=True):
        # Row 1: Score and Search
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Score Range")
            score_range = st.slider("Score Range", 0, 100, (0, 100))
            min_score, max_score = score_range
        with col2:
            st.subheader("Search")
            search_term = st.text_input("Search candidates", placeholder="Enter candidate name or email...")
        with col3:
            st.subheader("Date Range")
            date_filter = st.selectbox("Date Filter", ["All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days", "Custom Range"])
        
        # Row 2: Verdict and Job/Company
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Verdict")
            verdict_options = ["All", "Excellent Match", "Good Match", "Fair Match", "Poor Match"]
            verdict_filter = st.multiselect("Select Verdicts", verdict_options, default=["All"])
            if "All" in verdict_filter:
                verdict_filter = ["All"]
        with col2:
            st.subheader("Job Position")
            jobs = database.get_job_descriptions()
            job_options = ["All"] + [job['title'] for job in jobs]
            job_filter = st.selectbox("Job Filter", job_options)
        with col3:
            st.subheader("Company")
            companies = list(set([job['company'] for job in jobs]))
            company_options = ["All"] + sorted(companies)
            company_filter = st.selectbox("Company Filter", company_options)
        
        # Row 3: Sorting and Actions
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Sort By")
            sort_options = {
                "Score (High to Low)": ("final_score", False),
                "Score (Low to High)": ("final_score", True),
                "Date (Newest First)": ("evaluation_date", False),
                "Date (Oldest First)": ("evaluation_date", True),
                "Candidate Name (A-Z)": ("candidate_name", True),
                "Candidate Name (Z-A)": ("candidate_name", False),
                "Job Title": ("job_title", True),
                "Company": ("company", True)
            }
            sort_by = st.selectbox("Sort By", list(sort_options.keys()))
            sort_column, sort_ascending = sort_options[sort_by]
        with col2:
            st.subheader("Actions")
            if st.button("🗑️ Clear All Filters", key="clear_filters"):
                st.rerun()
        with col3:
            st.subheader("Email Filter")
            email_domain = st.text_input("Email Domain", placeholder="e.g., gmail.com")
        
        # Custom date range if selected
        if date_filter == "Custom Range":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date")
            with col2:
                end_date = st.date_input("End Date")
    
    # Get evaluations
    evaluations = database.get_evaluations()
    
    if evaluations:
        df = pd.DataFrame(evaluations)
        original_count = len(df)
        
        # Apply advanced filters
        filtered_df = df.copy()
        
        # 1. Score range filter
        filtered_df = filtered_df[
            (filtered_df['final_score'] >= min_score) & 
            (filtered_df['final_score'] <= max_score)
        ]
        
        # 2. Search filter (candidate name or email)
        if search_term:
            search_mask = (
                filtered_df['candidate_name'].str.contains(search_term, case=False, na=False) |
                filtered_df['candidate_email'].str.contains(search_term, case=False, na=False)
            )
            filtered_df = filtered_df[search_mask]
        
        # 3. Verdict filter
        if verdict_filter != ["All"] and "All" not in verdict_filter:
            filtered_df = filtered_df[filtered_df['verdict'].isin(verdict_filter)]
        
        # 4. Job filter
        if job_filter != "All":
            filtered_df = filtered_df[filtered_df['job_title'] == job_filter]
        
        # 5. Company filter
        if company_filter != "All":
            filtered_df = filtered_df[filtered_df['company'] == company_filter]
        
        # 6. Email domain filter
        if email_domain:
            email_mask = filtered_df['candidate_email'].str.endswith(f"@{email_domain}", na=False)
            filtered_df = filtered_df[email_mask]
        
        # 7. Date range filter
        if date_filter != "All Time":
            # Convert evaluation_date to datetime if it's string
            filtered_df['evaluation_date'] = pd.to_datetime(filtered_df['evaluation_date'], errors='coerce')
            current_date = datetime.now()
            
            if date_filter == "Last 7 Days":
                cutoff_date = current_date - pd.Timedelta(days=7)
                filtered_df = filtered_df[filtered_df['evaluation_date'] >= cutoff_date]
            elif date_filter == "Last 30 Days":
                cutoff_date = current_date - pd.Timedelta(days=30)
                filtered_df = filtered_df[filtered_df['evaluation_date'] >= cutoff_date]
            elif date_filter == "Last 90 Days":
                cutoff_date = current_date - pd.Timedelta(days=90)
                filtered_df = filtered_df[filtered_df['evaluation_date'] >= cutoff_date]
            elif date_filter == "Custom Range":
                if 'start_date' in locals() and 'end_date' in locals():
                    start_datetime = pd.to_datetime(start_date)
                    end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1)  # Include end date
                    filtered_df = filtered_df[
                        (filtered_df['evaluation_date'] >= start_datetime) & 
                        (filtered_df['evaluation_date'] <= end_datetime)
                    ]
        
        # 8. Apply sorting
        if len(filtered_df) > 0:
            try:
                if sort_column == "evaluation_date":
                    filtered_df['evaluation_date'] = pd.to_datetime(filtered_df['evaluation_date'], errors='coerce')
                filtered_df = filtered_df.sort_values(by=sort_column, ascending=sort_ascending, na_position='last')
            except KeyError:
                # Fallback to default sorting if column doesn't exist
                filtered_df = filtered_df.sort_values(by='final_score', ascending=False)
        
        df = filtered_df
        
        # Display filter summary
        active_filters = []
        if min_score > 0 or max_score < 100:
            active_filters.append(f"Score: {min_score}-{max_score}%")
        if search_term:
            active_filters.append(f"Search: '{search_term}'")
        if verdict_filter != ["All"] and "All" not in verdict_filter:
            active_filters.append(f"Verdicts: {', '.join(verdict_filter)}")
        if job_filter != "All":
            active_filters.append(f"Job: {job_filter}")
        if company_filter != "All":
            active_filters.append(f"Company: {company_filter}")
        if email_domain:
            active_filters.append(f"Email domain: @{email_domain}")
        if date_filter != "All Time":
            active_filters.append(f"Date: {date_filter}")
        
        if active_filters:
            st.info(f"🔍 **Active Filters:** {' | '.join(active_filters)} | **Results:** {len(df)} of {original_count} evaluations")
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Filtered Results", len(df), f"{len(df) - original_count:+d}" if len(df) != original_count else None)
        with col2:
            excellent_matches = len(df[df['verdict'] == 'Excellent Match']) if len(df) > 0 else 0
            st.metric("Excellent Matches", excellent_matches)
        with col3:
            avg_score = df['final_score'].mean() if len(df) > 0 else 0
            st.metric("Average Score", f"{avg_score:.1f}%")
        with col4:
            good_matches = len(df[df['final_score'] >= 60]) if len(df) > 0 else 0
            st.metric("Good+ Matches", good_matches)
        with col5:
            top_score = df['final_score'].max() if len(df) > 0 else 0
            st.metric("Top Score", f"{top_score:.1f}%")
        
        # Display evaluations table
        if len(df) > 0:
            st.subheader("Evaluation Results")
            
            # Format display dataframe
            display_df = df[[
                'candidate_name', 'candidate_email', 'job_title', 'company', 
                'final_score', 'verdict', 'evaluation_date'
            ]].copy()
            
            display_df['final_score'] = display_df['final_score'].round(1)
            # Handle datetime formatting for evaluation_date
            def format_date(date_val):
                if pd.isna(date_val):
                    return 'Unknown'
                if isinstance(date_val, str):
                    try:
                        return pd.to_datetime(date_val).strftime('%Y-%m-%d %H:%M')
                    except:
                        return str(date_val)
                else:
                    return date_val.strftime('%Y-%m-%d %H:%M')
            
            display_df['evaluation_date'] = display_df['evaluation_date'].apply(format_date)
            
            # Add verdict color indicators using emojis
            def add_verdict_emoji(verdict):
                emoji_map = {
                    'Excellent Match': '🟢',
                    'Good Match': '🟡', 
                    'Fair Match': '🟠',
                    'Poor Match': '🔴'
                }
                return f"{emoji_map.get(verdict, '⚪')} {verdict}"
            
            # Replace verdict with emoji version
            display_df['verdict'] = display_df['verdict'].apply(add_verdict_emoji)
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Export filtered results section
            st.subheader("📥 Export Filtered Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV export for filtered results
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📄 Export as CSV",
                    data=csv_data,
                    file_name=f"filtered_evaluations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # JSON export for filtered results
                json_data = json.dumps(df.to_dict('records'), indent=2, default=str)
                st.download_button(
                    label="📋 Export as JSON",
                    data=json_data,
                    file_name=f"filtered_evaluations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col3:
                # Export summary statistics
                summary_stats = {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_records': len(df),
                    'active_filters': active_filters,
                    'statistics': {
                        'average_score': float(df['final_score'].mean()) if len(df) > 0 else 0,
                        'top_score': float(df['final_score'].max()) if len(df) > 0 else 0,
                        'excellent_matches': int(excellent_matches),
                        'good_matches': int(good_matches),
                        'verdict_breakdown': df['verdict'].value_counts().to_dict() if len(df) > 0 else {}
                    }
                }
                summary_json = json.dumps(summary_stats, indent=2, default=str)
                st.download_button(
                    label="📊 Export Summary",
                    data=summary_json,
                    file_name=f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            # Detailed view
            st.subheader("Detailed View")
            selected_candidate = st.selectbox("Select candidate for detailed view", 
                                            options=df['candidate_name'].tolist())
            
            if selected_candidate:
                candidate_data = df[df['candidate_name'] == selected_candidate].iloc[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Email:** {candidate_data['candidate_email']}")
                    st.write(f"**Job Applied:** {candidate_data['job_title']}")
                    st.write(f"**Company:** {candidate_data['company']}")
                    st.write(f"**Final Score:** {candidate_data['final_score']:.1f}%")
                    st.write(f"**Verdict:** {candidate_data['verdict']}")
                
                with col2:
                    st.write("**Matched Skills:**")
                    matched_skills = candidate_data.get('matched_skills', [])
                    if matched_skills:
                        for skill in matched_skills[:10]:
                            st.write(f"✅ {skill}")
                    else:
                        st.write("No matched skills recorded")
                
                # LLM Analysis
                if candidate_data.get('llm_analysis'):
                    llm_data = candidate_data['llm_analysis']
                    if isinstance(llm_data, str):
                        try:
                            llm_data = json.loads(llm_data)
                        except:
                            pass
                    
                    if isinstance(llm_data, dict):
                        st.write("**AI Assessment:**")
                        st.write(llm_data.get('assessment', 'No assessment available'))
        else:
            st.info("No evaluations match the current filters.")
    else:
        st.info("No evaluations found. Start by evaluating some resumes!")

def job_description_management_page(database):
    st.header("💼 Job Description Management")
    
    # Add new job description
    with st.expander("➕ Add New Job Description", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            job_title = st.text_input("Job Title", placeholder="e.g., Software Engineer")
            company = st.text_input("Company", placeholder="e.g., Tech Corp")
        with col2:
            location = st.text_input("Location", placeholder="e.g., New York, NY")
            employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Internship"])
        
        job_description = st.text_area("Job Description", 
                                     placeholder="Paste the complete job description here...",
                                     height=200)
        
        if st.button("💾 Save Job Description", type="primary"):
            if job_title and company and job_description:
                job_data = {
                    'title': job_title,
                    'company': company,
                    'location': location,
                    'employment_type': employment_type,
                    'description': job_description,
                    'created_date': datetime.now().isoformat()
                }
                database.save_job_description(job_data)
                st.success("✅ Job description saved successfully!")
                st.rerun()
            else:
                st.error("Please fill in all required fields: Job Title, Company, and Description.")
    
    # Display existing job descriptions
    st.subheader("📋 Existing Job Descriptions")
    job_descriptions = database.get_job_descriptions()
    
    if job_descriptions:
        for job in job_descriptions:
            with st.expander(f"{job['title']} at {job['company']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Location:** {job.get('location', 'Not specified')}")
                    st.write(f"**Type:** {job.get('employment_type', 'Not specified')}")
                    created_date = job.get('created_date', 'Unknown')
                    if isinstance(created_date, str):
                        date_str = created_date[:10]
                    else:
                        date_str = created_date.strftime('%Y-%m-%d') if created_date != 'Unknown' else 'Unknown'
                    st.write(f"**Created:** {date_str}")
                
                with col2:
                    if st.button(f"🗑️ Delete", key=f"delete_{job['id']}"):
                        database.delete_job_description(job['id'])
                        st.success("Job description deleted!")
                        st.rerun()
                
                st.write("**Description:**")
                st.write(job['description'][:500] + "..." if len(job['description']) > 500 else job['description'])
    else:
        st.info("No job descriptions found. Add your first job description above!")

if __name__ == "__main__":
    main()
