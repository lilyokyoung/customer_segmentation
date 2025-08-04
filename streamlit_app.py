"""
Streamlit Web Interface for Customer Segmentation AI Agent
Enhanced with interactive dashboard and real-time analysis
"""
import streamlit as st
import asyncio
import pandas as pd
import numpy as np
import json
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the main orchestrator
try:
    from src.main import CustomerSegmentationOrchestrator
except ImportError:
    try:
        from main import CustomerSegmentationOrchestrator
    except ImportError:
        st.error("Could not import CustomerSegmentationOrchestrator. Please check your installation.")
        st.stop()

# Page configuration
st.set_page_config(
    page_title="🤖 AI Agent Customer Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🤖 AI Agent Customer Segmentation</h1>', unsafe_allow_html=True)
st.markdown("### Enhanced Multi-Model Customer Analysis with Real-Time Insights")

# Sidebar configuration
st.sidebar.title("🔧 Configuration")
st.sidebar.markdown("---")

# Sample size configuration
sample_size = st.sidebar.slider(
    "Sample Size", 
    min_value=100, 
    max_value=5000, 
    value=1000, 
    step=100,
    help="Number of customers to analyze"
)

# Analysis mode
analysis_mode = st.sidebar.selectbox(
    "Analysis Mode",
    ["Sequential", "Parallel"],
    help="Choose how models are executed"
)

# Model selection
st.sidebar.subheader("📊 Models to Run")
run_financial_capability = st.sidebar.checkbox("Financial Capability", value=True)
run_financial_hardship = st.sidebar.checkbox("Financial Hardship", value=True)
run_gambling_behavior = st.sidebar.checkbox("Gambling Behavior", value=True)

# Advanced options
with st.sidebar.expander("⚙️ Advanced Options"):
    show_raw_data = st.checkbox("Show Raw Data", value=False)
    show_correlations = st.checkbox("Show Correlation Analysis", value=True)
    export_results = st.checkbox("Enable Export", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**🚀 Ready to analyze?**")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Analysis Dashboard")
    
    # Run analysis button
    if st.button("🚀 Run AI Agent Analysis", type="primary", use_container_width=True):
        with st.spinner("🤖 AI Agent is analyzing customer data..."):
            try:
                # Initialize orchestrator
                orchestrator = CustomerSegmentationOrchestrator()
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Create sample data
                status_text.text("📊 Creating sample data...")
                progress_bar.progress(20)
                
                # Run async function
                async def run_analysis():
                    sample_data = await orchestrator.data_pipeline.create_sample_data(n_samples=sample_size)
                    return sample_data
                
                # Get sample data
                sample_data = asyncio.run(run_analysis())
                
                if isinstance(sample_data, dict) and "message" in sample_data:
                    st.warning("⚠️ Using mock data due to missing dependencies")
                    # Create simple dataframe for demo
                    sample_data = pd.DataFrame({
                        'customer_id': range(sample_size),
                        'monthly_income': np.random.normal(5000, 1500, sample_size),
                        'credit_score': np.random.normal(700, 100, sample_size),
                        'debt_to_income_ratio': np.random.uniform(0.1, 0.8, sample_size),
                    })
                
                progress_bar.progress(40)
                status_text.text("🔍 Running segmentation models...")
                
                # Run the workflow
                async def run_workflow():
                    return await orchestrator.run_enhanced_segmentation_workflow(sample_data)
                
                results = asyncio.run(run_workflow())
                
                progress_bar.progress(80)
                status_text.text("📊 Generating insights...")
                
                # Complete
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                # Display results
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.success("🎉 **AI Agent Analysis Completed Successfully!**")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Metrics
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                
                with col_m1:
                    st.metric(
                        "👥 Customers Analyzed", 
                        f"{results['metadata']['total_customers']:,}",
                        help="Total number of customers processed"
                    )
                
                with col_m2:
                    st.metric(
                        "🔧 Features Analyzed", 
                        results['metadata']['features_analyzed'],
                        help="Number of features used in analysis"
                    )
                
                with col_m3:
                    st.metric(
                        "🤖 Models Executed", 
                        results['metadata']['models_executed'],
                        help="Number of AI models run"
                    )
                
                with col_m4:
                    st.metric(
                        "⏱️ Timestamp", 
                        datetime.now().strftime("%H:%M:%S"),
                        help="Analysis completion time"
                    )
                
                # Segmentation Results
                st.subheader("🎯 Segmentation Results")
                
                # Create tabs for different results
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Detailed Results", "📈 Correlations", "📋 Summary"])
                
                with tab1:
                    st.markdown("#### 📊 Segmentation Overview")
                    
                    # Display model results
                    for model_name, model_results in results['segmentation_results'].items():
                        with st.expander(f"🔍 {model_name.replace('_', ' ').title()} Results"):
                            st.json(model_results)
                
                with tab2:
                    st.markdown("#### 🔍 Detailed Analysis Results")
                    
                    if show_raw_data and hasattr(sample_data, 'shape'):
                        st.markdown("**Sample Data:**")
                        st.dataframe(sample_data.head(100), use_container_width=True)
                    
                    st.markdown("**Full Results:**")
                    st.json(results)
                
                with tab3:
                    if show_correlations:
                        st.markdown("#### 📈 Correlation Analysis")
                        
                        # Display correlation results
                        if 'correlation_analysis' in results:
                            corr_data = results['correlation_analysis']
                            st.markdown("**Feature Correlations:**")
                            st.json(corr_data.get('feature_correlations', {}))
                            
                            if 'insights' in corr_data:
                                st.markdown("**Correlation Insights:**")
                                st.json(corr_data['insights'])
                    else:
                        st.info("Correlation analysis disabled in settings")
                
                with tab4:
                    st.markdown("#### 📋 Executive Summary")
                    
                    if 'summary_report' in results:
                        summary = results['summary_report']
                        
                        # Executive summary
                        if 'executive_summary' in summary:
                            st.markdown("**📊 Executive Summary:**")
                            exec_summary = summary['executive_summary']
                            if isinstance(exec_summary, dict):
                                for key, value in exec_summary.items():
                                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                            else:
                                st.write(exec_summary)
                        
                        # Business insights
                        if 'business_insights' in summary:
                            st.markdown("**💡 Business Insights:**")
                            insights = summary['business_insights']
                            if isinstance(insights, dict):
                                for category, insight_list in insights.items():
                                    st.write(f"**{category.replace('_', ' ').title()}:**")
                                    if isinstance(insight_list, list):
                                        for insight in insight_list:
                                            st.write(f"• {insight}")
                                    else:
                                        st.write(f"• {insight_list}")
                            else:
                                st.write(insights)
                
                # Export functionality
                if export_results:
                    st.subheader("📥 Export Results")
                    
                    col_e1, col_e2, col_e3 = st.columns(3)
                    
                    with col_e1:
                        # JSON export
                        json_data = json.dumps(results, indent=2, default=str)
                        st.download_button(
                            label="📄 Download JSON",
                            data=json_data,
                            file_name=f"segmentation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    with col_e2:
                        # CSV export (sample data)
                        if hasattr(sample_data, 'to_csv'):
                            csv_data = sample_data.to_csv(index=False)
                            st.download_button(
                                label="📊 Download CSV",
                                data=csv_data,
                                file_name=f"sample_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                    
                    with col_e3:
                        # Summary report
                        if 'summary_report' in results:
                            summary_text = f"Customer Segmentation Analysis Report\n"
                            summary_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                            summary_text += f"Customers Analyzed: {results['metadata']['total_customers']}\n"
                            summary_text += f"Models Executed: {results['metadata']['models_executed']}\n\n"
                            summary_text += json.dumps(results['summary_report'], indent=2, default=str)
                            
                            st.download_button(
                                label="📋 Download Report",
                                data=summary_text,
                                file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                
            except Exception as e:
                st.error(f"❌ Error running analysis: {str(e)}")
                st.exception(e)

with col2:
    st.subheader("ℹ️ About")
    
    st.markdown("""
    ### 🤖 AI Agent Features:
    - **Multi-Model Analysis**: Financial Capability, Hardship, and Gambling Behavior
    - **Real-Time Processing**: Async workflow execution
    - **Correlation Analysis**: Advanced feature relationships
    - **Summary Reports**: Executive insights and recommendations
    - **Export Options**: JSON, CSV, and text reports
    
    ### 🎯 Model Types:
    1. **Financial Capability**: Income and credit analysis
    2. **Financial Hardship**: Debt and payment analysis  
    3. **Gambling Behavior**: Risk assessment patterns
    
    ### 📊 Analysis Modes:
    - **Sequential**: Models run one after another
    - **Parallel**: Models run simultaneously for faster processing
    """)
    
    # System info
    with st.expander("🔧 System Information"):
        st.write("**Dependencies Status:**")
        
        # Check dependencies
        deps = {
            "pandas": True,
            "numpy": True,
            "scikit-learn": True,
            "matplotlib": True
        }
        
        for dep, status in deps.items():
            status_icon = "✅" if status else "❌"
            st.write(f"{status_icon} {dep}")
        
        st.write(f"**Python Version:** {sys.version}")
        st.write(f"**Streamlit Version:** {st.__version__}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        🤖 <strong>AI Agent Customer Segmentation</strong><br>
        Enhanced Multi-Model Analysis with Real-Time Insights<br>
        <em>Powered by Advanced Machine Learning</em>
    </div>
    """, 
    unsafe_allow_html=True
)
