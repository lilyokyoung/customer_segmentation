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
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Agent Customer Segmentation</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # Data Source Selection
    st.sidebar.subheader("📁 Data Source")
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["🎲 Generate Sample Data", "📂 Upload Your Dataset"],
        help="Select whether to use generated sample data or upload your own CSV file"
    )
    
    # Upload dataset option
    uploaded_file = None
    if data_source == "📂 Upload Your Dataset":
        st.sidebar.markdown("**Upload CSV File:**")
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with customer data. Should include columns like customer_id, monthly_income, credit_score, etc."
        )
        
        if uploaded_file is not None:
            # Show file info
            st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
            st.sidebar.info(f"📊 File size: {uploaded_file.size} bytes")
    
    # Sample data configuration (only show if generating sample data)
    if data_source == "🎲 Generate Sample Data":
        st.sidebar.subheader("📊 Sample Data Configuration")
        sample_size = st.sidebar.slider("Number of customers", min_value=100, max_value=5000, value=1000, step=100)
    else:
        sample_size = 1000  # Default value when uploading data
    
    # Analysis mode
    st.sidebar.subheader("🤖 Analysis Mode")
    orchestration_mode = st.sidebar.selectbox(
        "Orchestration Mode",
        ["sequential", "parallel"],
        help="Sequential runs models one by one, parallel runs them simultaneously"
    )
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = None
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🚀 Run AI Agent Analysis")
        
        # Show data format requirements if upload mode is selected
        if data_source == "📂 Upload Your Dataset":
            with st.expander("📋 Data Format Requirements", expanded=True):
                st.markdown("""
                **Required/Recommended Columns for Best Results:**
                
                **Financial Capability Analysis:**
                - `monthly_income` - Customer monthly income
                - `credit_score` - Credit score (300-850 range)
                - `customer_id` - Unique customer identifier
                
                **Financial Hardship Analysis:**
                - `debt_to_income_ratio` - Ratio of debt to income
                - `payment_delays_count` - Number of payment delays
                
                **Gambling Behavior Analysis:**
                - `gambling_merchant_frequency` - Frequency of gambling transactions
                - `large_cash_withdrawals` - Number of large cash withdrawals
                
                **📌 Notes:**
                - CSV format with headers in the first row
                - Numeric columns should contain numeric values
                - Missing columns will be handled gracefully by the system
                - The system will adapt to your available data columns
                """)
        
        if st.button("🤖 Start Customer Segmentation Analysis", type="primary", use_container_width=True):
            with st.spinner("Initializing AI Agent Orchestrator..."):
                try:
                    # Initialize orchestrator
                    config = {
                        'agent': {'name': 'CustomerSegmentationAgent', 'orchestration_mode': orchestration_mode},
                        'data': {'batch_size': sample_size, 'validation_split': 0.2},
                        'models': {
                            'financial_capability': {'features': ['monthly_income', 'credit_score']},
                            'financial_hardship': {'features': ['debt_to_income_ratio', 'payment_delays_count']},
                            'gambling_behavior': {'features': ['gambling_merchant_frequency', 'large_cash_withdrawals']}
                        },
                        'logging': {'level': 'INFO', 'file': 'logs/customer_segmentation.log'}
                    }
                    
                    # Create orchestrator with custom config
                    orchestrator = CustomerSegmentationOrchestrator()
                    orchestrator.config = config
                    st.session_state.orchestrator = orchestrator
                    
                    st.success("✅ Orchestrator initialized successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error initializing orchestrator: {str(e)}")
                    return
            
            # Load or create sample data
            if data_source == "📂 Upload Your Dataset" and uploaded_file is not None:
                # Handle uploaded file
                with st.spinner("📂 Loading uploaded dataset..."):
                    try:
                        # Read the uploaded CSV file
                        sample_data = pd.read_csv(uploaded_file)
                        
                        # Display dataset info
                        st.success(f"✅ Dataset loaded successfully!")
                        st.info(f"📊 Dataset shape: {sample_data.shape[0]} rows × {sample_data.shape[1]} columns")
                        
                        # Show column info
                        with st.expander("📋 Dataset Preview", expanded=False):
                            st.write("**First 5 rows:**")
                            st.dataframe(sample_data.head())
                            st.write("**Column Information:**")
                            col_info = pd.DataFrame({
                                'Column': sample_data.columns,
                                'Data Type': sample_data.dtypes,
                                'Non-Null Count': sample_data.count(),
                                'Null Count': sample_data.isnull().sum()
                            })
                            st.dataframe(col_info)
                        
                        # Validate required columns
                        required_columns = ['customer_id']
                        missing_columns = [col for col in required_columns if col not in sample_data.columns]
                        
                        if missing_columns:
                            st.warning(f"⚠️ Missing recommended columns: {missing_columns}")
                            st.info("💡 The analysis will work with your current columns, but having 'customer_id' is recommended for better results.")
                        
                        # Add customer_id if missing
                        if 'customer_id' not in sample_data.columns:
                            sample_data['customer_id'] = range(len(sample_data))
                            st.info("✅ Added 'customer_id' column automatically")
                        
                    except Exception as e:
                        st.error(f"❌ Error loading uploaded file: {str(e)}")
                        st.error("Please make sure your file is a valid CSV with proper formatting.")
                        return
            
            elif data_source == "📂 Upload Your Dataset" and uploaded_file is None:
                st.warning("📂 Please upload a CSV file to proceed with analysis")
                st.info("💡 You can also switch to 'Generate Sample Data' mode to test the system")
                return
            
            else:
                # Generate sample data
                with st.spinner(f"Creating sample data for {sample_size} customers..."):
                    try:
                        # Use asyncio to run the async function
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        sample_data = loop.run_until_complete(
                            st.session_state.orchestrator.data_pipeline.create_sample_data(n_samples=sample_size)
                        )
                        
                        loop.close()
                        
                        if isinstance(sample_data, dict) and "message" in sample_data:
                            st.warning("⚠️ Using mock data due to missing dependencies")
                            # Create a minimal dataframe for demo
                            sample_data = pd.DataFrame({
                                'customer_id': range(sample_size),
                                'monthly_income': np.random.normal(5000, 1500, sample_size),
                                'credit_score': np.random.normal(700, 100, sample_size),
                                'debt_to_income_ratio': np.random.uniform(0.1, 0.8, sample_size),
                            })
                        
                        st.success(f"✅ Created sample data with {len(sample_data)} customers")
                        
                    except Exception as e:
                        st.error(f"❌ Error creating sample data: {str(e)}")
                        return
            
            # Run analysis
            with st.spinner("Running enhanced segmentation workflow..."):
                try:
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("🔄 Step 1/5: Preprocessing data...")
                    progress_bar.progress(20)
                    
                    status_text.text("🔄 Step 2/5: Analyzing correlations...")
                    progress_bar.progress(40)
                    
                    status_text.text("🔄 Step 3/5: Running segmentation models...")
                    progress_bar.progress(60)
                    
                    status_text.text("🔄 Step 4/5: Cross-model analysis...")
                    progress_bar.progress(80)
                    
                    status_text.text("🔄 Step 5/5: Generating reports...")
                    
                    # Run the actual workflow with async
                    if pd and hasattr(sample_data, 'shape'):
                        # Create new event loop for the async workflow
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        results = loop.run_until_complete(
                            st.session_state.orchestrator.run_enhanced_segmentation_workflow(sample_data)
                        )
                        
                        loop.close()
                        
                        st.session_state.analysis_results = results
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Analysis completed successfully!")
                        
                        # Success message
                        st.markdown("""
                        <div class="success-box">
                            <h3>🎉 Analysis Completed Successfully!</h3>
                            <p>Your customer segmentation analysis has been completed. Check the results below.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    else:
                        st.error("⚠️ Cannot run workflow - pandas not available or invalid data format")
                        progress_bar.progress(0)
                        status_text.text("❌ Analysis failed")
                        return
                        
                except Exception as e:
                    st.error(f"❌ Error in analysis workflow: {str(e)}")
                    try:
                        progress_bar.progress(0)
                        status_text.text("❌ Analysis failed")
                    except:
                        pass  # In case progress_bar or status_text are not defined
                    return
    
    with col2:
        st.subheader("ℹ️ System Information")
        
        # System info
        st.markdown(f"""
        **🔧 Current Configuration:**
        - Data Source: {data_source}
        - Python Version: Available
        - Pandas: Available
        - NumPy: Available
        - Async Support: Enabled
        
        **📊 Data Input Options:**
        - 🎲 Generate synthetic sample data
        - 📂 Upload your own CSV files
        - 🔍 Automatic data validation
        - 📋 Column mapping assistance
        
        **🤖 Analysis Features:**
        - Financial Capability Segmentation
        - Financial Hardship Assessment  
        - Gambling Behavior Analysis
        - Cross-Model Correlation Analysis
        - Comprehensive Reporting
        
        **📁 Supported File Formats:**
        - CSV files (.csv)
        - UTF-8 encoding recommended
        - Headers in first row
        """)
        
        # Show upload status if in upload mode
        if data_source == "📂 Upload Your Dataset":
            if uploaded_file is not None:
                st.success("✅ File Ready for Analysis")
            else:
                st.warning("⏳ Awaiting File Upload")
    
    # Display results if available
    if st.session_state.analysis_results:
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔍 Detailed Results", "🔗 Correlations", "📝 Summary"])
        
        with tab1:
            st.subheader("Analysis Overview")
            
            # Key metrics
            results = st.session_state.analysis_results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "👥 Customers Analyzed", 
                    results['metadata']['total_customers']
                )
            
            with col2:
                st.metric(
                    "🔧 Features Analyzed", 
                    results['metadata']['features_analyzed']
                )
            
            with col3:
                st.metric(
                    "🤖 Models Executed", 
                    results['metadata']['models_executed']
                )
            
            with col4:
                st.metric(
                    "⏰ Analysis Time", 
                    "Real-time"
                )
            
            # Segmentation results overview
            if 'segmentation_results' in results:
                st.subheader("🎯 Segmentation Results")
                seg_results = results['segmentation_results']
                
                for model_name, model_results in seg_results.items():
                    with st.expander(f"📊 {model_name.replace('_', ' ').title()} Results"):
                        if isinstance(model_results, dict) and 'message' not in model_results:
                            st.json(model_results)
                        else:
                            st.write(model_results)
        
        with tab2:
            st.subheader("Detailed Analysis Results")
            
            # Full results in expandable sections
            with st.expander("🔍 Complete Segmentation Results", expanded=False):
                st.json(results['segmentation_results'])
            
            with st.expander("📊 Correlation Analysis", expanded=False):
                if 'correlation_analysis' in results:
                    st.json(results['correlation_analysis'])
                else:
                    st.write("Correlation analysis not available")
            
            with st.expander("🔗 Cross-Model Analysis", expanded=False):
                if 'cross_model_analysis' in results:
                    st.json(results['cross_model_analysis'])
                else:
                    st.write("Cross-model analysis not available")
        
        with tab3:
            st.subheader("Correlation Insights")
            
            if 'correlation_analysis' in results:
                corr_data = results['correlation_analysis']
                
                # Feature correlations
                if 'insights' in corr_data:
                    st.write("**🔍 Key Correlation Insights:**")
                    insights = corr_data['insights']
                    if isinstance(insights, dict):
                        for category, insight_list in insights.items():
                            st.write(f"**{category.replace('_', ' ').title()}:**")
                            if isinstance(insight_list, list):
                                for insight in insight_list:
                                    st.write(f"- {insight}")
                            else:
                                st.write(f"- {insight_list}")
                    else:
                        st.write(insights)
                
                # Model feature correlations
                if 'model_feature_correlations' in corr_data:
                    st.write("**🤖 Model-Specific Correlations:**")
                    for model, corr_info in corr_data['model_feature_correlations'].items():
                        with st.expander(f"📊 {model.replace('_', ' ').title()} Correlations"):
                            st.json(corr_info)
            else:
                st.write("Correlation analysis not available in current results")
        
        with tab4:
            st.subheader("Executive Summary")
            
            if 'summary_report' in results:
                summary = results['summary_report']
                
                # Executive summary
                if 'executive_summary' in summary:
                    st.write("**📋 Executive Summary:**")
                    exec_summary = summary['executive_summary']
                    if isinstance(exec_summary, dict):
                        for key, value in exec_summary.items():
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                    else:
                        st.write(exec_summary)
                
                # Business insights
                if 'business_insights' in summary:
                    st.write("**💡 Business Insights:**")
                    insights = summary['business_insights']
                    if isinstance(insights, dict):
                        for category, insight_list in insights.items():
                            st.write(f"**{category.replace('_', ' ').title()}:**")
                            if isinstance(insight_list, list):
                                for insight in insight_list:
                                    st.write(f"- {insight}")
                            else:
                                st.write(f"- {insight_list}")
                    else:
                        st.write(insights)
                
                # Risk assessment
                if 'risk_assessment' in summary:
                    st.write("**⚠️ Risk Assessment:**")
                    risk_data = summary['risk_assessment']
                    if isinstance(risk_data, dict):
                        for risk_type, risk_info in risk_data.items():
                            st.write(f"**{risk_type.replace('_', ' ').title()}:** {risk_info}")
                    else:
                        st.write(risk_data)
            else:
                st.write("Summary report not available in current results")
        
        # Export functionality
        st.markdown("---")
        st.subheader("📥 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export as JSON", use_container_width=True):
                json_str = json.dumps(st.session_state.analysis_results, indent=2, default=str)
                st.download_button(
                    label="⬇️ Download JSON",
                    data=json_str,
                    file_name=f"segmentation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Export Summary as Text", use_container_width=True):
                # Create text summary
                text_summary = f"""
CUSTOMER SEGMENTATION ANALYSIS SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
=====================================

OVERVIEW:
- Customers Analyzed: {results['metadata']['total_customers']}
- Features Analyzed: {results['metadata']['features_analyzed']}
- Models Executed: {results['metadata']['models_executed']}

RESULTS:
{json.dumps(results, indent=2, default=str)}
"""
                st.download_button(
                    label="⬇️ Download Text Report",
                    data=text_summary,
                    file_name=f"segmentation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        with col3:
            if st.button("📈 Export as CSV", use_container_width=True):
                # Create CSV data from metadata and summary
                csv_data = {
                    'Metric': ['Total Customers', 'Features Analyzed', 'Models Executed', 'Analysis Timestamp'],
                    'Value': [
                        results['metadata']['total_customers'],
                        results['metadata']['features_analyzed'], 
                        results['metadata']['models_executed'],
                        results['metadata']['timestamp']
                    ]
                }
                df_export = pd.DataFrame(csv_data)
                csv_string = df_export.to_csv(index=False)
                
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_string,
                    file_name=f"segmentation_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# Async wrapper for Streamlit
if __name__ == "__main__":
    # Run the Streamlit app
    main()
