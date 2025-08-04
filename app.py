"""
Streamlit Web Interface for Customer Segmentation AI Agent
Enhanced with interactive dashboard and real-time analysis
"""
import streamlit as st
import asyncio
import json
from datetime import datetime
import sys
import os

# Import data science libraries
import pandas as pd
import numpy as np

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

def detect_column_mapping(data):
    """
    Intelligently detect column mappings for customer segmentation analysis
    """
    columns = data.columns.tolist()
    column_mapping = {
        'suggested_mappings': {},
        'confidence_scores': {}
    }
    
    # Define keywords for different types of analysis
    mapping_keywords = {
        'financial_capability': {
            'monthly_income': ['income', 'salary', 'monthly_income', 'gross_income', 'net_income', 'earnings', 'revenue', 'wage'],
            'credit_score': ['credit_score', 'credit', 'score', 'credit_rating', 'fico', 'creditworthiness', 'rating'],
            'savings': ['savings', 'deposits', 'balance', 'investment', 'assets', 'wealth'],
            'employment_status': ['employment', 'job', 'work', 'occupation', 'employed', 'status'],
            'financial_literacy': ['literacy', 'education', 'knowledge', 'understanding', 'awareness']
        },
        'financial_hardship': {
            'debt_to_income_ratio': ['debt_to_income', 'debt_ratio', 'dti', 'debt_income_ratio', 'leverage', 'debt'],
            'payment_delays': ['payment_delays', 'late_payments', 'delays', 'delinquency', 'overdue', 'arrears'],
            'financial_stress': ['stress', 'difficulty', 'hardship', 'struggle', 'pressure', 'burden'],
            'missed_payments': ['missed', 'default', 'failed', 'unpaid', 'outstanding'],
            'emergency_fund': ['emergency', 'fund', 'reserve', 'contingency', 'backup']
        },
        'gambling_behavior': {
            'gambling_frequency': ['gambling', 'casino', 'betting', 'gaming', 'lottery', 'wager', 'poker', 'slots'],
            'gambling_spend': ['spend', 'amount', 'expenditure', 'loss', 'stake', 'bet_amount'],
            'gambling_venues': ['venue', 'location', 'casino', 'online', 'app', 'platform'],
            'gambling_pattern': ['pattern', 'behavior', 'habit', 'frequency', 'regular', 'occasional'],
            'problem_gambling': ['problem', 'addiction', 'compulsive', 'excessive', 'control']
        },
        'demographics': {
            'age': ['age', 'birth', 'born', 'year', 'old', 'generation'],
            'gender': ['gender', 'sex', 'male', 'female', 'm', 'f'],
            'location': ['location', 'city', 'state', 'region', 'area', 'postcode', 'zip', 'address'],
            'education': ['education', 'degree', 'qualification', 'school', 'university', 'college'],
            'marital_status': ['marital', 'married', 'single', 'divorced', 'status', 'relationship'],
            'household_size': ['household', 'family', 'dependents', 'children', 'size', 'members']
        },
        'general': {
            'customer_id': ['customer_id', 'id', 'customer', 'client_id', 'account_id', 'user_id', 'person_id', 
                           'cust_id', 'customerid', 'clientid', 'userid', 'personid', 'accountid', 'member_id', 
                           'memberid', 'unique_id', 'uniqueid', 'identifier', 'key', 'index', 'number', 'no']
        }
    }
    
    # Score columns based on keyword matching
    for analysis_type, features in mapping_keywords.items():
        column_mapping['suggested_mappings'][analysis_type] = {}
        
        for feature, keywords in features.items():
            best_match = None
            best_score = 0
            
            for col in columns:
                col_lower = col.lower().replace('_', ' ').replace('-', ' ')
                score = 0
                
                # Exact match gets highest score
                if col_lower in [kw.replace('_', ' ') for kw in keywords]:
                    score = 1.0
                else:
                    # Partial match scoring
                    for keyword in keywords:
                        keyword_lower = keyword.replace('_', ' ')
                        if keyword_lower in col_lower:
                            score = max(score, 0.8)
                        elif any(word in col_lower for word in keyword_lower.split()):
                            score = max(score, 0.6)
                
                if score > best_score:
                    best_score = score
                    best_match = col
            
            if best_match and best_score > 0.5:
                column_mapping['suggested_mappings'][analysis_type][feature] = best_match
                column_mapping['confidence_scores'][f"{analysis_type}_{feature}"] = best_score
    
    return column_mapping

def apply_column_mapping(data, auto_mapping, manual_mapping):
    """
    Apply column mapping to prepare data for analysis
    """
    mapped_data = data.copy()
    mapping_applied = {}
    
    # Priority: manual mapping > auto mapping
    all_mappings = {}
    
    # Add auto mappings
    for analysis_type, features in auto_mapping['suggested_mappings'].items():
        for feature, column in features.items():
            all_mappings[feature] = column
    
    # Override with manual mappings
    all_mappings.update(manual_mapping)
    
    # Rename columns according to mapping
    rename_dict = {}
    for target_name, source_column in all_mappings.items():
        if source_column in mapped_data.columns:
            rename_dict[source_column] = target_name
            mapping_applied[target_name] = source_column
    
    if rename_dict:
        mapped_data = mapped_data.rename(columns=rename_dict)
    
    return mapped_data, mapping_applied

def create_advanced_segments(data, dynamics_selection, column_mapping):
    """
    Create segments for selected dynamics individually
    """
    segments_results = {}
    dummy_variables = {}
    
    # Map columns to dynamics
    dynamic_columns = {}
    for dynamic, selected in dynamics_selection.items():
        if selected and dynamic in column_mapping['suggested_mappings']:
            dynamic_columns[dynamic] = []
            for feature, column in column_mapping['suggested_mappings'][dynamic].items():
                if column in data.columns:
                    dynamic_columns[dynamic].append(column)
    
    # Create segments for each dynamic
    for dynamic, columns in dynamic_columns.items():
        if len(columns) > 0:
            # Simple clustering-based segmentation
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            
            try:
                # Prepare data for this dynamic
                dynamic_data = data[columns].fillna(data[columns].median())
                
                # Standardize the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(dynamic_data)
                
                # Perform clustering (3 segments: Low, Medium, High)
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                segments = kmeans.fit_predict(scaled_data)
                
                # Create meaningful segment labels
                segment_means = []
                for i in range(3):
                    mask = segments == i
                    if mask.sum() > 0:
                        segment_mean = dynamic_data[mask].mean().mean()
                        segment_means.append((i, segment_mean))
                
                # Sort by mean values and assign labels
                segment_means.sort(key=lambda x: x[1])
                segment_labels = {
                    segment_means[0][0]: f"{dynamic}_low",
                    segment_means[1][0]: f"{dynamic}_medium", 
                    segment_means[2][0]: f"{dynamic}_high"
                }
                
                # Map segments to labels
                labeled_segments = [segment_labels[seg] for seg in segments]
                
                segments_results[dynamic] = {
                    'segments': labeled_segments,
                    'features_used': columns,
                    'segment_counts': {label: labeled_segments.count(label) for label in set(labeled_segments)}
                }
                
                # Create dummy variables for regression
                for label in set(labeled_segments):
                    dummy_col_name = f"segment_{label}"
                    dummy_variables[dummy_col_name] = [1 if seg == label else 0 for seg in labeled_segments]
                
            except Exception as e:
                segments_results[dynamic] = {'error': str(e)}
    
    return segments_results, dummy_variables

def perform_regression_analysis(data, segments_results, dummy_variables, regression_approach, dynamics_selection):
    """
    Perform regression analysis with segment dummy variables
    """
    regression_results = {}
    
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error
        import numpy as np
        
        # Create enhanced dataset with dummy variables
        enhanced_data = data.copy()
        for dummy_name, dummy_values in dummy_variables.items():
            enhanced_data[dummy_name] = dummy_values
        
        # Automatic variable selection logic
        if regression_approach == "Automatic Selection":
            # Logical selection based on financial theory
            
            # Potential dependent variables (outcomes we want to predict)
            potential_dependent = []
            if dynamics_selection.get('financial_hardship'):
                potential_dependent.extend([col for col in enhanced_data.columns 
                                          if any(keyword in col.lower() for keyword in 
                                          ['debt', 'stress', 'hardship', 'delay', 'missed'])])
            
            if dynamics_selection.get('gambling_behavior'):
                potential_dependent.extend([col for col in enhanced_data.columns 
                                          if any(keyword in col.lower() for keyword in 
                                          ['gambling', 'spend', 'loss', 'problem'])])
            
            # Independent variables (predictors)
            potential_independent = []
            if dynamics_selection.get('financial_capability'):
                potential_independent.extend([col for col in enhanced_data.columns 
                                            if any(keyword in col.lower() for keyword in 
                                            ['income', 'credit', 'savings', 'employment'])])
            
            if dynamics_selection.get('demographics'):
                potential_independent.extend([col for col in enhanced_data.columns 
                                            if any(keyword in col.lower() for keyword in 
                                            ['age', 'gender', 'education', 'location'])])
            
            # Add segment dummy variables as independent variables
            potential_independent.extend(list(dummy_variables.keys()))
            
            # Run multiple regression models
            for dep_var in potential_dependent:
                if dep_var in enhanced_data.columns:
                    # Select independent variables (exclude the dependent variable)
                    indep_vars = [var for var in potential_independent 
                                 if var in enhanced_data.columns and var != dep_var]
                    
                    if len(indep_vars) > 0:
                        try:
                            # Prepare data
                            y = enhanced_data[dep_var].fillna(enhanced_data[dep_var].median())
                            X = enhanced_data[indep_vars].fillna(enhanced_data[indep_vars].median())
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.3, random_state=42
                            )
                            
                            # Fit model
                            model = LinearRegression()
                            model.fit(X_train, y_train)
                            
                            # Predictions
                            y_pred = model.predict(X_test)
                            
                            # Metrics
                            r2 = r2_score(y_test, y_pred)
                            mse = mean_squared_error(y_test, y_pred)
                            
                            # Feature importance (coefficients)
                            feature_importance = dict(zip(indep_vars, model.coef_))
                            
                            regression_results[dep_var] = {
                                'r_squared': r2,
                                'mse': mse,
                                'feature_importance': feature_importance,
                                'independent_variables': indep_vars,
                                'n_observations': len(y_test),
                                'model_intercept': model.intercept_
                            }
                            
                        except Exception as e:
                            regression_results[dep_var] = {'error': str(e)}
        
        # Add explanation of variable selection logic
        regression_results['variable_selection_logic'] = {
            'dependent_variables_rationale': 
                "Selected variables representing outcomes of interest (financial stress, gambling problems) "
                "that could be influenced by other factors.",
            'independent_variables_rationale':
                "Selected financial capability indicators, demographics, and segment membership "
                "as potential predictors of financial outcomes.",
            'segment_variables_usage':
                "Segment dummy variables allow us to understand how belonging to different "
                "customer segments affects the dependent variables."
        }
        
    except Exception as e:
        regression_results = {'error': str(e)}
    
    return regression_results

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
        st.sidebar.markdown("**Upload Data File:**")
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a CSV or Excel file with customer data. Should include columns like customer_id, monthly_income, credit_score, etc."
        )
        
        if uploaded_file is not None:
            # Show file info
            st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
            st.sidebar.info(f"📊 File size: {uploaded_file.size} bytes")
            
            # Advanced Analysis Configuration
            st.sidebar.markdown("---")
            st.sidebar.subheader("🔬 Advanced Analysis Configuration")
            
            # Dynamic Selection
            st.sidebar.markdown("**Select Dynamics for Analysis:**")
            st.sidebar.info("💡 Tip: Uncheck dynamics you don't want to analyze. Only selected dynamics will be processed.")
            dynamics_selection = {
                'financial_capability': st.sidebar.checkbox("💰 Financial Capability", value=False),
                'financial_hardship': st.sidebar.checkbox("⚠️ Financial Hardship", value=False),
                'gambling_behavior': st.sidebar.checkbox("🎰 Gambling Behaviour", value=False),
                'demographics': st.sidebar.checkbox("👥 Demographics", value=True)
            }
            
            # Analysis Strategy
            st.sidebar.markdown("**Analysis Strategy:**")
            analysis_strategy = st.sidebar.radio(
                "Segmentation Approach:",
                ["Individual Dynamics", "Combined Dynamics", "Sequential Analysis"],
                help="Individual: Run segmentation separately for each dynamic\nCombined: Use all dynamics together\nSequential: Run individual then combined"
            )
            
            # Regression Analysis Setup
            st.sidebar.markdown("**Regression Analysis Setup:**")
            enable_regression = st.sidebar.checkbox("🔍 Enable Regression Analysis", value=True)
            
            if enable_regression:
                regression_approach = st.sidebar.selectbox(
                    "Regression Approach:",
                    ["Automatic Selection", "Manual Variable Selection", "Stepwise Selection"],
                    help="Choose how to select dependent and independent variables"
                )
                
                # Store configuration in session state
                if any(dynamics_selection.values()):
                    st.session_state.analysis_config = {
                        'dynamics_selection': dynamics_selection,
                        'analysis_strategy': analysis_strategy,
                        'enable_regression': enable_regression,
                        'regression_approach': regression_approach
                    }
                else:
                    st.sidebar.warning("⚠️ Please select at least one dynamic for analysis")
    
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
                - **Supported formats:** CSV (.csv) and Excel (.xlsx, .xls)
                - Headers should be in the first row
                - For Excel files, data should be in the first sheet
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
                        # Determine file type and read accordingly
                        file_extension = uploaded_file.name.split('.')[-1].lower()
                        
                        if file_extension == 'csv':
                            sample_data = pd.read_csv(uploaded_file)
                        elif file_extension in ['xlsx', 'xls']:
                            # For Excel files, try to read the first sheet
                            try:
                                sample_data = pd.read_excel(uploaded_file, sheet_name=0)
                            except ImportError:
                                st.error("❌ Excel file support requires the 'openpyxl' library. Please use CSV format or contact support.")
                                return
                            except Exception as e:
                                st.error(f"❌ Error reading Excel file: {str(e)}")
                                st.info("💡 Try saving your Excel file as CSV format or install openpyxl: pip install openpyxl")
                                return
                        else:
                            st.error(f"❌ Unsupported file format: {file_extension}")
                            st.info("💡 Please upload a CSV (.csv) or Excel (.xlsx, .xls) file.")
                            return
                        
                        # Display dataset info
                        st.success(f"✅ Dataset loaded successfully! ({file_extension.upper()} format)")
                        st.info(f"📊 Dataset shape: {sample_data.shape[0]} rows × {sample_data.shape[1]} columns")
                        
                        # Intelligent column mapping
                        column_mapping = detect_column_mapping(sample_data)
                        
                        # Show column info and mapping suggestions
                        with st.expander("📋 Dataset Preview & Column Mapping", expanded=True):
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
                            
                            # Show intelligent mapping suggestions
                            st.write("**🤖 Intelligent Column Mapping Suggestions:**")
                            if column_mapping['suggested_mappings']:
                                for analysis_type, suggestions in column_mapping['suggested_mappings'].items():
                                    if suggestions:
                                        st.write(f"**{analysis_type.replace('_', ' ').title()}:**")
                                        for feature, column in suggestions.items():
                                            confidence = column_mapping['confidence_scores'].get(f"{analysis_type}_{feature}", 0)
                                            confidence_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"
                                            st.write(f"  {confidence_emoji} {feature}: `{column}` (confidence: {confidence:.1%})")
                            else:
                                st.info("💡 No automatic column mapping detected. The system will use available columns adaptively.")
                            
                            # Allow manual column mapping override
                            st.write("**🔧 Manual Column Mapping Override (Optional):**")
                            manual_mapping = {}
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                manual_mapping['customer_id'] = st.selectbox(
                                    "Customer ID Column:",
                                    ["Auto-detect"] + list(sample_data.columns),
                                    help="Select the column that contains unique customer identifiers"
                                )
                                manual_mapping['monthly_income'] = st.selectbox(
                                    "Monthly Income Column:",
                                    ["Auto-detect"] + list(sample_data.columns),
                                    help="Select the column that contains customer income data"
                                )
                                manual_mapping['credit_score'] = st.selectbox(
                                    "Credit Score Column:",
                                    ["Auto-detect"] + list(sample_data.columns),
                                    help="Select the column that contains credit score data"
                                )
                            
                            with col_b:
                                manual_mapping['debt_to_income_ratio'] = st.selectbox(
                                    "Debt-to-Income Ratio:",
                                    ["Auto-detect"] + list(sample_data.columns),
                                    help="Select the column that contains debt ratio data"
                                )
                                manual_mapping['payment_delays'] = st.selectbox(
                                    "Payment Delays:",
                                    ["Auto-detect"] + list(sample_data.columns),
                                    help="Select the column that contains payment delay information"
                                )
                                manual_mapping['transaction_frequency'] = st.selectbox(
                                    "Transaction Frequency:",
                                    ["Auto-detect"] + list(sample_data.columns),
                                    help="Select the column that contains transaction frequency data"
                                )
                            
                            # Store mapping in session state
                            st.session_state.column_mapping = {
                                'auto': column_mapping,
                                'manual': {k: v for k, v in manual_mapping.items() if v != "Auto-detect"}
                            }
                            st.dataframe(col_info)
                        
                        # Check if we can use an existing column as customer_id
                        detected_customer_id = None
                        if column_mapping.get('suggested_mappings', {}).get('general', {}).get('customer_id'):
                            detected_customer_id = column_mapping['suggested_mappings']['general']['customer_id']
                        
                        # Add customer_id if missing (do this first)
                        if 'customer_id' not in sample_data.columns:
                            if detected_customer_id and detected_customer_id in sample_data.columns:
                                # Rename the detected column to customer_id
                                sample_data = sample_data.rename(columns={detected_customer_id: 'customer_id'})
                                st.info(f"✅ Using '{detected_customer_id}' as customer_id column")
                            else:
                                # Create a new customer_id column
                                sample_data['customer_id'] = range(len(sample_data))
                                st.info("✅ Added 'customer_id' column automatically")
                        
                        # Validate required columns (after adding missing ones)
                        required_columns = ['customer_id']
                        missing_columns = [col for col in required_columns if col not in sample_data.columns]
                        
                        if missing_columns:
                            st.warning(f"⚠️ Missing recommended columns: {missing_columns}")
                            st.info("💡 The analysis will work with your current columns, but having these columns is recommended for better results.")
                        else:
                            st.success("✅ All recommended columns are present or have been added automatically")
                        
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
                        # Apply column mapping if uploaded data
                        if data_source == "📂 Upload Your Dataset" and hasattr(st.session_state, 'column_mapping'):
                            status_text.text("🔄 Applying column mapping...")
                            try:
                                mapped_data, mapping_applied = apply_column_mapping(
                                    sample_data, 
                                    st.session_state.column_mapping['auto'],
                                    st.session_state.column_mapping['manual']
                                )
                                sample_data = mapped_data
                                
                                if mapping_applied:
                                    st.info(f"✅ Applied column mapping: {mapping_applied}")
                                else:
                                    st.info("💡 Using original column names - no mapping needed")
                            except Exception as e:
                                st.warning(f"⚠️ Column mapping failed: {e}. Using original columns.")
                        
                        # Advanced Analysis Workflow
                        if (data_source == "📂 Upload Your Dataset" and 
                            hasattr(st.session_state, 'analysis_config') and
                            any(st.session_state.analysis_config['dynamics_selection'].values())):
                            
                            status_text.text("🔄 Step 1/6: Individual Dynamics Segmentation...")
                            progress_bar.progress(16)
                            
                            # Perform advanced segmentation
                            segments_results, dummy_variables = create_advanced_segments(
                                sample_data,
                                st.session_state.analysis_config['dynamics_selection'],
                                st.session_state.column_mapping['auto']
                            )
                            
                            status_text.text("🔄 Step 2/6: Creating Dummy Variables...")
                            progress_bar.progress(33)
                            
                            # Perform regression analysis if enabled
                            regression_results = {}
                            if st.session_state.analysis_config['enable_regression']:
                                status_text.text("🔄 Step 3/6: Regression Analysis...")
                                progress_bar.progress(50)
                                
                                regression_results = perform_regression_analysis(
                                    sample_data,
                                    segments_results,
                                    dummy_variables,
                                    st.session_state.analysis_config['regression_approach'],
                                    st.session_state.analysis_config['dynamics_selection']
                                )
                            
                            status_text.text("🔄 Step 4/6: Generating Insights...")
                            progress_bar.progress(66)
                            
                            # Combine results
                            results = {
                                'advanced_segmentation': segments_results,
                                'dummy_variables': dummy_variables,
                                'regression_analysis': regression_results,
                                'analysis_config': st.session_state.analysis_config,
                                'metadata': {
                                    'timestamp': datetime.now().isoformat(),
                                    'total_customers': len(sample_data),
                                    'features_analyzed': len(sample_data.columns) if hasattr(sample_data, 'columns') else 0,
                                    'dynamics_analyzed': list(st.session_state.analysis_config['dynamics_selection'].keys()),
                                    'analysis_type': 'advanced_multi_dynamic'
                                }
                            }
                            
                            status_text.text("🔄 Step 5/6: Creating Visualizations...")
                            progress_bar.progress(83)
                            
                        else:
                            # Standard workflow for sample data
                            status_text.text("🔄 Step 3/5: Running standard segmentation...")
                            progress_bar.progress(60)
                            
                            # Create new event loop for the async workflow
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            
                            results = loop.run_until_complete(
                                st.session_state.orchestrator.run_enhanced_segmentation_workflow(sample_data)
                            )
                            
                            loop.close()
                        
                        status_text.text("🔄 Step 6/6: Finalizing Results...")
                        progress_bar.progress(100)
                        
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
        - 🆕 Intelligent Column Mapping
        - 🆕 Data-Adaptive Analysis
        
        **📁 Supported File Formats:**
        - CSV files (.csv)
        - Excel files (.xlsx, .xls)
        - UTF-8 encoding recommended
        - Headers in first row
        """)
        
        # Show data-specific recommendations if file is uploaded
        if data_source == "📂 Upload Your Dataset" and uploaded_file is not None:
            file_type = uploaded_file.name.split('.')[-1].upper()
            st.success(f"✅ {file_type} File Ready for Analysis")
            
            # Show analysis recommendations based on uploaded data
            if hasattr(st.session_state, 'column_mapping'):
                mapping = st.session_state.column_mapping['auto']
                recommendations = []
                
                # Check what analysis types are possible
                if mapping['suggested_mappings'].get('financial_capability'):
                    recommendations.append("💰 Financial Capability Analysis")
                if mapping['suggested_mappings'].get('financial_hardship'):
                    recommendations.append("⚠️ Financial Hardship Assessment")
                if mapping['suggested_mappings'].get('gambling_behavior'):
                    recommendations.append("🎰 Gambling Behavior Analysis")
                
                if recommendations:
                    st.info("🎯 **Recommended Analysis Types:**")
                    for rec in recommendations:
                        st.write(f"  • {rec}")
                else:
                    st.info("💡 **General Analysis Available:** The system will adapt to your data structure")
        
        elif data_source == "📂 Upload Your Dataset":
            st.warning("⏳ Awaiting File Upload")
        else:
            st.info("🎲 Sample Data Mode Active")
    
    # Display results if available
    if st.session_state.analysis_results:
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        # Create tabs for different views
        results = st.session_state.analysis_results
        
        # Check if this is advanced analysis
        is_advanced = 'advanced_segmentation' in results
        
        if is_advanced:
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Overview", "� Individual Segments", "📊 Regression Analysis", 
                "🎯 Dummy Variables", "📝 Export"
            ])
        else:
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Overview", "�🔍 Detailed Results", "🔗 Correlations", "📝 Summary"
            ])
        
        with tab1:
            st.subheader("Analysis Overview")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "👥 Customers Analyzed", 
                    results['metadata']['total_customers']
                )
            
            with col2:
                if is_advanced:
                    dynamics_count = sum(1 for selected in results['analysis_config']['dynamics_selection'].values() if selected)
                    st.metric("� Dynamics Analyzed", dynamics_count)
                else:
                    st.metric("🔧 Features Analyzed", results['metadata']['features_analyzed'])
            
            with col3:
                if is_advanced:
                    segments_count = len(results.get('advanced_segmentation', {}))
                    st.metric("🎯 Segments Created", segments_count)
                else:
                    st.metric("🤖 Models Executed", results['metadata']['models_executed'])
            
            with col4:
                if is_advanced and results.get('regression_analysis'):
                    regression_count = len([k for k, v in results['regression_analysis'].items() 
                                          if k != 'variable_selection_logic' and 'error' not in v])
                    st.metric("📈 Regression Models", regression_count)
                else:
                    st.metric("⏰ Analysis Time", "Real-time")
            
            # Display results based on analysis type
            if is_advanced:
                st.subheader("🔬 Advanced Multi-Dynamic Analysis Results")
                
                # Show selected dynamics
                st.write("**Selected Dynamics:**")
                for dynamic, selected in results['analysis_config']['dynamics_selection'].items():
                    if selected:
                        emoji = {"financial_capability": "💰", "financial_hardship": "⚠️", 
                                "gambling_behavior": "🎰", "demographics": "👥"}.get(dynamic, "📊")
                        st.write(f"  {emoji} {dynamic.replace('_', ' ').title()}")
                
                # Segmentation overview
                if results.get('advanced_segmentation'):
                    st.write("**Segmentation Summary:**")
                    for dynamic, seg_data in results['advanced_segmentation'].items():
                        if 'error' not in seg_data:
                            st.write(f"**{dynamic.replace('_', ' ').title()}:**")
                            for segment, count in seg_data['segment_counts'].items():
                                st.write(f"  • {segment}: {count} customers")
                        else:
                            st.error(f"Error in {dynamic}: {seg_data['error']}")
            
            else:
                # Standard segmentation results overview
                if 'segmentation_results' in results:
                    st.subheader("🎯 Segmentation Results")
                    seg_results = results['segmentation_results']
                    
                    for model_name, model_results in seg_results.items():
                        with st.expander(f"📊 {model_name.replace('_', ' ').title()} Results"):
                            if isinstance(model_results, dict) and 'message' not in model_results:
                                st.json(model_results)
                            else:
                                st.write(model_results)
        
        if is_advanced:
            # Advanced analysis tabs
            with tab2:
                st.subheader("🔬 Individual Dynamic Segments")
                
                if results.get('advanced_segmentation'):
                    for dynamic, seg_data in results['advanced_segmentation'].items():
                        with st.expander(f"📊 {dynamic.replace('_', ' ').title()} Segmentation", expanded=True):
                            if 'error' not in seg_data:
                                st.write(f"**Features Used:** {', '.join(seg_data['features_used'])}")
                                
                                # Segment distribution
                                st.write("**Segment Distribution:**")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    for segment, count in seg_data['segment_counts'].items():
                                        percentage = (count / sum(seg_data['segment_counts'].values())) * 100
                                        st.write(f"• **{segment}:** {count} customers ({percentage:.1f}%)")
                                
                                with col2:
                                    # Create a simple bar chart
                                    import matplotlib.pyplot as plt
                                    fig, ax = plt.subplots(figsize=(8, 4))
                                    segments = list(seg_data['segment_counts'].keys())
                                    counts = list(seg_data['segment_counts'].values())
                                    ax.bar(segments, counts)
                                    ax.set_title(f"{dynamic.replace('_', ' ').title()} Segment Distribution")
                                    ax.tick_params(axis='x', rotation=45)
                                    st.pyplot(fig)
                                    plt.close()
                            else:
                                st.error(f"Segmentation failed: {seg_data['error']}")
            
            with tab3:
                st.subheader("📊 Regression Analysis Results")
                
                if results.get('regression_analysis'):
                    reg_results = results['regression_analysis']
                    
                    # Variable selection logic explanation
                    if 'variable_selection_logic' in reg_results:
                        st.write("**🧠 Variable Selection Logic:**")
                        logic = reg_results['variable_selection_logic']
                        st.info(logic['dependent_variables_rationale'])
                        st.info(logic['independent_variables_rationale']) 
                        st.info(logic['segment_variables_usage'])
                    
                    # Regression models
                    st.write("**📈 Regression Models:**")
                    for dep_var, model_data in reg_results.items():
                        if dep_var != 'variable_selection_logic' and 'error' not in model_data:
                            with st.expander(f"Model: {dep_var}", expanded=True):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("R-squared", f"{model_data['r_squared']:.3f}")
                                    st.metric("MSE", f"{model_data['mse']:.3f}")
                                    st.metric("N Observations", model_data['n_observations'])
                                
                                with col2:
                                    st.write("**Feature Importance (Coefficients):**")
                                    for feature, coef in model_data['feature_importance'].items():
                                        color = "green" if coef > 0 else "red"
                                        st.write(f"• **{feature}:** {coef:.4f}")
                                
                                st.write(f"**Independent Variables:** {', '.join(model_data['independent_variables'])}")
                        elif 'error' in model_data:
                            st.error(f"Model {dep_var} failed: {model_data['error']}")
                else:
                    st.info("No regression analysis performed")
            
            with tab4:
                st.subheader("🎯 Dummy Variables Created")
                
                if results.get('dummy_variables'):
                    st.write("**Segment Dummy Variables for Regression Analysis:**")
                    st.write("These binary variables indicate segment membership and can be used as independent variables in regression models.")
                    
                    dummy_vars = results['dummy_variables']
                    st.write(f"**Created {len(dummy_vars)} dummy variables:**")
                    
                    for var_name in dummy_vars.keys():
                        segment_sum = sum(dummy_vars[var_name])
                        total_count = len(dummy_vars[var_name])
                        percentage = (segment_sum / total_count) * 100
                        st.write(f"• **{var_name}:** {segment_sum} customers ({percentage:.1f}%)")
                    
                    # Show sample of dummy variable data
                    with st.expander("📊 Sample Dummy Variable Data", expanded=False):
                        dummy_df = pd.DataFrame(dummy_vars)
                        st.dataframe(dummy_df.head(10))
                else:
                    st.info("No dummy variables created")
            
            with tab5:
                st.subheader("📥 Export Advanced Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📄 Export Complete Results", use_container_width=True):
                        json_str = json.dumps(results, indent=2, default=str)
                        st.download_button(
                            label="⬇️ Download JSON",
                            data=json_str,
                            file_name=f"advanced_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                
                with col2:
                    if st.button("📊 Export Segment Data", use_container_width=True) and results.get('dummy_variables'):
                        # Create CSV with dummy variables
                        dummy_df = pd.DataFrame(results['dummy_variables'])
                        csv_string = dummy_df.to_csv(index=False)
                        st.download_button(
                            label="⬇️ Download CSV",
                            data=csv_string,
                            file_name=f"segment_dummy_vars_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                with col3:
                    if st.button("📈 Export Regression Results", use_container_width=True) and results.get('regression_analysis'):
                        # Create summary of regression results
                        reg_summary = []
                        for dep_var, model_data in results['regression_analysis'].items():
                            if dep_var != 'variable_selection_logic' and 'error' not in model_data:
                                reg_summary.append({
                                    'dependent_variable': dep_var,
                                    'r_squared': model_data['r_squared'],
                                    'mse': model_data['mse'],
                                    'n_observations': model_data['n_observations'],
                                    'n_features': len(model_data['independent_variables'])
                                })
                        
                        if reg_summary:
                            reg_df = pd.DataFrame(reg_summary)
                            csv_string = reg_df.to_csv(index=False)
                            st.download_button(
                                label="⬇️ Download CSV",
                                data=csv_string,
                                file_name=f"regression_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
        
        else:
            # Standard analysis tabs
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
