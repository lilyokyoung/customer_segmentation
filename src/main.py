"""
Main AI Agent Orchestrator for Customer Segmentation
Enhanced with Correlation Analysis and Summary Reporting
"""
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# Handle optional dependencies gracefully
try:
    import yaml
except ImportError:
    print("Warning: PyYAML not installed. Using default configuration.")
    yaml = None

try:
    from agents.segmentation_agent import CustomerSegmentationAgent
except ImportError:
    print("Warning: CustomerSegmentationAgent not found. Using mock agent.")
    CustomerSegmentationAgent = None

try:
    from pipeline.data_pipeline import DataPipeline
except ImportError:
    print("Warning: DataPipeline not found. Using basic data processing.")
    DataPipeline = None

try:
    from models.model_manager import ModelManager
except ImportError:
    print("Warning: ModelManager not found. Using basic model management.")
    ModelManager = None

try:
    from utils.logger import setup_logger
except ImportError:
    print("Warning: Custom logger not found. Using basic logging.")
    def setup_logger(config):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

try:
    from utils.correlation_analyzer import CorrelationAnalyzer
except ImportError:
    print("Warning: CorrelationAnalyzer not found. Correlation analysis will be limited.")
    CorrelationAnalyzer = None

try:
    from utils.summary_reporter import SummaryReporter
except ImportError:
    print("Warning: SummaryReporter not found. Summary reporting will be limited.")
    SummaryReporter = None


# Mock classes for graceful degradation
class MockDataPipeline:
    def __init__(self, config=None):
        self.config = config or {}
    
    async def preprocess(self, data):
        return data
    
    async def engineer_features(self, data):
        return data
    
    async def create_sample_data(self, n_samples=1000):
        # Create basic sample data for testing
        if pd:
            try:
                import numpy as np
                return pd.DataFrame({
                    'customer_id': range(n_samples),
                    'monthly_income': np.random.normal(5000, 1500, n_samples),
                    'credit_score': np.random.normal(700, 100, n_samples),
                    'debt_to_income_ratio': np.random.uniform(0.1, 0.8, n_samples),
                })
            except ImportError:
                # Fallback without numpy
                return pd.DataFrame({
                    'customer_id': range(n_samples),
                    'monthly_income': [5000] * n_samples,
                    'credit_score': [700] * n_samples,
                    'debt_to_income_ratio': [0.3] * n_samples,
                })
        else:
            return {"message": "Sample data creation unavailable - pandas not installed"}


class MockModelManager:
    def __init__(self, config=None):
        self.config = config or {}
    
    async def train_models(self, data):
        return {}
    
    async def predict(self, data):
        return {}


class MockSegmentationAgent:
    def __init__(self, config=None):
        self.config = config or {}
    
    async def analyze_segments(self, predictions):
        return {}
    
    async def segment_financial_capability(self, data):
        return {"message": "Financial capability segmentation unavailable - mock mode"}
    
    async def segment_financial_hardship(self, data):
        return {"message": "Financial hardship segmentation unavailable - mock mode"}
    
    async def segment_gambling_behavior(self, data):
        return {"message": "Gambling behavior segmentation unavailable - mock mode"}


class MockCorrelationAnalyzer:
    def __init__(self):
        pass
    
    def analyze_feature_correlations(self, data, prefix=""):
        return {"message": "Feature correlation analysis unavailable - missing dependencies"}
    
    def analyze_target_correlations(self, data):
        return {"message": "Target correlation analysis unavailable - missing dependencies"}
    
    def generate_correlation_insights(self, feature_corr, target_corr=None):
        return {"message": "Correlation insights unavailable - missing dependencies"}
    
    def analyze_cross_correlations(self, data, predictions, prefix=""):
        return {"message": "Cross-correlation analysis unavailable - missing dependencies"}
    
    def analyze_model_relationships(self, predictions, model_correlations=None, feature_model_correlations=None):
        return {"message": "Model relationship analysis unavailable - missing dependencies"}


class MockSummaryReporter:
    def __init__(self):
        pass
    
    def generate_dataset_summary(self, data):
        return {"message": "Dataset summary unavailable - missing dependencies"}
    
    def generate_segmentation_summary(self, results):
        return {"message": "Segmentation summary unavailable - missing dependencies"}
    
    def generate_correlation_summary(self, correlations, cross_model_results=None):
        return {"message": "Correlation summary unavailable - missing dependencies"}
    
    def generate_performance_summary(self, results):
        return {"message": "Performance summary unavailable - missing dependencies"}
    
    def generate_business_insights(self, dataset_summary=None, segmentation_summary=None, 
                                 correlation_summary=None, *args):
        return {"message": "Business insights unavailable - missing dependencies"}
    
    def generate_risk_assessment(self, segmentation_results=None, correlation_results=None, *args):
        return {"message": "Risk assessment unavailable - missing dependencies"}
    
    def generate_executive_summary(self, dataset_summary=None, segmentation_summary=None, 
                                 performance_summary=None, business_insights=None, *args):
        return {"message": "Executive summary unavailable - missing dependencies"}


class CustomerSegmentationOrchestrator:
    """
    Enhanced AI Agent Orchestrator that coordinates multiple segmentation models
    with comprehensive correlation analysis and summary reporting
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the orchestrator with configuration"""
        self.config = self._load_config(config_path)
        
        # Ensure we have a logging config
        if not self.config or 'logging' not in self.config:
            self.config = self._get_default_config()
            
        self.logger = setup_logger(self.config['logging'])
        
        # Initialize components with error handling
        try:
            if DataPipeline:
                self.data_pipeline = DataPipeline(self.config['data'])
            else:
                self.data_pipeline = MockDataPipeline()
                
            if ModelManager:
                self.model_manager = ModelManager(self.config['models'])
            else:
                self.model_manager = MockModelManager()
                
            if CustomerSegmentationAgent:
                self.agent = CustomerSegmentationAgent(self.config['agent'])
            else:
                self.agent = MockSegmentationAgent()
        except Exception as e:
            self.logger.warning(f"Using mock components due to import issues: {e}")
            self.data_pipeline = MockDataPipeline()
            self.model_manager = MockModelManager()
            self.agent = MockSegmentationAgent()
        
        # Initialize analysis components
        try:
            self.correlation_analyzer = CorrelationAnalyzer() if CorrelationAnalyzer else MockCorrelationAnalyzer()
            self.summary_reporter = SummaryReporter() if SummaryReporter else MockSummaryReporter()
        except Exception as e:
            self.logger.warning(f"Using mock analytics components: {e}")
            self.correlation_analyzer = MockCorrelationAnalyzer()
            self.summary_reporter = MockSummaryReporter()
        
        # Store analysis results
        self.analysis_results = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if yaml and Path(config_path).exists():
                with open(config_path, 'r') as file:
                    config = yaml.safe_load(file)
                    return config if config else self._get_default_config()
            else:
                if not yaml:
                    print("Warning: YAML not available, using default config")
                else:
                    print(f"Warning: Config file {config_path} not found, using default config")
                return self._get_default_config()
        except Exception as e:
            print(f"Warning: Error loading config: {e}, using default config")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'agent': {'name': 'CustomerSegmentationAgent', 'orchestration_mode': 'sequential'},
            'data': {'batch_size': 1000, 'validation_split': 0.2},
            'models': {
                'financial_capability': {'features': ['monthly_income', 'credit_score']},
                'financial_hardship': {'features': ['debt_to_income_ratio', 'payment_delays_count']},
                'gambling_behavior': {'features': ['gambling_merchant_frequency', 'large_cash_withdrawals']}
            },
            'logging': {'level': 'INFO', 'file': 'logs/customer_segmentation.log'}
        }
    
    async def run_enhanced_segmentation_workflow(self, input_data: Any) -> Dict[str, Any]:
        """
        Run the complete enhanced customer segmentation workflow with analysis
        
        Args:
            input_data: Customer data for segmentation
            
        Returns:
            Dictionary containing segmentation results, correlation analysis, and summary reports
        """
        self.logger.info("Starting enhanced customer segmentation workflow")
        
        try:
            # Step 1: Data preprocessing and feature engineering
            self.logger.info("Step 1: Preprocessing and engineering features")
            processed_data = await self.data_pipeline.preprocess(input_data)
            feature_data = await self.data_pipeline.engineer_features(processed_data)
            
            # Step 2: Correlation Analysis
            self.logger.info("Step 2: Performing comprehensive correlation analysis")
            correlation_results = await self._perform_correlation_analysis(feature_data)
            
            # Step 3: Run segmentation models
            self.logger.info("Step 3: Running segmentation models")
            if self.config['agent']['orchestration_mode'] == 'parallel':
                segmentation_results = await self._run_parallel_segmentation(feature_data)
            else:
                segmentation_results = await self._run_sequential_segmentation(feature_data)
            
            # Step 4: Cross-model correlation analysis
            self.logger.info("Step 4: Analyzing cross-model correlations")
            cross_model_analysis = await self._analyze_cross_model_correlations(
                feature_data, segmentation_results
            )
            
            # Step 5: Generate comprehensive summary report
            self.logger.info("Step 5: Generating comprehensive summary report")
            summary_report = await self._generate_comprehensive_summary(
                feature_data, segmentation_results, correlation_results, cross_model_analysis
            )
            
            # Step 6: Combine and structure final results
            final_results = {
                'segmentation_results': segmentation_results,
                'correlation_analysis': correlation_results,
                'cross_model_analysis': cross_model_analysis,
                'summary_report': summary_report,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_customers': len(input_data),
                    'features_analyzed': len(feature_data.columns),
                    'models_executed': len(segmentation_results)
                }
            }
            
            # Store results for future reference
            self.analysis_results = final_results
            
            self.logger.info("Enhanced customer segmentation workflow completed successfully")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error in enhanced segmentation workflow: {str(e)}")
            raise
    
    async def _perform_correlation_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive correlation analysis on the dataset"""
        self.logger.info("Performing comprehensive correlation analysis")
        
        # Feature correlation analysis
        feature_correlations = self.correlation_analyzer.analyze_feature_correlations(data)
        
        # Model-specific feature correlations
        model_feature_correlations = {}
        for model_name, config in self.config['models'].items():
            if 'features' in config:
                model_features = [f for f in config['features'] if f in data.columns]
                if len(model_features) > 1:
                    model_feature_correlations[model_name] = (
                        self.correlation_analyzer.analyze_feature_correlations(
                            data[model_features], prefix=f"{model_name}_"
                        )
                    )
        
        # Target variable correlations (if available)
        target_correlations = self.correlation_analyzer.analyze_target_correlations(data)
        
        # Correlation insights and recommendations
        correlation_insights = self.correlation_analyzer.generate_correlation_insights(
            feature_correlations, model_feature_correlations
        )
        
        return {
            'feature_correlations': feature_correlations,
            'model_feature_correlations': model_feature_correlations,
            'target_correlations': target_correlations,
            'insights': correlation_insights,
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'features_analyzed': len(data.columns),
                'correlation_method': 'pearson'
            }
        }
    
    async def _run_parallel_segmentation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run all segmentation models in parallel"""
        tasks = [
            self.agent.segment_financial_capability(data),
            self.agent.segment_financial_hardship(data),
            self.agent.segment_gambling_behavior(data)
        ]
        
        results = await asyncio.gather(*tasks)
        
        return {
            'financial_capability': results[0],
            'financial_hardship': results[1],
            'gambling_behavior': results[2]
        }
    
    async def _run_sequential_segmentation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run segmentation models sequentially"""
        results = {}
        
        # Financial Capability Segmentation
        results['financial_capability'] = await self.agent.segment_financial_capability(data)
        
        # Financial Hardship Segmentation
        results['financial_hardship'] = await self.agent.segment_financial_hardship(data)
        
        # Gambling Behavior Segmentation
        results['gambling_behavior'] = await self.agent.segment_gambling_behavior(data)
        
        return results
    
    async def _analyze_cross_model_correlations(self, data: pd.DataFrame, 
                                              segmentation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze correlations between different model outputs"""
        self.logger.info("Analyzing cross-model correlations")
        
        # Extract predictions from each model
        predictions_data = {}
        
        # Get model predictions
        for model_name, results in segmentation_results.items():
            if 'segments' in results:
                # Convert segments to numerical format for correlation analysis
                segment_scores = self._convert_segments_to_scores(results['segments'])
                predictions_data[f"{model_name}_score"] = segment_scores
        
        if len(predictions_data) < 2:
            return {'warning': 'Insufficient model outputs for cross-correlation analysis'}
        
        # Create DataFrame with all model predictions
        predictions_df = pd.DataFrame(predictions_data)
        
        # Analyze correlations between model outputs
        model_correlations = self.correlation_analyzer.analyze_feature_correlations(
            predictions_df, prefix="cross_model_"
        )
        
        # Analyze relationships between model outputs and input features
        feature_model_correlations = self.correlation_analyzer.analyze_cross_correlations(
            data, predictions_df
        )
        
        # Generate insights about model relationships
        cross_model_insights = self.correlation_analyzer.analyze_model_relationships(
            model_correlations, feature_model_correlations
        )
        
        return {
            'model_correlations': model_correlations,
            'feature_model_correlations': feature_model_correlations,
            'insights': cross_model_insights,
            'model_predictions': predictions_data
        }
    
    def _convert_segments_to_scores(self, segments: Dict[str, List[int]]) -> List[float]:
        """Convert segment assignments to numerical scores"""
        scores = [0.0] * sum(len(customers) for customers in segments.values())
        
        # Assign scores based on segment type
        segment_scores = {
            'high_capability': 0.8, 'medium_capability': 0.5, 'low_capability': 0.2,
            'no_hardship': 0.2, 'moderate_hardship': 0.5, 'severe_hardship': 0.8,
            'no_gambling_risk': 0.2, 'moderate_gambling_risk': 0.5, 'high_gambling_risk': 0.8
        }
        
        for segment_name, customer_indices in segments.items():
            score = segment_scores.get(segment_name, 0.5)
            for idx in customer_indices:
                if idx < len(scores):
                    scores[idx] = score
        
        return scores
    
    async def _generate_comprehensive_summary(self, data: pd.DataFrame, 
                                            segmentation_results: Dict[str, Any],
                                            correlation_results: Dict[str, Any],
                                            cross_model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        self.logger.info("Generating comprehensive summary report")
        
        # Dataset summary
        dataset_summary = self.summary_reporter.generate_dataset_summary(data)
        
        # Segmentation summary
        segmentation_summary = self.summary_reporter.generate_segmentation_summary(
            segmentation_results
        )
        
        # Correlation summary
        correlation_summary = self.summary_reporter.generate_correlation_summary(
            correlation_results, cross_model_results
        )
        
        # Performance summary
        performance_summary = self.summary_reporter.generate_performance_summary(
            segmentation_results
        )
        
        # Business insights and recommendations
        business_insights = self.summary_reporter.generate_business_insights(
            dataset_summary, segmentation_summary, correlation_summary
        )
        
        # Risk assessment summary
        risk_assessment = self.summary_reporter.generate_risk_assessment(
            segmentation_results, correlation_results
        )
        
        # Executive summary
        executive_summary = self.summary_reporter.generate_executive_summary(
            dataset_summary, segmentation_summary, performance_summary, business_insights
        )
        
        return {
            'executive_summary': executive_summary,
            'dataset_summary': dataset_summary,
            'segmentation_summary': segmentation_summary,
            'correlation_summary': correlation_summary,
            'performance_summary': performance_summary,
            'business_insights': business_insights,
            'risk_assessment': risk_assessment,
            'generation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'report_version': '1.0.0',
                'analysis_scope': 'comprehensive'
            }
        }
    
    def export_analysis_results(self, output_path: str = "data/output/") -> Dict[str, str]:
        """Export analysis results to various formats"""
        if not self.analysis_results:
            raise ValueError("No analysis results available. Run workflow first.")
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        try:
            # Export to JSON
            json_path = output_dir / "segmentation_analysis_results.json"
            import json
            with open(json_path, 'w') as f:
                json.dump(self.analysis_results, f, indent=2, default=str)
            exported_files['json'] = str(json_path)
            
            # Export summary report to text
            txt_path = output_dir / "summary_report.txt"
            with open(txt_path, 'w') as f:
                self._write_text_summary(f, self.analysis_results['summary_report'])
            exported_files['text_report'] = str(txt_path)
            
            # Export correlation matrices to CSV
            if 'correlation_analysis' in self.analysis_results:
                corr_path = output_dir / "correlation_analysis.csv"
                self._export_correlation_data(corr_path, self.analysis_results['correlation_analysis'])
                exported_files['correlation_csv'] = str(corr_path)
            
            self.logger.info(f"Analysis results exported to {len(exported_files)} files")
            return exported_files
            
        except Exception as e:
            self.logger.error(f"Error exporting analysis results: {str(e)}")
            raise
    
    def _write_text_summary(self, file_handle, summary_data: Dict[str, Any]) -> None:
        """Write summary report to text file"""
        file_handle.write("CUSTOMER SEGMENTATION ANALYSIS SUMMARY REPORT\n")
        file_handle.write("=" * 60 + "\n\n")
        
        # Executive Summary
        if 'executive_summary' in summary_data:
            file_handle.write("EXECUTIVE SUMMARY\n")
            file_handle.write("-" * 20 + "\n")
            for key, value in summary_data['executive_summary'].items():
                file_handle.write(f"{key.replace('_', ' ').title()}: {value}\n")
            file_handle.write("\n")
        
        # Dataset Summary
        if 'dataset_summary' in summary_data:
            file_handle.write("DATASET SUMMARY\n")
            file_handle.write("-" * 20 + "\n")
            for key, value in summary_data['dataset_summary'].items():
                file_handle.write(f"{key.replace('_', ' ').title()}: {value}\n")
            file_handle.write("\n")
        
        # Business Insights
        if 'business_insights' in summary_data:
            file_handle.write("BUSINESS INSIGHTS\n")
            file_handle.write("-" * 20 + "\n")
            insights = summary_data['business_insights']
            if isinstance(insights, dict):
                for category, insight_list in insights.items():
                    file_handle.write(f"\n{category.replace('_', ' ').title()}:\n")
                    if isinstance(insight_list, list):
                        for insight in insight_list:
                            file_handle.write(f"  • {insight}\n")
                    else:
                        file_handle.write(f"  {insight_list}\n")
    
    def _export_correlation_data(self, file_path: Path, correlation_data: Dict[str, Any]) -> None:
        """Export correlation analysis data to CSV"""
        if 'feature_correlations' in correlation_data:
            corr_matrix = correlation_data['feature_correlations'].get('correlation_matrix')
            if corr_matrix is not None and hasattr(corr_matrix, 'to_csv'):
                corr_matrix.to_csv(file_path)


async def main():
    """Main entry point for the enhanced customer segmentation workflow"""
    # Initialize orchestrator
    orchestrator = CustomerSegmentationOrchestrator()
    
    # Create or load sample data
    try:
        sample_data = await orchestrator.data_pipeline.create_sample_data(n_samples=1000)
        if isinstance(sample_data, dict) and "message" in sample_data:
            print("⚠️ Warning: Using mock data due to missing dependencies")
            # Create a minimal dataframe for demo
            sample_data = {"customer_id": [1, 2, 3], "income": [50000, 60000, 70000]}
        print(f"📊 Created sample data with {len(sample_data)} customers")
    except Exception as e:
        print(f"Error creating sample data: {e}")
        return

    # Run enhanced segmentation workflow
    try:
        if pd and hasattr(sample_data, 'shape'):
            results = await orchestrator.run_enhanced_segmentation_workflow(sample_data)
        else:
            print("⚠️ Skipping workflow due to missing pandas or invalid data format")
            return        # Display summary
        print("\n" + "="*60)
        print("🎉 ENHANCED SEGMENTATION WORKFLOW COMPLETED")
        print("="*60)
        
        print(f"📊 Total customers analyzed: {results['metadata']['total_customers']}")
        print(f"🔧 Features analyzed: {results['metadata']['features_analyzed']}")
        print(f"🤖 Models executed: {results['metadata']['models_executed']}")
        
        # Export results
        exported_files = orchestrator.export_analysis_results()
        print(f"\n📁 Results exported to {len(exported_files)} files:")
        for file_type, file_path in exported_files.items():
            print(f"  • {file_type}: {file_path}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in enhanced workflow: {e}")
        raise


if __name__ == "__main__":
    results = asyncio.run(main())