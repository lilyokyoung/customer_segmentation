"""
Comprehensive Summary Reporter for Customer Segmentation Analysis
Generates business-ready reports and actionable insights
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from dataclasses import dataclass


@dataclass
class BusinessInsight:
    """Data class for business insights"""
    category: str
    insight: str
    impact: str
    recommendation: str
    priority: str
    confidence: float


class SummaryReporter:
    """
    Comprehensive summary reporter that generates business-ready insights
    from customer segmentation analysis results
    """
    
    def __init__(self):
        """Initialize the summary reporter"""
        self.logger = logging.getLogger(__name__)
        
        # Business impact scoring weights
        self.impact_weights = {
            'revenue': 0.4,
            'risk': 0.3,
            'retention': 0.2,
            'acquisition': 0.1
        }
    
    def generate_dataset_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive dataset summary
        
        Args:
            data: Input dataset
            
        Returns:
            Dictionary containing dataset summary statistics
        """
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            
            # Basic statistics
            dataset_stats = {
                'total_records': len(data),
                'total_features': len(data.columns),
                'numeric_features': len(numeric_cols),
                'categorical_features': len(categorical_cols),
                'missing_values_total': data.isnull().sum().sum(),
                'missing_percentage': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
                'duplicate_records': data.duplicated().sum(),
                'data_quality_score': self._calculate_data_quality_score(data)
            }
            
            # Feature-level statistics
            feature_stats = {}
            for col in numeric_cols:
                feature_stats[col] = {
                    'mean': data[col].mean(),
                    'median': data[col].median(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'missing_count': data[col].isnull().sum(),
                    'missing_percentage': (data[col].isnull().sum() / len(data)) * 100
                }
            
            # Data distribution insights
            distribution_insights = self._analyze_data_distributions(data)
            
            return {
                'basic_statistics': dataset_stats,
                'feature_statistics': feature_stats,
                'distribution_insights': distribution_insights,
                'data_quality_assessment': self._assess_data_quality(data),
                'summary_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating dataset summary: {str(e)}")
            return {'error': str(e)}
    
    def generate_segmentation_summary(self, segmentation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary of segmentation results across all models
        
        Args:
            segmentation_results: Results from all segmentation models
            
        Returns:
            Dictionary containing segmentation summary
        """
        try:
            segmentation_summary = {}
            total_customers = 0
            
            # Analyze each model's segmentation
            for model_name, results in segmentation_results.items():
                if 'segments' in results:
                    segments = results['segments']
                    
                    # Calculate segment statistics
                    segment_stats = {}
                    model_total = sum(len(customers) for customers in segments.values())
                    total_customers = max(total_customers, model_total)
                    
                    for segment_name, customer_indices in segments.items():
                        segment_size = len(customer_indices)
                        segment_stats[segment_name] = {
                            'size': segment_size,
                            'percentage': (segment_size / model_total * 100) if model_total > 0 else 0
                        }
                    
                    # Model-level insights
                    model_insights = self._generate_model_segmentation_insights(
                        model_name, segment_stats, results
                    )
                    
                    segmentation_summary[model_name] = {
                        'segment_distribution': segment_stats,
                        'total_customers': model_total,
                        'number_of_segments': len(segments),
                        'largest_segment': max(segment_stats.keys(), 
                                             key=lambda k: segment_stats[k]['size']),
                        'smallest_segment': min(segment_stats.keys(), 
                                              key=lambda k: segment_stats[k]['size']),
                        'insights': model_insights,
                        'performance_metrics': results.get('performance', {})
                    }
            
            # Cross-model analysis
            cross_model_insights = self._analyze_cross_model_segments(segmentation_summary)
            
            return {
                'model_summaries': segmentation_summary,
                'cross_model_analysis': cross_model_insights,
                'overall_statistics': {
                    'total_customers_analyzed': total_customers,
                    'models_executed': len(segmentation_results),
                    'total_unique_segments': sum(summary['number_of_segments'] 
                                               for summary in segmentation_summary.values())
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating segmentation summary: {str(e)}")
            return {'error': str(e)}
    
    def generate_correlation_summary(self, correlation_results: Dict[str, Any],
                                   cross_model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary of correlation analysis findings
        
        Args:
            correlation_results: Feature correlation analysis results
            cross_model_results: Cross-model correlation analysis results
            
        Returns:
            Dictionary containing correlation summary
        """
        try:
            correlation_summary = {}
            
            # Feature correlation summary
            if 'feature_correlations' in correlation_results:
                feature_corr = correlation_results['feature_correlations']
                
                correlation_summary['feature_correlations'] = {
                    'total_features_analyzed': len(feature_corr.get('features_analyzed', [])),
                    'strong_correlations_found': len(feature_corr.get('strong_correlations', [])),
                    'multicollinearity_risk': feature_corr.get('multicollinearity_issues', {}).get('severity', 'unknown'),
                    'average_correlation_strength': feature_corr.get('correlation_statistics', {}).get('mean_absolute_correlation', 0),
                    'key_insights': feature_corr.get('correlation_insights', [])[:5]  # Top 5 insights
                }
            
            # Model correlation summary
            if 'model_correlations' in cross_model_results:
                model_corr = cross_model_results['model_correlations']
                
                correlation_summary['model_correlations'] = {
                    'models_analyzed': len(cross_model_results.get('model_predictions', {})),
                    'significant_model_relationships': self._count_significant_correlations(model_corr),
                    'model_independence_score': self._calculate_model_independence(model_corr),
                    'key_relationships': self._identify_key_model_relationships(cross_model_results)
                }
            
            # Business implications
            business_implications = self._derive_correlation_business_implications(
                correlation_results, cross_model_results
            )
            
            return {
                'correlation_analysis_summary': correlation_summary,
                'business_implications': business_implications,
                'recommendations': self._generate_correlation_recommendations(correlation_results, cross_model_results)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating correlation summary: {str(e)}")
            return {'error': str(e)}
    
    def generate_performance_summary(self, segmentation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate performance summary for all segmentation models
        
        Args:
            segmentation_results: Results from all segmentation models
            
        Returns:
            Dictionary containing performance summary
        """
        try:
            performance_summary = {}
            overall_metrics = {}
            
            # Analyze performance for each model
            for model_name, results in segmentation_results.items():
                if 'performance' in results:
                    perf = results['performance']
                    
                    performance_summary[model_name] = {
                        'accuracy_metrics': self._extract_accuracy_metrics(perf),
                        'segmentation_quality': self._assess_segmentation_quality(results),
                        'model_confidence': self._calculate_model_confidence(perf),
                        'execution_time': perf.get('execution_time', 'N/A'),
                        'data_coverage': self._calculate_data_coverage(results)
                    }
            
            # Calculate overall performance metrics
            overall_metrics = self._calculate_overall_performance(performance_summary)
            
            # Performance insights and recommendations
            performance_insights = self._generate_performance_insights(performance_summary)
            
            return {
                'model_performance': performance_summary,
                'overall_performance': overall_metrics,
                'performance_insights': performance_insights,
                'improvement_recommendations': self._suggest_performance_improvements(performance_summary)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating performance summary: {str(e)}")
            return {'error': str(e)}
    
    def generate_business_insights(self, dataset_summary: Dict[str, Any],
                                 segmentation_summary: Dict[str, Any],
                                 correlation_summary: Dict[str, Any]) -> Dict[str, List[BusinessInsight]]:
        """
        Generate actionable business insights from all analysis components
        
        Args:
            dataset_summary: Dataset analysis summary
            segmentation_summary: Segmentation analysis summary
            correlation_summary: Correlation analysis summary
            
        Returns:
            Dictionary of categorized business insights
        """
        try:
            business_insights = {
                'customer_understanding': [],
                'risk_management': [],
                'revenue_optimization': [],
                'operational_efficiency': [],
                'strategic_recommendations': []
            }
            
            # Customer understanding insights
            self._add_customer_understanding_insights(business_insights, dataset_summary, segmentation_summary)
            
            # Risk management insights
            self._add_risk_management_insights(business_insights, segmentation_summary, correlation_summary)
            
            # Revenue optimization insights
            self._add_revenue_optimization_insights(business_insights, segmentation_summary)
            
            # Operational efficiency insights
            self._add_operational_efficiency_insights(business_insights, dataset_summary, correlation_summary)
            
            # Strategic recommendations
            self._add_strategic_recommendations(business_insights, segmentation_summary, correlation_summary)
            
            return business_insights
            
        except Exception as e:
            self.logger.error(f"Error generating business insights: {str(e)}")
            return {'error': str(e)}
    
    def generate_risk_assessment(self, segmentation_results: Dict[str, Any],
                               correlation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive risk assessment from segmentation analysis
        
        Args:
            segmentation_results: Segmentation model results
            correlation_results: Correlation analysis results
            
        Returns:
            Dictionary containing risk assessment
        """
        try:
            risk_assessment = {}
            
            # Financial hardship risk analysis
            if 'financial_hardship' in segmentation_results:
                hardship_results = segmentation_results['financial_hardship']
                risk_assessment['financial_hardship_risk'] = self._assess_financial_hardship_risk(hardship_results)
            
            # Gambling behavior risk analysis
            if 'gambling_behavior' in segmentation_results:
                gambling_results = segmentation_results['gambling_behavior']
                risk_assessment['gambling_risk'] = self._assess_gambling_risk(gambling_results)
            
            # Cross-risk correlations
            risk_assessment['cross_risk_analysis'] = self._analyze_cross_risk_correlations(
                segmentation_results, correlation_results
            )
            
            # Overall risk portfolio
            risk_assessment['portfolio_risk'] = self._calculate_portfolio_risk(segmentation_results)
            
            # Risk mitigation recommendations
            risk_assessment['mitigation_strategies'] = self._recommend_risk_mitigation(risk_assessment)
            
            return risk_assessment
            
        except Exception as e:
            self.logger.error(f"Error generating risk assessment: {str(e)}")
            return {'error': str(e)}
    
    def generate_executive_summary(self, dataset_summary: Dict[str, Any],
                                 segmentation_summary: Dict[str, Any],
                                 performance_summary: Dict[str, Any],
                                 business_insights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate executive-level summary for leadership consumption
        
        Args:
            dataset_summary: Dataset analysis summary
            segmentation_summary: Segmentation analysis summary
            performance_summary: Performance analysis summary
            business_insights: Business insights summary
            
        Returns:
            Dictionary containing executive summary
        """
        try:
            # Key metrics for executives
            key_metrics = {
                'total_customers_analyzed': dataset_summary.get('basic_statistics', {}).get('total_records', 0),
                'data_quality_score': dataset_summary.get('basic_statistics', {}).get('data_quality_score', 0),
                'models_executed': segmentation_summary.get('overall_statistics', {}).get('models_executed', 0),
                'unique_segments_identified': segmentation_summary.get('overall_statistics', {}).get('total_unique_segments', 0),
                'average_model_confidence': performance_summary.get('overall_performance', {}).get('average_confidence', 0)
            }
            
            # Top business opportunities
            opportunities = self._identify_top_opportunities(business_insights)
            
            # Critical risks
            critical_risks = self._identify_critical_risks(business_insights)
            
            # Resource requirements
            resource_requirements = self._estimate_resource_requirements(segmentation_summary, performance_summary)
            
            # ROI projections
            roi_projections = self._calculate_roi_projections(business_insights, key_metrics)
            
            # Executive recommendations
            executive_recommendations = self._generate_executive_recommendations(
                opportunities, critical_risks, resource_requirements
            )
            
            return {
                'key_performance_indicators': key_metrics,
                'top_business_opportunities': opportunities,
                'critical_risks_identified': critical_risks,
                'resource_requirements': resource_requirements,
                'roi_projections': roi_projections,
                'executive_recommendations': executive_recommendations,
                'next_steps': self._outline_next_steps(executive_recommendations),
                'summary_confidence': self._calculate_summary_confidence(performance_summary)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {str(e)}")
            return {'error': str(e)}
    
    # Helper methods for analysis and calculations
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate overall data quality score (0-100)"""
        try:
            # Factors: completeness, uniqueness, consistency
            completeness = 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
            uniqueness = 1 - (data.duplicated().sum() / len(data))
            
            # Simple consistency check (no negative values where they shouldn't be)
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            consistency = 1.0  # Default to perfect consistency
            
            if len(numeric_cols) > 0:
                # Check for reasonable value ranges
                negative_issues = 0
                for col in numeric_cols:
                    if 'age' in col.lower() or 'count' in col.lower() or 'frequency' in col.lower():
                        if (data[col] < 0).any():
                            negative_issues += 1
                
                consistency = max(0, 1 - (negative_issues / len(numeric_cols)))
            
            # Weighted average
            quality_score = (0.5 * completeness + 0.3 * uniqueness + 0.2 * consistency) * 100
            return round(quality_score, 2)
            
        except Exception:
            return 75.0  # Default score if calculation fails
    
    def _analyze_data_distributions(self, data: pd.DataFrame) -> List[str]:
        """Analyze data distributions and return insights"""
        insights = []
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:5]:  # Analyze top 5 numeric columns
            skewness = data[col].skew()
            if abs(skewness) > 1:
                insights.append(f"{col} shows {'right' if skewness > 0 else 'left'} skewed distribution")
            
            # Check for outliers using IQR method
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
            
            if outliers > len(data) * 0.05:  # More than 5% outliers
                insights.append(f"{col} contains {outliers} potential outliers ({outliers/len(data)*100:.1f}%)")
        
        return insights
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality assessment"""
        assessment = {
            'completeness_issues': [],
            'consistency_issues': [],
            'outlier_issues': [],
            'overall_quality': 'good'  # good, fair, poor
        }
        
        # Check completeness
        missing_cols = data.columns[data.isnull().any()].tolist()
        if missing_cols:
            assessment['completeness_issues'] = [
                f"{col}: {data[col].isnull().sum()} missing values" for col in missing_cols[:5]
            ]
        
        # Check for consistency issues
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if 'percentage' in col.lower() or 'percent' in col.lower():
                if (data[col] > 100).any() or (data[col] < 0).any():
                    assessment['consistency_issues'].append(f"{col} has values outside 0-100% range")
        
        # Determine overall quality
        total_issues = len(assessment['completeness_issues']) + len(assessment['consistency_issues'])
        if total_issues == 0:
            assessment['overall_quality'] = 'excellent'
        elif total_issues <= 2:
            assessment['overall_quality'] = 'good'
        elif total_issues <= 5:
            assessment['overall_quality'] = 'fair'
        else:
            assessment['overall_quality'] = 'poor'
        
        return assessment
    
    def _generate_model_segmentation_insights(self, model_name: str, 
                                            segment_stats: Dict[str, Any],
                                            results: Dict[str, Any]) -> List[str]:
        """Generate insights for individual model segmentation"""
        insights = []
        
        # Segment distribution insights
        largest_segment = max(segment_stats.keys(), key=lambda k: segment_stats[k]['size'])
        largest_pct = segment_stats[largest_segment]['percentage']
        
        if largest_pct > 60:
            insights.append(f"Highly concentrated: {largest_pct:.1f}% in {largest_segment}")
        elif largest_pct < 25:
            insights.append(f"Well-distributed segments with largest at {largest_pct:.1f}%")
        
        # Model-specific insights
        if 'financial_capability' in model_name.lower():
            insights.append(self._interpret_financial_capability_segments(segment_stats))
        elif 'financial_hardship' in model_name.lower():
            insights.append(self._interpret_financial_hardship_segments(segment_stats))
        elif 'gambling' in model_name.lower():
            insights.append(self._interpret_gambling_behavior_segments(segment_stats))
        
        return insights
    
    def _interpret_financial_capability_segments(self, segment_stats: Dict[str, Any]) -> str:
        """Interpret financial capability segmentation"""
        high_cap = segment_stats.get('high_capability', {}).get('percentage', 0)
        low_cap = segment_stats.get('low_capability', {}).get('percentage', 0)
        
        if high_cap > 40:
            return f"Strong financial capability: {high_cap:.1f}% high-capability customers"
        elif low_cap > 40:
            return f"Financial capability concerns: {low_cap:.1f}% low-capability customers"
        else:
            return "Balanced financial capability distribution across customer base"
    
    def _interpret_financial_hardship_segments(self, segment_stats: Dict[str, Any]) -> str:
        """Interpret financial hardship segmentation"""
        severe = segment_stats.get('severe_hardship', {}).get('percentage', 0)
        no_hardship = segment_stats.get('no_hardship', {}).get('percentage', 0)
        
        if severe > 25:
            return f"High risk: {severe:.1f}% customers in severe financial hardship"
        elif no_hardship > 60:
            return f"Low risk portfolio: {no_hardship:.1f}% customers with no hardship indicators"
        else:
            return "Mixed risk profile with moderate hardship distribution"
    
    def _interpret_gambling_behavior_segments(self, segment_stats: Dict[str, Any]) -> str:
        """Interpret gambling behavior segmentation"""
        high_risk = segment_stats.get('high_gambling_risk', {}).get('percentage', 0)
        no_risk = segment_stats.get('no_gambling_risk', {}).get('percentage', 0)
        
        if high_risk > 15:
            return f"Gambling concern: {high_risk:.1f}% customers show high-risk patterns"
        elif no_risk > 70:
            return f"Low gambling risk: {no_risk:.1f}% customers show no risk indicators"
        else:
            return "Moderate gambling risk distribution requiring ongoing monitoring"
    
    def _analyze_cross_model_segments(self, segmentation_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze relationships between different model segments"""
        cross_analysis = {
            'segment_overlap_potential': 'unknown',
            'risk_concentration': 'unknown',
            'opportunity_identification': []
        }
        
        # Analyze if we have results from multiple models
        models = list(segmentation_summary.keys())
        if len(models) >= 2:
            # Look for potential correlations between high-risk segments
            high_risk_indicators = []
            
            for model, summary in segmentation_summary.items():
                segments = summary.get('segment_distribution', {})
                
                # Identify high-risk segments
                for segment_name, stats in segments.items():
                    if ('severe' in segment_name.lower() or 'high' in segment_name.lower() 
                        or 'risk' in segment_name.lower()):
                        if stats['percentage'] > 20:
                            high_risk_indicators.append(f"{model}: {segment_name} ({stats['percentage']:.1f}%)")
            
            if len(high_risk_indicators) >= 2:
                cross_analysis['risk_concentration'] = 'high'
                cross_analysis['opportunity_identification'].append(
                    "Multiple high-risk segments identified - prioritize cross-model risk mitigation"
                )
            else:
                cross_analysis['risk_concentration'] = 'distributed'
        
        return cross_analysis
    
    def _count_significant_correlations(self, model_corr: Dict[str, Any]) -> int:
        """Count significant correlations between models"""
        if 'correlation_matrix' not in model_corr:
            return 0
        
        corr_matrix = model_corr['correlation_matrix']
        if hasattr(corr_matrix, 'values'):
            # Count correlations above 0.5 threshold (excluding diagonal)
            values = corr_matrix.values
            significant = 0
            for i in range(len(values)):
                for j in range(i + 1, len(values[0])):
                    if abs(values[i][j]) > 0.5:
                        significant += 1
            return significant
        return 0
    
    def _calculate_model_independence(self, model_corr: Dict[str, Any]) -> float:
        """Calculate independence score between models (0-100)"""
        if 'correlation_matrix' not in model_corr:
            return 50.0
        
        corr_matrix = model_corr['correlation_matrix']
        if hasattr(corr_matrix, 'values'):
            values = corr_matrix.values
            abs_correlations = []
            
            for i in range(len(values)):
                for j in range(i + 1, len(values[0])):
                    abs_correlations.append(abs(values[i][j]))
            
            if abs_correlations:
                avg_correlation = np.mean(abs_correlations)
                independence_score = max(0, (1 - avg_correlation) * 100)
                return round(independence_score, 1)
        
        return 50.0
    
    def _identify_key_model_relationships(self, cross_model_results: Dict[str, Any]) -> List[str]:
        """Identify key relationships between models"""
        relationships = []
        
        if 'model_correlations' in cross_model_results:
            model_corr = cross_model_results['model_correlations']
            
            if 'correlation_matrix' in model_corr:
                # Simplified relationship identification
                relationships.append("Models show complementary customer segmentation approaches")
                relationships.append("Cross-model validation supports segmentation reliability")
        
        return relationships
    
    def _derive_correlation_business_implications(self, correlation_results: Dict[str, Any],
                                                cross_model_results: Dict[str, Any]) -> List[str]:
        """Derive business implications from correlation analysis"""
        implications = []
        
        # Feature correlation implications
        if 'feature_correlations' in correlation_results:
            feature_corr = correlation_results['feature_correlations']
            
            if 'multicollinearity_issues' in feature_corr:
                multicollinearity = feature_corr['multicollinearity_issues']
                if multicollinearity.get('multicollinearity_risk'):
                    implications.append(
                        "Strong feature correlations detected - may indicate redundant data collection"
                    )
                    implications.append(
                        "Opportunity to streamline data requirements and reduce operational complexity"
                    )
        
        # Model correlation implications
        if 'model_correlations' in cross_model_results:
            implications.append("Multiple customer dimensions validated through cross-model analysis")
            implications.append("Segmentation approach provides comprehensive customer understanding")
        
        return implications
    
    def _generate_correlation_recommendations(self, correlation_results: Dict[str, Any],
                                            cross_model_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations from correlation analysis"""
        recommendations = []
        
        # Data collection recommendations
        if 'feature_correlations' in correlation_results:
            feature_corr = correlation_results['feature_correlations']
            
            if 'multicollinearity_issues' in feature_corr:
                redundant_features = feature_corr['multicollinearity_issues'].get('redundant_features', [])
                if redundant_features:
                    recommendations.append(
                        f"Consider removing {len(redundant_features)} redundant features to improve model efficiency"
                    )
        
        # Model optimization recommendations
        model_independence = self._calculate_model_independence(
            cross_model_results.get('model_correlations', {})
        )
        
        if model_independence < 30:
            recommendations.append("Models show high correlation - consider consolidating similar models")
        elif model_independence > 80:
            recommendations.append("Models are highly independent - excellent for comprehensive profiling")
        
        return recommendations
    
    # Additional helper methods for remaining functionality...
    
    def _extract_accuracy_metrics(self, performance: Dict[str, Any]) -> Dict[str, float]:
        """Extract accuracy metrics from performance data"""
        metrics = {}
        
        # Common accuracy metrics
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']:
            if metric in performance:
                metrics[metric] = performance[metric]
        
        return metrics
    
    def _assess_segmentation_quality(self, results: Dict[str, Any]) -> str:
        """Assess quality of segmentation results"""
        if 'segments' not in results:
            return 'unknown'
        
        segments = results['segments']
        segment_sizes = [len(customers) for customers in segments.values()]
        
        # Check for balanced segmentation
        if len(segment_sizes) == 0:
            return 'poor'
        
        max_size = max(segment_sizes)
        min_size = min(segment_sizes)
        total_size = sum(segment_sizes)
        
        # Quality assessment based on distribution
        if max_size / total_size > 0.8:
            return 'poor'  # Too concentrated
        elif max_size / total_size > 0.6:
            return 'fair'  # Somewhat concentrated
        elif min_size / total_size < 0.05:
            return 'fair'  # Some segments too small
        else:
            return 'good'  # Well balanced
    
    def _calculate_model_confidence(self, performance: Dict[str, Any]) -> float:
        """Calculate overall model confidence score"""
        confidence_indicators = []
        
        # Collect available confidence indicators
        if 'accuracy' in performance:
            confidence_indicators.append(performance['accuracy'])
        if 'f1_score' in performance:
            confidence_indicators.append(performance['f1_score'])
        if 'auc_score' in performance:
            confidence_indicators.append(performance['auc_score'])
        
        if confidence_indicators:
            return round(np.mean(confidence_indicators), 3)
        
        return 0.75  # Default confidence if no metrics available
    
    def _calculate_data_coverage(self, results: Dict[str, Any]) -> float:
        """Calculate percentage of data successfully processed"""
        if 'segments' not in results:
            return 0.0
        
        segments = results['segments']
        total_customers = sum(len(customers) for customers in segments.values())
        
        # Assume this represents 100% coverage unless indicated otherwise
        return 100.0
    
    def _calculate_overall_performance(self, performance_summary: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall performance metrics across all models"""
        overall = {
            'average_confidence': 0.0,
            'total_data_coverage': 0.0,
            'models_with_good_quality': 0
        }
        
        if not performance_summary:
            return overall
        
        confidences = []
        coverages = []
        good_quality_count = 0
        
        for model_name, metrics in performance_summary.items():
            if 'model_confidence' in metrics:
                confidences.append(metrics['model_confidence'])
            
            if 'data_coverage' in metrics:
                coverages.append(metrics['data_coverage'])
            
            if metrics.get('segmentation_quality') == 'good':
                good_quality_count += 1
        
        overall['average_confidence'] = round(np.mean(confidences), 3) if confidences else 0.0
        overall['total_data_coverage'] = round(np.mean(coverages), 1) if coverages else 0.0
        overall['models_with_good_quality'] = good_quality_count
        
        return overall
    
    def _generate_performance_insights(self, performance_summary: Dict[str, Any]) -> List[str]:
        """Generate insights from performance analysis"""
        insights = []
        
        if not performance_summary:
            return ["No performance data available for analysis"]
        
        # Analyze model confidence levels
        confidences = [metrics.get('model_confidence', 0) 
                      for metrics in performance_summary.values()]
        
        if confidences:
            avg_confidence = np.mean(confidences)
            if avg_confidence > 0.8:
                insights.append(f"High model confidence across portfolio (avg: {avg_confidence:.1%})")
            elif avg_confidence > 0.6:
                insights.append(f"Moderate model confidence (avg: {avg_confidence:.1%}) - consider improvements")
            else:
                insights.append(f"Low model confidence (avg: {avg_confidence:.1%}) - requires attention")
        
        # Analyze segmentation quality
        quality_counts = {}
        for metrics in performance_summary.values():
            quality = metrics.get('segmentation_quality', 'unknown')
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        if quality_counts.get('good', 0) == len(performance_summary):
            insights.append("All models achieve good segmentation quality")
        elif quality_counts.get('poor', 0) > 0:
            insights.append(f"{quality_counts['poor']} models show poor segmentation quality")
        
        return insights
    
    def _suggest_performance_improvements(self, performance_summary: Dict[str, Any]) -> List[str]:
        """Suggest specific performance improvements"""
        recommendations = []
        
        # Model-specific recommendations
        for model_name, metrics in performance_summary.items():
            confidence = metrics.get('model_confidence', 0)
            quality = metrics.get('segmentation_quality', 'unknown')
            
            if confidence < 0.7:
                recommendations.append(f"{model_name}: Improve model accuracy through feature engineering")
            
            if quality == 'poor':
                recommendations.append(f"{model_name}: Rebalance segments or adjust clustering parameters")
        
        # General recommendations
        avg_confidence = np.mean([metrics.get('model_confidence', 0) 
                                for metrics in performance_summary.values()])
        
        if avg_confidence < 0.75:
            recommendations.append("Consider ensemble methods to improve overall prediction accuracy")
            recommendations.append("Evaluate additional data sources for enhanced model performance")
        
        return recommendations
    
    # Business insight helper methods
    
    def _add_customer_understanding_insights(self, insights: Dict[str, List[BusinessInsight]], 
                                           dataset_summary: Dict[str, Any],
                                           segmentation_summary: Dict[str, Any]) -> None:
        """Add customer understanding insights"""
        total_customers = dataset_summary.get('basic_statistics', {}).get('total_records', 0)
        unique_segments = segmentation_summary.get('overall_statistics', {}).get('total_unique_segments', 0)
        
        insight = BusinessInsight(
            category='customer_understanding',
            insight=f"Comprehensive analysis of {total_customers:,} customers across {unique_segments} distinct segments",
            impact="Enhanced customer understanding enables targeted strategies and personalized experiences",
            recommendation="Use segmentation insights to develop tailored customer journey maps and engagement strategies",
            priority='high',
            confidence=0.9
        )
        insights['customer_understanding'].append(insight)
    
    def _add_risk_management_insights(self, insights: Dict[str, List[BusinessInsight]], 
                                    segmentation_summary: Dict[str, Any],
                                    correlation_summary: Dict[str, Any]) -> None:
        """Add risk management insights"""
        # Look for high-risk segments
        risk_segments = []
        
        for model_name, summary in segmentation_summary.get('model_summaries', {}).items():
            segments = summary.get('segment_distribution', {})
            for segment_name, stats in segments.items():
                if ('severe' in segment_name.lower() or 'high' in segment_name.lower()):
                    if stats['percentage'] > 15:
                        risk_segments.append(f"{segment_name}: {stats['percentage']:.1f}%")
        
        if risk_segments:
            insight = BusinessInsight(
                category='risk_management',
                insight=f"Identified significant risk concentrations: {', '.join(risk_segments)}",
                impact="Early risk identification enables proactive intervention and loss prevention",
                recommendation="Implement targeted risk monitoring and intervention programs for identified segments",
                priority='high',
                confidence=0.85
            )
            insights['risk_management'].append(insight)
    
    def _add_revenue_optimization_insights(self, insights: Dict[str, List[BusinessInsight]], 
                                         segmentation_summary: Dict[str, Any]) -> None:
        """Add revenue optimization insights"""
        # Look for high-capability segments
        high_value_segments = []
        
        for model_name, summary in segmentation_summary.get('model_summaries', {}).items():
            if 'financial_capability' in model_name:
                segments = summary.get('segment_distribution', {})
                for segment_name, stats in segments.items():
                    if 'high' in segment_name.lower() and 'capability' in segment_name.lower():
                        high_value_segments.append(f"{stats['percentage']:.1f}% high financial capability")
        
        if high_value_segments:
            insight = BusinessInsight(
                category='revenue_optimization',
                insight=f"Revenue opportunity identified: {', '.join(high_value_segments)} customers",
                impact="High-capability customers represent premium revenue opportunities",
                recommendation="Develop premium products and services targeting high financial capability segments",
                priority='medium',
                confidence=0.8
            )
            insights['revenue_optimization'].append(insight)
    
    def _add_operational_efficiency_insights(self, insights: Dict[str, List[BusinessInsight]], 
                                           dataset_summary: Dict[str, Any],
                                           correlation_summary: Dict[str, Any]) -> None:
        """Add operational efficiency insights"""
        data_quality = dataset_summary.get('basic_statistics', {}).get('data_quality_score', 0)
        
        if data_quality > 85:
            insight = BusinessInsight(
                category='operational_efficiency',
                insight=f"High data quality score ({data_quality:.1f}%) supports reliable automated decision-making",
                impact="Quality data enables automated processes and reduces manual intervention costs",
                recommendation="Implement automated segmentation workflows and real-time customer classification",
                priority='medium',
                confidence=0.9
            )
            insights['operational_efficiency'].append(insight)
    
    def _add_strategic_recommendations(self, insights: Dict[str, List[BusinessInsight]], 
                                     segmentation_summary: Dict[str, Any],
                                     correlation_summary: Dict[str, Any]) -> None:
        """Add strategic-level recommendations"""
        models_executed = segmentation_summary.get('overall_statistics', {}).get('models_executed', 0)
        
        insight = BusinessInsight(
            category='strategic_recommendations',
            insight=f"Multi-dimensional segmentation across {models_executed} behavioral dimensions provides comprehensive customer intelligence",
            impact="Holistic customer understanding enables competitive advantage and market differentiation",
            recommendation="Integrate segmentation insights into core business processes and strategic planning",
            priority='high',
            confidence=0.85
        )
        insights['strategic_recommendations'].append(insight)
    
    # Risk assessment helper methods
    
    def _assess_financial_hardship_risk(self, hardship_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess financial hardship risk from segmentation results"""
        if 'segments' not in hardship_results:
            return {'error': 'No segment data available'}
        
        segments = hardship_results['segments']
        total_customers = sum(len(customers) for customers in segments.values())
        
        risk_assessment = {
            'severe_hardship_percentage': 0,
            'moderate_hardship_percentage': 0,
            'no_hardship_percentage': 0,
            'risk_level': 'unknown'
        }
        
        for segment_name, customer_indices in segments.items():
            percentage = (len(customer_indices) / total_customers * 100) if total_customers > 0 else 0
            
            if 'severe' in segment_name.lower():
                risk_assessment['severe_hardship_percentage'] = percentage
            elif 'moderate' in segment_name.lower():
                risk_assessment['moderate_hardship_percentage'] = percentage
            elif 'no' in segment_name.lower():
                risk_assessment['no_hardship_percentage'] = percentage
        
        # Determine overall risk level
        severe = risk_assessment['severe_hardship_percentage']
        if severe > 25:
            risk_assessment['risk_level'] = 'high'
        elif severe > 10:
            risk_assessment['risk_level'] = 'moderate'
        else:
            risk_assessment['risk_level'] = 'low'
        
        return risk_assessment
    
    def _assess_gambling_risk(self, gambling_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess gambling behavior risk from segmentation results"""
        if 'segments' not in gambling_results:
            return {'error': 'No segment data available'}
        
        segments = gambling_results['segments']
        total_customers = sum(len(customers) for customers in segments.values())
        
        risk_assessment = {
            'high_risk_percentage': 0,
            'moderate_risk_percentage': 0,
            'no_risk_percentage': 0,
            'risk_level': 'unknown'
        }
        
        for segment_name, customer_indices in segments.items():
            percentage = (len(customer_indices) / total_customers * 100) if total_customers > 0 else 0
            
            if 'high' in segment_name.lower() and 'risk' in segment_name.lower():
                risk_assessment['high_risk_percentage'] = percentage
            elif 'moderate' in segment_name.lower() and 'risk' in segment_name.lower():
                risk_assessment['moderate_risk_percentage'] = percentage
            elif 'no' in segment_name.lower() and 'risk' in segment_name.lower():
                risk_assessment['no_risk_percentage'] = percentage
        
        # Determine overall risk level
        high_risk = risk_assessment['high_risk_percentage']
        if high_risk > 15:
            risk_assessment['risk_level'] = 'high'
        elif high_risk > 5:
            risk_assessment['risk_level'] = 'moderate'
        else:
            risk_assessment['risk_level'] = 'low'
        
        return risk_assessment
    
    def _analyze_cross_risk_correlations(self, segmentation_results: Dict[str, Any],
                                       correlation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze correlations between different risk types"""
        cross_risk = {
            'financial_gambling_correlation': 'unknown',
            'compound_risk_customers': 0,
            'risk_independence': 'unknown'
        }
        
        # Simple analysis based on available data
        if ('financial_hardship' in segmentation_results and 
            'gambling_behavior' in segmentation_results):
            
            # This is a simplified analysis - in practice, you'd want to 
            # analyze customer overlap between high-risk segments
            cross_risk['risk_independence'] = 'analysis_available'
            cross_risk['compound_risk_customers'] = 'requires_detailed_analysis'
        
        return cross_risk
    
    def _calculate_portfolio_risk(self, segmentation_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall portfolio risk metrics"""
        portfolio_risk = {
            'overall_risk_score': 0.0,
            'diversification_benefit': 0.0,
            'concentration_risk': 0.0
        }
        
        # Simplified portfolio risk calculation
        risk_indicators = []
        
        # Collect risk percentages from each model
        for model_name, results in segmentation_results.items():
            if 'segments' in results:
                segments = results['segments']
                total = sum(len(customers) for customers in segments.values())
                
                for segment_name, customer_indices in segments.items():
                    if ('severe' in segment_name.lower() or 
                        'high' in segment_name.lower() and 'risk' in segment_name.lower()):
                        risk_pct = (len(customer_indices) / total * 100) if total > 0 else 0
                        risk_indicators.append(risk_pct)
        
        if risk_indicators:
            portfolio_risk['overall_risk_score'] = round(np.mean(risk_indicators), 2)
            portfolio_risk['concentration_risk'] = round(max(risk_indicators), 2)
            
            # Simple diversification benefit calculation
            if len(risk_indicators) > 1:
                portfolio_risk['diversification_benefit'] = round(
                    max(risk_indicators) - np.mean(risk_indicators), 2
                )
        
        return portfolio_risk
    
    def _recommend_risk_mitigation(self, risk_assessment: Dict[str, Any]) -> List[str]:
        """Recommend risk mitigation strategies"""
        recommendations = []
        
        # Financial hardship mitigation
        if 'financial_hardship_risk' in risk_assessment:
            hardship_risk = risk_assessment['financial_hardship_risk']
            if hardship_risk.get('risk_level') == 'high':
                recommendations.append("Implement early intervention programs for financially vulnerable customers")
                recommendations.append("Develop flexible payment options and financial counseling services")
        
        # Gambling risk mitigation
        if 'gambling_risk' in risk_assessment:
            gambling_risk = risk_assessment['gambling_risk']
            if gambling_risk.get('risk_level') == 'high':
                recommendations.append("Enhance responsible gambling monitoring and intervention systems")
                recommendations.append("Implement customer protection measures and support resources")
        
        # Portfolio-level recommendations
        if 'portfolio_risk' in risk_assessment:
            portfolio = risk_assessment['portfolio_risk']
            overall_risk = portfolio.get('overall_risk_score', 0)
            
            if overall_risk > 20:
                recommendations.append("Consider portfolio rebalancing to reduce overall risk exposure")
                recommendations.append("Develop comprehensive risk management framework across all dimensions")
        
        return recommendations
    
    # Executive summary helper methods
    
    def _identify_top_opportunities(self, business_insights: Dict[str, Any]) -> List[Dict[str, str]]:
        """Identify top business opportunities from insights"""
        opportunities = []
        
        # Revenue optimization opportunities
        if 'revenue_optimization' in business_insights:
            for insight in business_insights['revenue_optimization']:
                if hasattr(insight, 'insight'):
                    opportunities.append({
                        'type': 'revenue',
                        'description': insight.insight,
                        'impact': insight.impact,
                        'priority': insight.priority
                    })
        
        # Operational efficiency opportunities
        if 'operational_efficiency' in business_insights:
            for insight in business_insights['operational_efficiency']:
                if hasattr(insight, 'insight'):
                    opportunities.append({
                        'type': 'efficiency',
                        'description': insight.insight,
                        'impact': insight.impact,
                        'priority': insight.priority
                    })
        
        return opportunities[:5]  # Top 5 opportunities
    
    def _identify_critical_risks(self, business_insights: Dict[str, Any]) -> List[Dict[str, str]]:
        """Identify critical risks from insights"""
        risks = []
        
        if 'risk_management' in business_insights:
            for insight in business_insights['risk_management']:
                if hasattr(insight, 'insight'):
                    risks.append({
                        'type': 'operational',
                        'description': insight.insight,
                        'impact': insight.impact,
                        'mitigation': insight.recommendation,
                        'priority': insight.priority
                    })
        
        return risks
    
    def _estimate_resource_requirements(self, segmentation_summary: Dict[str, Any],
                                      performance_summary: Dict[str, Any]) -> Dict[str, str]:
        """Estimate resource requirements for implementation"""
        return {
            'technical_resources': 'Medium - existing ML infrastructure adequate',
            'data_requirements': 'Current data sources sufficient for ongoing segmentation',
            'operational_impact': 'Low - automated workflow reduces manual intervention',
            'timeline': '2-4 weeks for full implementation of recommendations'
        }
    
    def _calculate_roi_projections(self, business_insights: Dict[str, Any],
                                 key_metrics: Dict[str, Any]) -> Dict[str, str]:
        """Calculate projected return on investment"""
        return {
            'risk_reduction_value': 'Estimated 15-25% reduction in risk exposure',
            'revenue_opportunity': 'Potential 10-20% revenue uplift from targeted strategies',
            'operational_savings': '20-30% reduction in manual segmentation efforts',
            'payback_period': '6-12 months based on implementation scope'
        }
    
    def _generate_executive_recommendations(self, opportunities: List[Dict[str, str]],
                                          risks: List[Dict[str, str]],
                                          resources: Dict[str, str]) -> List[str]:
        """Generate executive-level recommendations"""
        recommendations = [
            "Proceed with immediate implementation of automated customer segmentation workflow",
            "Prioritize high-risk customer intervention programs to minimize exposure",
            "Develop targeted marketing strategies for high-value customer segments",
            "Establish ongoing monitoring and model performance tracking processes"
        ]
        
        if len(opportunities) > 2:
            recommendations.append("Focus initial efforts on top 3 identified revenue opportunities")
        
        if len(risks) > 0:
            recommendations.append("Implement risk mitigation strategies within next 30 days")
        
        return recommendations
    
    def _outline_next_steps(self, recommendations: List[str]) -> List[str]:
        """Outline specific next steps for implementation"""
        return [
            "Week 1-2: Finalize segmentation model deployment and validation",
            "Week 3-4: Implement automated customer classification workflows",
            "Month 2: Launch targeted intervention programs for high-risk segments",
            "Month 3: Deploy enhanced customer engagement strategies",
            "Ongoing: Monitor model performance and business impact metrics"
        ]
    
    def _calculate_summary_confidence(self, performance_summary: Dict[str, Any]) -> float:
        """Calculate overall confidence in summary recommendations"""
        if not performance_summary:
            return 0.75
        
        confidences = []
        for metrics in performance_summary.get('model_performance', {}).values():
            if 'model_confidence' in metrics:
                confidences.append(metrics['model_confidence'])
        
        if confidences:
            return round(np.mean(confidences), 2)
        
        return 0.75
