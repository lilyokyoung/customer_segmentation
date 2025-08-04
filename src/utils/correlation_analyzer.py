"""
Advanced Correlation Analysis Module for Customer Segmentation
Provides comprehensive correlation analysis capabilities
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from scipy.stats import pearsonr, spearmanr, kendalltau
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CorrelationInsight:
    """Data class for storing correlation insights"""
    correlation_type: str
    strength: str
    variables: Tuple[str, str]
    value: float
    p_value: Optional[float]
    interpretation: str
    recommendation: str


class CorrelationAnalyzer:
    """
    Advanced correlation analyzer for multi-dimensional customer segmentation data
    """
    
    def __init__(self):
        """Initialize the correlation analyzer"""
        self.logger = logging.getLogger(__name__)
        
        # Correlation strength thresholds
        self.strength_thresholds = {
            'very_weak': 0.1,
            'weak': 0.3,
            'moderate': 0.5,
            'strong': 0.7,
            'very_strong': 0.9
        }
    
    def analyze_feature_correlations(self, data: pd.DataFrame, 
                                   method: str = 'pearson',
                                   prefix: str = "") -> Dict[str, Any]:
        """
        Analyze correlations between features in the dataset
        
        Args:
            data: Input DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')
            prefix: Prefix for analysis keys
            
        Returns:
            Dictionary containing correlation analysis results
        """
        try:
            # Filter numeric columns only
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                return {'error': 'No numeric columns found for correlation analysis'}
            
            # Calculate correlation matrix
            if method == 'pearson':
                corr_matrix = numeric_data.corr(method='pearson')
            elif method == 'spearman':
                corr_matrix = numeric_data.corr(method='spearman')
            elif method == 'kendall':
                corr_matrix = numeric_data.corr(method='kendall')
            else:
                raise ValueError(f"Unsupported correlation method: {method}")
            
            # Find strong correlations
            strong_correlations = self._find_strong_correlations(corr_matrix)
            
            # Generate correlation insights
            insights = self._generate_correlation_insights(corr_matrix, strong_correlations)
            
            # Calculate correlation statistics
            correlation_stats = self._calculate_correlation_statistics(corr_matrix)
            
            # Identify multicollinearity issues
            multicollinearity = self._detect_multicollinearity(corr_matrix)
            
            return {
                f'{prefix}correlation_matrix': corr_matrix,
                f'{prefix}strong_correlations': strong_correlations,
                f'{prefix}correlation_insights': insights,
                f'{prefix}correlation_statistics': correlation_stats,
                f'{prefix}multicollinearity_issues': multicollinearity,
                f'{prefix}method': method,
                f'{prefix}features_analyzed': list(numeric_data.columns)
            }
            
        except Exception as e:
            self.logger.error(f"Error in feature correlation analysis: {str(e)}")
            return {'error': str(e)}
    
    def analyze_target_correlations(self, data: pd.DataFrame, 
                                  target_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze correlations between features and target variables
        
        Args:
            data: Input DataFrame
            target_columns: List of target column names (auto-detect if None)
            
        Returns:
            Dictionary containing target correlation analysis
        """
        try:
            # Auto-detect potential target columns if not specified
            if target_columns is None:
                target_columns = self._detect_target_columns(data)
            
            if not target_columns:
                return {'warning': 'No target columns detected or specified'}
            
            numeric_data = data.select_dtypes(include=[np.number])
            target_correlations = {}
            
            for target in target_columns:
                if target in numeric_data.columns:
                    # Calculate correlations with target
                    feature_target_corr = numeric_data.corrwith(numeric_data[target])
                    
                    # Remove the target's correlation with itself
                    feature_target_corr = feature_target_corr.drop(target, errors='ignore')
                    
                    # Sort by absolute correlation strength
                    sorted_correlations = feature_target_corr.abs().sort_values(ascending=False)
                    
                    target_correlations[target] = {
                        'correlations': feature_target_corr.to_dict(),
                        'top_positive': feature_target_corr.nlargest(5).to_dict(),
                        'top_negative': feature_target_corr.nsmallest(5).to_dict(),
                        'strongest_overall': sorted_correlations.head(10).to_dict()
                    }
            
            return {
                'target_correlations': target_correlations,
                'targets_analyzed': target_columns,
                'correlation_summary': self._summarize_target_correlations(target_correlations)
            }
            
        except Exception as e:
            self.logger.error(f"Error in target correlation analysis: {str(e)}")
            return {'error': str(e)}
    
    def analyze_cross_correlations(self, features_data: pd.DataFrame, 
                                 predictions_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze correlations between input features and model predictions
        
        Args:
            features_data: Input features DataFrame
            predictions_data: Model predictions DataFrame
            
        Returns:
            Dictionary containing cross-correlation analysis
        """
        try:
            # Get numeric columns from both datasets
            numeric_features = features_data.select_dtypes(include=[np.number])
            numeric_predictions = predictions_data.select_dtypes(include=[np.number])
            
            cross_correlations = {}
            
            # Calculate correlations between each feature and each prediction
            for pred_col in numeric_predictions.columns:
                pred_series = numeric_predictions[pred_col]
                feature_pred_corr = numeric_features.corrwith(pred_series)
                
                cross_correlations[pred_col] = {
                    'correlations': feature_pred_corr.to_dict(),
                    'strongest_positive': feature_pred_corr.nlargest(5).to_dict(),
                    'strongest_negative': feature_pred_corr.nsmallest(5).to_dict()
                }
            
            # Overall cross-correlation summary
            cross_summary = self._summarize_cross_correlations(cross_correlations)
            
            return {
                'cross_correlations': cross_correlations,
                'cross_correlation_summary': cross_summary,
                'features_analyzed': list(numeric_features.columns),
                'predictions_analyzed': list(numeric_predictions.columns)
            }
            
        except Exception as e:
            self.logger.error(f"Error in cross-correlation analysis: {str(e)}")
            return {'error': str(e)}
    
    def analyze_model_relationships(self, model_correlations: Dict[str, Any],
                                  feature_model_correlations: Dict[str, Any]) -> List[CorrelationInsight]:
        """
        Analyze relationships between different models and generate insights
        
        Args:
            model_correlations: Correlations between model outputs
            feature_model_correlations: Correlations between features and model outputs
            
        Returns:
            List of correlation insights
        """
        insights = []
        
        try:
            # Analyze model-to-model relationships
            if 'correlation_matrix' in model_correlations:
                corr_matrix = model_correlations['correlation_matrix']
                
                # Find significant model relationships
                for i, model1 in enumerate(corr_matrix.columns):
                    for j, model2 in enumerate(corr_matrix.columns):
                        if i < j:  # Avoid duplicates and self-correlation
                            corr_value = corr_matrix.loc[model1, model2]
                            
                            if abs(corr_value) > self.strength_thresholds['moderate']:
                                strength = self._categorize_correlation_strength(abs(corr_value))
                                
                                insight = CorrelationInsight(
                                    correlation_type='model_relationship',
                                    strength=strength,
                                    variables=(model1, model2),
                                    value=corr_value,
                                    p_value=None,
                                    interpretation=self._interpret_model_correlation(
                                        model1, model2, corr_value, strength
                                    ),
                                    recommendation=self._recommend_model_action(
                                        model1, model2, corr_value, strength
                                    )
                                )
                                insights.append(insight)
            
            # Analyze feature-model relationships
            if 'cross_correlations' in feature_model_correlations:
                cross_corrs = feature_model_correlations['cross_correlations']
                
                for model_name, model_data in cross_corrs.items():
                    if 'correlations' in model_data:
                        for feature, corr_value in model_data['correlations'].items():
                            if not pd.isna(corr_value) and abs(corr_value) > self.strength_thresholds['moderate']:
                                strength = self._categorize_correlation_strength(abs(corr_value))
                                
                                insight = CorrelationInsight(
                                    correlation_type='feature_model_relationship',
                                    strength=strength,
                                    variables=(feature, model_name),
                                    value=corr_value,
                                    p_value=None,
                                    interpretation=self._interpret_feature_model_correlation(
                                        feature, model_name, corr_value, strength
                                    ),
                                    recommendation=self._recommend_feature_action(
                                        feature, model_name, corr_value, strength
                                    )
                                )
                                insights.append(insight)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error analyzing model relationships: {str(e)}")
            return []
    
    def generate_correlation_insights(self, feature_correlations: Dict[str, Any],
                                    model_feature_correlations: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Generate high-level insights from correlation analysis
        
        Args:
            feature_correlations: Feature correlation analysis results
            model_feature_correlations: Model-specific feature correlations
            
        Returns:
            Dictionary of categorized insights
        """
        insights = {
            'data_quality_insights': [],
            'feature_engineering_insights': [],
            'model_performance_insights': [],
            'business_insights': []
        }
        
        try:
            # Data quality insights
            if 'multicollinearity_issues' in feature_correlations:
                multicollinearity = feature_correlations['multicollinearity_issues']
                if multicollinearity['high_correlation_pairs']:
                    insights['data_quality_insights'].append(
                        f"Found {len(multicollinearity['high_correlation_pairs'])} highly correlated feature pairs "
                        "that may cause multicollinearity issues"
                    )
                
                if multicollinearity['redundant_features']:
                    insights['data_quality_insights'].append(
                        f"Identified {len(multicollinearity['redundant_features'])} potentially redundant features "
                        "that could be removed to reduce dimensionality"
                    )
            
            # Feature engineering insights
            if 'strong_correlations' in feature_correlations:
                strong_corrs = feature_correlations['strong_correlations']
                if strong_corrs:
                    insights['feature_engineering_insights'].append(
                        f"Found {len(strong_corrs)} strong feature correlations that could be used "
                        "for creating composite features or interaction terms"
                    )
            
            # Model-specific insights
            for model_name, model_corrs in model_feature_correlations.items():
                if 'correlation_statistics' in model_corrs:
                    stats = model_corrs['correlation_statistics']
                    avg_corr = stats.get('mean_absolute_correlation', 0)
                    
                    if avg_corr > 0.5:
                        insights['model_performance_insights'].append(
                            f"{model_name} features show strong internal correlations "
                            f"(avg: {avg_corr:.3f}), suggesting good feature relevance"
                        )
                    elif avg_corr < 0.2:
                        insights['model_performance_insights'].append(
                            f"{model_name} features show weak internal correlations "
                            f"(avg: {avg_corr:.3f}), may need feature selection review"
                        )
            
            # Business insights
            self._add_business_correlation_insights(insights, feature_correlations, model_feature_correlations)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating correlation insights: {str(e)}")
            return insights
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, 
                                threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find pairs of features with strong correlations"""
        strong_correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold:
                    strong_correlations.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value,
                        'abs_correlation': abs(corr_value),
                        'strength': self._categorize_correlation_strength(abs(corr_value))
                    })
        
        # Sort by absolute correlation strength
        strong_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
        return strong_correlations
    
    def _generate_correlation_insights(self, corr_matrix: pd.DataFrame, 
                                     strong_correlations: List[Dict[str, Any]]) -> List[str]:
        """Generate textual insights from correlation analysis"""
        insights = []
        
        # Overall correlation insights
        non_diagonal = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        mean_corr = np.mean(np.abs(non_diagonal))
        max_corr = np.max(np.abs(non_diagonal))
        
        insights.append(f"Average absolute correlation: {mean_corr:.3f}")
        insights.append(f"Maximum absolute correlation: {max_corr:.3f}")
        
        if strong_correlations:
            insights.append(f"Found {len(strong_correlations)} strong correlations (>0.7)")
            
            # Highlight top correlations
            top_3 = strong_correlations[:3]
            for i, corr in enumerate(top_3, 1):
                insights.append(
                    f"#{i}: {corr['feature1']} ↔ {corr['feature2']} "
                    f"(r={corr['correlation']:.3f}, {corr['strength']})"
                )
        else:
            insights.append("No strong correlations (>0.7) found between features")
        
        return insights
    
    def _calculate_correlation_statistics(self, corr_matrix: pd.DataFrame) -> Dict[str, float]:
        """Calculate statistical summary of correlations"""
        # Get upper triangle (excluding diagonal)
        non_diagonal = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        abs_correlations = np.abs(non_diagonal)
        
        return {
            'mean_absolute_correlation': np.mean(abs_correlations),
            'median_absolute_correlation': np.median(abs_correlations),
            'std_absolute_correlation': np.std(abs_correlations),
            'max_absolute_correlation': np.max(abs_correlations),
            'min_absolute_correlation': np.min(abs_correlations),
            'correlation_range': np.max(abs_correlations) - np.min(abs_correlations)
        }
    
    def _detect_multicollinearity(self, corr_matrix: pd.DataFrame, 
                                threshold: float = 0.8) -> Dict[str, Any]:
        """Detect potential multicollinearity issues"""
        high_corr_pairs = []
        redundant_features = set()
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = abs(corr_matrix.iloc[i, j])
                
                if corr_value >= threshold:
                    feature1, feature2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    high_corr_pairs.append({
                        'feature1': feature1,
                        'feature2': feature2,
                        'correlation': corr_matrix.iloc[i, j],
                        'abs_correlation': corr_value
                    })
                    
                    # Mark one as potentially redundant
                    redundant_features.add(feature2)
        
        return {
            'high_correlation_pairs': high_corr_pairs,
            'redundant_features': list(redundant_features),
            'multicollinearity_risk': len(high_corr_pairs) > 0,
            'severity': 'high' if len(high_corr_pairs) > 3 else 'moderate' if len(high_corr_pairs) > 0 else 'low'
        }
    
    def _detect_target_columns(self, data: pd.DataFrame) -> List[str]:
        """Auto-detect potential target columns based on naming patterns"""
        target_patterns = [
            'target', 'label', 'class', 'category', 'segment', 'group',
            'outcome', 'result', 'prediction', 'score', 'risk', 'status'
        ]
        
        potential_targets = []
        for col in data.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in target_patterns):
                potential_targets.append(col)
        
        return potential_targets
    
    def _categorize_correlation_strength(self, abs_correlation: float) -> str:
        """Categorize correlation strength based on absolute value"""
        if abs_correlation >= self.strength_thresholds['very_strong']:
            return 'very_strong'
        elif abs_correlation >= self.strength_thresholds['strong']:
            return 'strong'
        elif abs_correlation >= self.strength_thresholds['moderate']:
            return 'moderate'
        elif abs_correlation >= self.strength_thresholds['weak']:
            return 'weak'
        else:
            return 'very_weak'
    
    def _summarize_target_correlations(self, target_correlations: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize target correlation analysis"""
        summary = {}
        
        for target, target_data in target_correlations.items():
            correlations = target_data['correlations']
            abs_correlations = {k: abs(v) for k, v in correlations.items() if not pd.isna(v)}
            
            if abs_correlations:
                summary[target] = {
                    'strongest_feature': max(abs_correlations, key=abs_correlations.get),
                    'strongest_correlation': max(abs_correlations.values()),
                    'mean_absolute_correlation': np.mean(list(abs_correlations.values())),
                    'features_with_strong_correlation': len([v for v in abs_correlations.values() if v > 0.5])
                }
        
        return summary
    
    def _summarize_cross_correlations(self, cross_correlations: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize cross-correlation analysis"""
        summary = {}
        
        for model, model_data in cross_correlations.items():
            correlations = model_data['correlations']
            abs_correlations = {k: abs(v) for k, v in correlations.items() if not pd.isna(v)}
            
            if abs_correlations:
                summary[model] = {
                    'most_influential_feature': max(abs_correlations, key=abs_correlations.get),
                    'max_correlation': max(abs_correlations.values()),
                    'mean_correlation': np.mean(list(abs_correlations.values())),
                    'features_above_threshold': len([v for v in abs_correlations.values() if v > 0.3])
                }
        
        return summary
    
    def _interpret_model_correlation(self, model1: str, model2: str, 
                                   correlation: float, strength: str) -> str:
        """Interpret correlation between two models"""
        direction = "positively" if correlation > 0 else "negatively"
        
        return (f"{model1} and {model2} are {direction} correlated ({strength} correlation, "
                f"r={correlation:.3f}). This suggests that customers who score high on one "
                f"dimension tend to score {'high' if correlation > 0 else 'low'} on the other.")
    
    def _recommend_model_action(self, model1: str, model2: str, 
                              correlation: float, strength: str) -> str:
        """Recommend actions based on model correlation"""
        if strength in ['strong', 'very_strong']:
            if abs(correlation) > 0.8:
                return ("Consider whether these models are measuring overlapping constructs. "
                        "High correlation may indicate redundancy or suggest creating a composite score.")
            else:
                return ("Models show strong relationship but remain distinct. "
                        "Use this correlation for cross-validation and ensemble methods.")
        else:
            return ("Models appear to measure independent dimensions. "
                    "This diversity is valuable for comprehensive customer profiling.")
    
    def _interpret_feature_model_correlation(self, feature: str, model: str, 
                                           correlation: float, strength: str) -> str:
        """Interpret correlation between feature and model"""
        direction = "positively" if correlation > 0 else "negatively"
        
        return (f"Feature '{feature}' is {direction} correlated with {model} "
                f"({strength} correlation, r={correlation:.3f}). This feature "
                f"{'strongly influences' if abs(correlation) > 0.6 else 'moderately affects'} "
                f"the model's predictions.")
    
    def _recommend_feature_action(self, feature: str, model: str, 
                                correlation: float, strength: str) -> str:
        """Recommend actions based on feature-model correlation"""
        if strength in ['strong', 'very_strong']:
            return (f"Feature '{feature}' is a key driver for {model}. "
                    f"Ensure data quality and consider feature engineering opportunities.")
        elif strength == 'moderate':
            return (f"Feature '{feature}' has moderate influence on {model}. "
                    f"Monitor its importance and consider interaction terms.")
        else:
            return (f"Feature '{feature}' has weak influence on {model}. "
                    f"Consider removal if not important for other models.")
    
    def _add_business_correlation_insights(self, insights: Dict[str, List[str]], 
                                         feature_correlations: Dict[str, Any],
                                         model_feature_correlations: Dict[str, Any]) -> None:
        """Add business-specific correlation insights"""
        # Look for business-relevant patterns
        business_patterns = {
            'income': ['financial_capability', 'spending', 'credit'],
            'debt': ['financial_hardship', 'risk', 'payment'],
            'gambling': ['risk', 'behavior', 'frequency'],
            'credit': ['score', 'rating', 'risk'],
            'payment': ['delay', 'default', 'history']
        }
        
        # Check for expected business correlations
        if 'strong_correlations' in feature_correlations:
            strong_corrs = feature_correlations['strong_correlations']
            
            for corr in strong_corrs:
                feature1_lower = corr['feature1'].lower()
                feature2_lower = corr['feature2'].lower()
                
                for business_concept, keywords in business_patterns.items():
                    if (any(keyword in feature1_lower for keyword in keywords) and 
                        any(keyword in feature2_lower for keyword in keywords)):
                        
                        insights['business_insights'].append(
                            f"Strong correlation between {corr['feature1']} and {corr['feature2']} "
                            f"aligns with expected {business_concept}-related patterns "
                            f"(r={corr['correlation']:.3f})"
                        )
                        break
        
        # Add general business insights
        if len(insights['business_insights']) == 0:
            insights['business_insights'].append(
                "Correlation patterns suggest diverse customer characteristics across multiple dimensions"
            )
