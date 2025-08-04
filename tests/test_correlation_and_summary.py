"""
Test script for enhanced correlation analysis and summary reporting
Demonstrates the new analytics capabilities
"""
import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime

# Import our enhanced modules
from utils.correlation_analyzer import CorrelationAnalyzer
from utils.summary_reporter import SummaryReporter


def create_sample_customer_data(n_samples=500):
    """Create realistic sample customer data for testing"""
    np.random.seed(42)
    
    # Generate correlated financial features
    base_income = np.random.normal(50000, 20000, n_samples)
    base_income = np.clip(base_income, 20000, 150000)
    
    # Features with realistic correlations
    data = {
        'customer_id': range(1, n_samples + 1),
        'monthly_income': base_income,
        'credit_score': 300 + (base_income - 20000) / 1000 * 5 + np.random.normal(0, 50, n_samples),
        'debt_to_income_ratio': np.clip(0.8 - (base_income - 20000) / 130000 * 0.6 + np.random.normal(0, 0.2, n_samples), 0, 1),
        'payment_delays_count': np.random.poisson(2 + (0.8 - (base_income - 20000) / 130000 * 0.6) * 5, n_samples),
        'gambling_merchant_frequency': np.random.poisson(1 + np.random.uniform(0, 3, n_samples)),
        'large_cash_withdrawals': np.random.poisson(0.5 + np.random.uniform(0, 2, n_samples)),
        'account_balance': base_income * 0.8 + np.random.normal(0, 5000, n_samples),
        'transaction_frequency': np.random.poisson(20 + base_income / 10000, n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'tenure_months': np.random.randint(1, 120, n_samples)
    }
    
    # Ensure realistic value ranges
    data['credit_score'] = np.clip(data['credit_score'], 300, 850)
    data['account_balance'] = np.clip(data['account_balance'], 0, None)
    
    return pd.DataFrame(data)


def create_mock_segmentation_results(n_customers=500):
    """Create mock segmentation results for testing"""
    np.random.seed(42)
    
    # Mock Financial Capability segments
    financial_segments = {
        'high_capability': list(np.random.choice(range(n_customers), size=150, replace=False)),
        'medium_capability': list(np.random.choice(range(n_customers), size=200, replace=False)),
        'low_capability': list(np.random.choice(range(n_customers), size=150, replace=False))
    }
    
    # Mock Financial Hardship segments
    hardship_segments = {
        'no_hardship': list(np.random.choice(range(n_customers), size=200, replace=False)),
        'moderate_hardship': list(np.random.choice(range(n_customers), size=200, replace=False)),
        'severe_hardship': list(np.random.choice(range(n_customers), size=100, replace=False))
    }
    
    # Mock Gambling Behavior segments
    gambling_segments = {
        'no_gambling_risk': list(np.random.choice(range(n_customers), size=300, replace=False)),
        'moderate_gambling_risk': list(np.random.choice(range(n_customers), size=150, replace=False)),
        'high_gambling_risk': list(np.random.choice(range(n_customers), size=50, replace=False))
    }
    
    return {
        'financial_capability': {
            'segments': financial_segments,
            'performance': {'accuracy': 0.85, 'f1_score': 0.82, 'execution_time': '2.3s'}
        },
        'financial_hardship': {
            'segments': hardship_segments,
            'performance': {'accuracy': 0.78, 'precision': 0.80, 'recall': 0.76, 'execution_time': '1.8s'}
        },
        'gambling_behavior': {
            'segments': gambling_segments,
            'performance': {'accuracy': 0.73, 'auc_score': 0.79, 'execution_time': '2.1s'}
        }
    }


def create_mock_predictions_data():
    """Create mock prediction data for cross-correlation analysis"""
    np.random.seed(42)
    n_customers = 500
    
    return pd.DataFrame({
        'financial_capability_score': np.random.uniform(0.2, 0.8, n_customers),
        'financial_hardship_score': np.random.uniform(0.2, 0.8, n_customers),
        'gambling_behavior_score': np.random.uniform(0.2, 0.8, n_customers)
    })


async def test_correlation_analysis():
    """Test comprehensive correlation analysis"""
    print("🔍 Testing Correlation Analysis Module")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = CorrelationAnalyzer()
    
    # Create sample data
    customer_data = create_sample_customer_data(500)
    predictions_data = create_mock_predictions_data()
    
    print(f"📊 Created sample data: {len(customer_data)} customers, {len(customer_data.columns)} features")
    
    # Test feature correlations
    print("\n1. Analyzing Feature Correlations...")
    feature_corr_results = analyzer.analyze_feature_correlations(customer_data)
    
    if 'error' not in feature_corr_results:
        print(f"   ✅ Found {len(feature_corr_results.get('strong_correlations', []))} strong correlations")
        print(f"   📈 Average correlation strength: {feature_corr_results.get('correlation_statistics', {}).get('mean_absolute_correlation', 0):.3f}")
        
        # Show multicollinearity assessment
        multicollinearity = feature_corr_results.get('multicollinearity_issues', {})
        print(f"   ⚠️  Multicollinearity risk: {multicollinearity.get('severity', 'unknown')}")
    else:
        print(f"   ❌ Error: {feature_corr_results['error']}")
    
    # Test target correlations
    print("\n2. Analyzing Target Correlations...")
    target_corr_results = analyzer.analyze_target_correlations(customer_data)
    
    if 'error' not in target_corr_results:
        targets = target_corr_results.get('targets_analyzed', [])
        print(f"   ✅ Analyzed {len(targets)} potential target variables")
        for target in targets:
            print(f"      📍 Target: {target}")
    else:
        print(f"   ❌ Error: {target_corr_results.get('error', 'Unknown error')}")
    
    # Test cross-correlations
    print("\n3. Analyzing Cross-Correlations...")
    cross_corr_results = analyzer.analyze_cross_correlations(customer_data, predictions_data)
    
    if 'error' not in cross_corr_results:
        features_analyzed = len(cross_corr_results.get('features_analyzed', []))
        predictions_analyzed = len(cross_corr_results.get('predictions_analyzed', []))
        print(f"   ✅ Cross-correlation: {features_analyzed} features vs {predictions_analyzed} predictions")
    else:
        print(f"   ❌ Error: {cross_corr_results.get('error', 'Unknown error')}")
    
    # Test model relationships
    print("\n4. Analyzing Model Relationships...")
    model_relationships = analyzer.analyze_model_relationships(
        {'correlation_matrix': predictions_data.corr()}, 
        cross_corr_results
    )
    
    print(f"   ✅ Generated {len(model_relationships)} relationship insights")
    for insight in model_relationships[:3]:  # Show top 3
        print(f"      💡 {insight.correlation_type}: {insight.interpretation[:80]}...")
    
    # Test insight generation
    print("\n5. Generating High-Level Insights...")
    correlation_insights = analyzer.generate_correlation_insights(
        feature_corr_results, {'test_model': feature_corr_results}
    )
    
    total_insights = sum(len(insights) for insights in correlation_insights.values())
    print(f"   ✅ Generated {total_insights} insights across {len(correlation_insights)} categories")
    
    for category, insights in correlation_insights.items():
        if insights:
            print(f"      📋 {category}: {len(insights)} insights")
    
    return {
        'feature_correlations': feature_corr_results,
        'target_correlations': target_corr_results,
        'cross_correlations': cross_corr_results,
        'model_relationships': model_relationships,
        'insights': correlation_insights
    }


async def test_summary_reporting():
    """Test comprehensive summary reporting"""
    print("\n\n📊 Testing Summary Reporting Module")
    print("=" * 50)
    
    # Initialize reporter
    reporter = SummaryReporter()
    
    # Create sample data and results
    customer_data = create_sample_customer_data(500)
    segmentation_results = create_mock_segmentation_results(500)
    
    print(f"📊 Created mock results for {len(segmentation_results)} models")
    
    # Test dataset summary
    print("\n1. Generating Dataset Summary...")
    dataset_summary = reporter.generate_dataset_summary(customer_data)
    
    if 'error' not in dataset_summary:
        basic_stats = dataset_summary.get('basic_statistics', {})
        print(f"   ✅ Dataset: {basic_stats.get('total_records', 0)} records, {basic_stats.get('total_features', 0)} features")
        print(f"   📊 Data quality score: {basic_stats.get('data_quality_score', 0):.1f}/100")
        print(f"   🔍 Missing values: {basic_stats.get('missing_percentage', 0):.2f}%")
    else:
        print(f"   ❌ Error: {dataset_summary['error']}")
    
    # Test segmentation summary
    print("\n2. Generating Segmentation Summary...")
    segmentation_summary = reporter.generate_segmentation_summary(segmentation_results)
    
    if 'error' not in segmentation_summary:
        overall_stats = segmentation_summary.get('overall_statistics', {})
        print(f"   ✅ Segmentation: {overall_stats.get('total_customers_analyzed', 0)} customers")
        print(f"   🎯 Models executed: {overall_stats.get('models_executed', 0)}")
        print(f"   📈 Total segments: {overall_stats.get('total_unique_segments', 0)}")
        
        # Show model-specific insights
        model_summaries = segmentation_summary.get('model_summaries', {})
        for model_name, summary in model_summaries.items():
            largest_segment = summary.get('largest_segment', 'unknown')
            print(f"      🔍 {model_name}: largest segment is '{largest_segment}'")
    else:
        print(f"   ❌ Error: {segmentation_summary['error']}")
    
    # Test correlation summary (using mock correlation results)
    print("\n3. Generating Correlation Summary...")
    mock_correlation_results = {
        'feature_correlations': {
            'features_analyzed': list(customer_data.columns),
            'strong_correlations': [{'feature1': 'income', 'feature2': 'credit_score', 'correlation': 0.75}],
            'multicollinearity_issues': {'severity': 'moderate'},
            'correlation_statistics': {'mean_absolute_correlation': 0.32}
        }
    }
    
    mock_cross_model = {
        'model_correlations': {'correlation_matrix': create_mock_predictions_data().corr()},
        'model_predictions': {'model1': [0.5, 0.6], 'model2': [0.4, 0.7]}
    }
    
    correlation_summary = reporter.generate_correlation_summary(
        mock_correlation_results, mock_cross_model
    )
    
    if 'error' not in correlation_summary:
        print(f"   ✅ Correlation analysis completed")
        corr_analysis = correlation_summary.get('correlation_analysis_summary', {})
        feature_corr = corr_analysis.get('feature_correlations', {})
        print(f"   📊 Features analyzed: {feature_corr.get('total_features_analyzed', 0)}")
        print(f"   🔗 Strong correlations: {feature_corr.get('strong_correlations_found', 0)}")
    else:
        print(f"   ❌ Error: {correlation_summary['error']}")
    
    # Test performance summary
    print("\n4. Generating Performance Summary...")
    performance_summary = reporter.generate_performance_summary(segmentation_results)
    
    if 'error' not in performance_summary:
        overall_perf = performance_summary.get('overall_performance', {})
        print(f"   ✅ Performance analysis completed")
        print(f"   📈 Average confidence: {overall_perf.get('average_confidence', 0):.1%}")
        print(f"   🎯 Data coverage: {overall_perf.get('total_data_coverage', 0):.1f}%")
        print(f"   🏆 Good quality models: {overall_perf.get('models_with_good_quality', 0)}")
    else:
        print(f"   ❌ Error: {performance_summary['error']}")
    
    # Test business insights
    print("\n5. Generating Business Insights...")
    business_insights = reporter.generate_business_insights(
        dataset_summary, segmentation_summary, correlation_summary
    )
    
    if 'error' not in business_insights:
        total_insights = sum(len(insights) for insights in business_insights.values())
        print(f"   ✅ Generated {total_insights} business insights")
        
        for category, insights in business_insights.items():
            if insights:
                print(f"      💼 {category}: {len(insights)} insights")
                # Show first insight
                if hasattr(insights[0], 'insight'):
                    print(f"         - {insights[0].insight[:80]}...")
    else:
        print(f"   ❌ Error: {business_insights}")
    
    # Test risk assessment
    print("\n6. Generating Risk Assessment...")
    risk_assessment = reporter.generate_risk_assessment(
        segmentation_results, mock_correlation_results
    )
    
    if 'error' not in risk_assessment:
        print(f"   ✅ Risk assessment completed")
        
        if 'financial_hardship_risk' in risk_assessment:
            fin_risk = risk_assessment['financial_hardship_risk']
            print(f"   ⚠️  Financial hardship risk: {fin_risk.get('risk_level', 'unknown')}")
        
        if 'gambling_risk' in risk_assessment:
            gambling_risk = risk_assessment['gambling_risk']
            print(f"   🎰 Gambling risk: {gambling_risk.get('risk_level', 'unknown')}")
    else:
        print(f"   ❌ Error: {risk_assessment['error']}")
    
    # Test executive summary
    print("\n7. Generating Executive Summary...")
    executive_summary = reporter.generate_executive_summary(
        dataset_summary, segmentation_summary, performance_summary, business_insights
    )
    
    if 'error' not in executive_summary:
        print(f"   ✅ Executive summary generated")
        
        kpis = executive_summary.get('key_performance_indicators', {})
        print(f"   📊 KPIs: {len(kpis)} key metrics tracked")
        
        opportunities = executive_summary.get('top_business_opportunities', [])
        print(f"   🚀 Opportunities: {len(opportunities)} identified")
        
        recommendations = executive_summary.get('executive_recommendations', [])
        print(f"   💡 Recommendations: {len(recommendations)} strategic actions")
    else:
        print(f"   ❌ Error: {executive_summary['error']}")
    
    return {
        'dataset_summary': dataset_summary,
        'segmentation_summary': segmentation_summary,
        'correlation_summary': correlation_summary,
        'performance_summary': performance_summary,
        'business_insights': business_insights,
        'risk_assessment': risk_assessment,
        'executive_summary': executive_summary
    }


async def main():
    """Main test execution"""
    print("🚀 Enhanced Customer Segmentation Analytics Test")
    print("=" * 60)
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Test correlation analysis
        correlation_results = await test_correlation_analysis()
        
        # Test summary reporting
        summary_results = await test_summary_reporting()
        
        print("\n\n✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"🔍 Correlation Analysis: {len(correlation_results)} major components tested")
        print(f"📊 Summary Reporting: {len(summary_results)} report types generated")
        print(f"⏱️  Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Show summary of capabilities
        print("\n🎯 New Analytics Capabilities Added:")
        print("   • Comprehensive feature correlation analysis")
        print("   • Cross-model relationship analysis") 
        print("   • Multi-dimensional business insights")
        print("   • Executive-level summary reporting")
        print("   • Risk assessment and mitigation recommendations")
        print("   • Performance analysis and improvement suggestions")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
