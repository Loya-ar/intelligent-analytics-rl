"""
Custom Analytics Tools for Enhanced Agent Capabilities
Specialized tools that extend agent analysis capabilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedCorrelationTool:
    """
    Advanced correlation analysis beyond basic Pearson correlation
    """
    
    def __init__(self):
        self.name = "Advanced Correlation Analysis"
        self.description = "Multi-method correlation analysis with statistical significance"
    
    def analyze(self, data, target_column=None, significance_level=0.05):
        """
        Perform comprehensive correlation analysis
        """
        print(f"\n🔬 {self.name}")
        print("=" * 40)
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {'error': 'Need at least 2 numeric columns'}
        
        results = {
            'tool': self.name,
            'correlations': {},
            'significance_tests': {},
            'recommendations': []
        }
        
        # Multiple correlation methods
        correlation_methods = {
            'pearson': 'linear relationships',
            'spearman': 'monotonic relationships', 
            'kendall': 'rank-based relationships'
        }
        
        for method, description in correlation_methods.items():
            print(f"\n📊 {method.title()} Correlation ({description}):")
            
            if method == 'pearson':
                corr_matrix = data[numeric_cols].corr(method='pearson')
            elif method == 'spearman':
                corr_matrix = data[numeric_cols].corr(method='spearman')
            else:  # kendall
                corr_matrix = data[numeric_cols].corr(method='kendall')
            
            # Find significant correlations
            significant_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]
                    
                    if abs(corr_val) > 0.3:  # Moderate correlation threshold
                        # Statistical significance test
                        if method == 'pearson':
                            stat, p_value = stats.pearsonr(data[col1].dropna(), data[col2].dropna())
                        elif method == 'spearman':
                            stat, p_value = stats.spearmanr(data[col1].dropna(), data[col2].dropna())
                        else:  # kendall
                            stat, p_value = stats.kendalltau(data[col1].dropna(), data[col2].dropna())
                        
                        is_significant = p_value < significance_level
                        
                        significant_pairs.append({
                            'variables': (col1, col2),
                            'correlation': corr_val,
                            'p_value': p_value,
                            'significant': is_significant
                        })
                        
                        if is_significant:
                            direction = "positive" if corr_val > 0 else "negative"
                            print(f"  🔗 {col1} ↔ {col2}: {corr_val:.3f} ({direction}, p={p_value:.4f})")
            
            results['correlations'][method] = significant_pairs
        
        # Generate recommendations
        all_significant = []
        for method_results in results['correlations'].values():
            all_significant.extend([pair for pair in method_results if pair['significant']])
        
        if all_significant:
            strongest = max(all_significant, key=lambda x: abs(x['correlation']))
            results['recommendations'].append(f"Strongest relationship: {strongest['variables'][0]} ↔ {strongest['variables'][1]} (r={strongest['correlation']:.3f})")
            
            # Count variables involved in multiple relationships
            variable_counts = {}
            for pair in all_significant:
                for var in pair['variables']:
                    variable_counts[var] = variable_counts.get(var, 0) + 1
            
            if variable_counts:
                most_connected = max(variable_counts.items(), key=lambda x: x[1])
                if most_connected[1] > 1:
                    results['recommendations'].append(f"Most connected variable: {most_connected[0]} ({most_connected[1]} significant relationships)")
        
        print(f"\n💡 Found {len(all_significant)} statistically significant correlations")
        
        return results

class SmartClusteringTool:
    """
    Intelligent clustering with automatic parameter optimization
    """
    
    def __init__(self):
        self.name = "Smart Clustering Analysis"
        self.description = "Automated clustering with optimal parameter selection"
    
    def analyze(self, data, max_clusters=8, min_clusters=2):
        """
        Perform intelligent clustering analysis
        """
        print(f"\n🎯 {self.name}")
        print("=" * 40)
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {'error': 'Need at least 2 numeric columns for clustering'}
        
        # Prepare data
        clustering_data = data[numeric_cols].dropna()
        if len(clustering_data) < 10:
            return {'error': 'Need at least 10 complete records for clustering'}
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clustering_data)
        
        results = {
            'tool': self.name,
            'optimal_clusters': None,
            'cluster_profiles': {},
            'recommendations': [],
            'silhouette_scores': {}
        }
        
        # Find optimal number of clusters
        print("🔍 Finding optimal number of clusters...")
        
        silhouette_scores = {}
        inertias = []
        
        cluster_range = range(min_clusters, min(max_clusters + 1, len(clustering_data) // 2))
        
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            # Calculate silhouette score
            sil_score = silhouette_score(scaled_data, cluster_labels)
            silhouette_scores[n_clusters] = sil_score
            inertias.append(kmeans.inertia_)
            
            print(f"  {n_clusters} clusters: silhouette score = {sil_score:.3f}")
        
        # Select optimal clusters (highest silhouette score)
        optimal_k = max(silhouette_scores.items(), key=lambda x: x[1])[0]
        results['optimal_clusters'] = optimal_k
        results['silhouette_scores'] = silhouette_scores
        
        print(f"🎯 Optimal clusters: {optimal_k} (silhouette score: {silhouette_scores[optimal_k]:.3f})")
        
        # Perform final clustering
        final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        final_labels = final_kmeans.fit_predict(scaled_data)
        
        # Add cluster labels to original data
        clustered_data = clustering_data.copy()
        clustered_data['Cluster'] = final_labels
        
        # Analyze cluster profiles
        print(f"\n📊 Cluster Profiles:")
        
        for cluster_id in range(optimal_k):
            cluster_data = clustered_data[clustered_data['Cluster'] == cluster_id]
            cluster_size = len(cluster_data)
            cluster_percentage = (cluster_size / len(clustered_data)) * 100
            
            print(f"\n  🔸 Cluster {cluster_id + 1} ({cluster_size} records, {cluster_percentage:.1f}%):")
            
            # Calculate cluster characteristics
            cluster_profile = {}
            for col in numeric_cols:
                col_mean = cluster_data[col].mean()
                overall_mean = clustered_data[col].mean()
                deviation = ((col_mean - overall_mean) / overall_mean) * 100
                
                cluster_profile[col] = {
                    'mean': col_mean,
                    'deviation_from_average': deviation
                }
                
                if abs(deviation) > 20:  # Significant deviation
                    direction = "higher" if deviation > 0 else "lower"
                    print(f"    • {col}: {direction} than average ({deviation:+.1f}%)")
            
            results['cluster_profiles'][f'cluster_{cluster_id + 1}'] = {
                'size': cluster_size,
                'percentage': cluster_percentage,
                'characteristics': cluster_profile
            }
        
        # Generate recommendations
        if optimal_k > 1:
            results['recommendations'].append(f"Data naturally segments into {optimal_k} distinct groups")
            
            # Identify most distinctive clusters
            max_deviation_cluster = None
            max_deviation = 0
            
            for cluster_name, profile in results['cluster_profiles'].items():
                for col, stats in profile['characteristics'].items():
                    deviation = abs(stats['deviation_from_average'])
                    if deviation > max_deviation:
                        max_deviation = deviation
                        max_deviation_cluster = cluster_name
            
            if max_deviation_cluster and max_deviation > 30:
                results['recommendations'].append(f"Most distinctive segment: {max_deviation_cluster} (requires specialized strategy)")
        
        return results

class TimeSeriesInsightTool:
    """
    Advanced time series analysis for temporal patterns
    """
    
    def __init__(self):
        self.name = "Time Series Insight Analysis"
        self.description = "Advanced temporal pattern discovery and forecasting"
    
    def analyze(self, data, date_column=None, value_column=None, seasonal_periods=[7, 30, 365]):
        """
        Perform comprehensive time series analysis
        """
        print(f"\n📈 {self.name}")
        print("=" * 40)
        
        # Auto-detect date and value columns
        if date_column is None:
            date_candidates = [col for col in data.columns 
                             if 'date' in col.lower() or 'time' in col.lower()]
            if date_candidates:
                date_column = date_candidates[0]
            elif 'month' in data.columns and 'year' in data.columns:
                # Create date column from month/year
                data = data.copy()
                data['constructed_date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))
                date_column = 'constructed_date'
            else:
                return {'error': 'No date/time column found'}
        
        if value_column is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            value_candidates = [col for col in numeric_cols 
                              if any(keyword in col.lower() for keyword in ['sales', 'revenue', 'total', 'amount'])]
            if value_candidates:
                value_column = value_candidates[0]
            elif len(numeric_cols) > 0:
                value_column = numeric_cols[0]
            else:
                return {'error': 'No numeric value column found'}
        
        print(f"📊 Analyzing {value_column} over {date_column}")
        
        # Prepare time series data
        try:
            if date_column != 'constructed_date':
                data[date_column] = pd.to_datetime(data[date_column])
            
            ts_data = data[[date_column, value_column]].copy()
            ts_data = ts_data.dropna().sort_values(date_column)
            
            if len(ts_data) < 10:
                return {'error': 'Need at least 10 time points for analysis'}
            
        except Exception as e:
            return {'error': f'Date parsing error: {str(e)}'}
        
        results = {
            'tool': self.name,
            'trend_analysis': {},
            'seasonality_analysis': {},
            'anomaly_detection': {},
            'recommendations': []
        }
        
        # Trend Analysis
        print("\n📈 Trend Analysis:")
        
        values = ts_data[value_column].values
        time_points = np.arange(len(values))
        
        # Linear trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_points, values)
        
        trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        trend_strength = abs(r_value)
        
        print(f"  📊 Overall trend: {trend_direction}")
        print(f"  📊 Trend strength: {trend_strength:.3f} (R²={r_value**2:.3f})")
        print(f"  📊 Statistical significance: p={p_value:.4f}")
        
        results['trend_analysis'] = {
            'direction': trend_direction,
            'slope': slope,
            'strength': trend_strength,
            'r_squared': r_value**2,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        # Seasonality Analysis (if enough data)
        if len(ts_data) >= 24:  # Need sufficient data for seasonality
            print("\n🔄 Seasonality Analysis:")
            
            # Extract time features
            if date_column == 'constructed_date' or 'month' in data.columns:
                # Use month data
                if 'month' in data.columns:
                    seasonal_data = data.groupby('month')[value_column].agg(['mean', 'std', 'count']).round(2)
                    seasonal_period = 'monthly'
                else:
                    ts_data['month'] = ts_data[date_column].dt.month
                    seasonal_data = ts_data.groupby('month')[value_column].agg(['mean', 'std', 'count']).round(2)
                    seasonal_period = 'monthly'
                
                # Find peak and trough months
                peak_month = seasonal_data['mean'].idxmax()
                trough_month = seasonal_data['mean'].idxmin()
                
                seasonal_variation = (seasonal_data['mean'].max() - seasonal_data['mean'].min()) / seasonal_data['mean'].mean() * 100
                
                print(f"  🔄 Seasonality detected: {seasonal_period}")
                print(f"  📈 Peak period: Month {peak_month} (avg: {seasonal_data.loc[peak_month, 'mean']:.2f})")
                print(f"  📉 Trough period: Month {trough_month} (avg: {seasonal_data.loc[trough_month, 'mean']:.2f})")
                print(f"  📊 Seasonal variation: {seasonal_variation:.1f}%")
                
                results['seasonality_analysis'] = {
                    'detected': True,
                    'period': seasonal_period,
                    'peak_period': peak_month,
                    'trough_period': trough_month,
                    'variation_percentage': seasonal_variation,
                    'seasonal_data': seasonal_data.to_dict()
                }
            else:
                results['seasonality_analysis'] = {'detected': False, 'reason': 'Insufficient temporal resolution'}
        
        # Simple Anomaly Detection
        print("\n🚨 Anomaly Detection:")
        
        # Statistical outliers (Z-score method)
        z_scores = np.abs(stats.zscore(values))
        outlier_threshold = 2.5
        outliers = np.where(z_scores > outlier_threshold)[0]
        
        if len(outliers) > 0:
            print(f"  🚨 Found {len(outliers)} potential anomalies")
            
            anomaly_dates = ts_data.iloc[outliers][date_column].tolist()
            anomaly_values = ts_data.iloc[outliers][value_column].tolist()
            
            results['anomaly_detection'] = {
                'anomalies_found': len(outliers),
                'anomaly_dates': [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in anomaly_dates],
                'anomaly_values': anomaly_values,
                'threshold_used': outlier_threshold
            }
            
            for i, (date, value) in enumerate(zip(anomaly_dates[:3], anomaly_values[:3])):  # Show first 3
                print(f"    • {date}: {value:.2f}")
        else:
            print("  ✅ No significant anomalies detected")
            results['anomaly_detection'] = {'anomalies_found': 0}
        
        # Generate recommendations
        if results['trend_analysis']['significant']:
            if results['trend_analysis']['direction'] == 'increasing':
                results['recommendations'].append("Strong upward trend detected - consider capacity planning")
            elif results['trend_analysis']['direction'] == 'decreasing':
                results['recommendations'].append("Downward trend detected - investigate root causes")
        
        if results['seasonality_analysis'].get('detected'):
            peak = results['seasonality_analysis']['peak_period']
            results['recommendations'].append(f"Strong seasonality - optimize resources for peak period (Month {peak})")
        
        if results['anomaly_detection']['anomalies_found'] > 0:
            results['recommendations'].append("Anomalies detected - investigate unusual events during flagged periods")
        
        return results

class BusinessMetricsTool:
    """
    Business-specific metrics and KPI calculations
    """
    
    def __init__(self):
        self.name = "Business Metrics Calculator"
        self.description = "Calculate key business performance indicators"
    
    def analyze(self, data, metrics_type='auto'):
        """
        Calculate relevant business metrics based on data characteristics
        """
        print(f"\n💼 {self.name}")
        print("=" * 40)
        
        results = {
            'tool': self.name,
            'metrics_calculated': {},
            'benchmarks': {},
            'recommendations': []
        }
        
        # Auto-detect business domain
        col_names = ' '.join(data.columns).lower()
        
        if any(word in col_names for word in ['sales', 'revenue', 'customer']):
            domain = 'sales'
        elif any(word in col_names for word in ['campaign', 'click', 'impression', 'conversion']):
            domain = 'marketing'
        elif any(word in col_names for word in ['employee', 'salary', 'performance']):
            domain = 'hr'
        else:
            domain = 'general'
        
        print(f"📊 Detected domain: {domain.title()}")
        
        if domain == 'sales':
            return self._calculate_sales_metrics(data, results)
        elif domain == 'marketing':
            return self._calculate_marketing_metrics(data, results)
        elif domain == 'hr':
            return self._calculate_hr_metrics(data, results)
        else:
            return self._calculate_general_metrics(data, results)
    
    def _calculate_sales_metrics(self, data, results):
        """Calculate sales-specific metrics"""
        
        metrics = {}
        
        # Revenue metrics
        if 'total_sales' in data.columns:
            total_revenue = data['total_sales'].sum()
            avg_transaction = data['total_sales'].mean()
            
            metrics['total_revenue'] = total_revenue
            metrics['average_transaction_value'] = avg_transaction
            
            print(f"💰 Total Revenue: ${total_revenue:,.2f}")
            print(f"💰 Average Transaction: ${avg_transaction:.2f}")
            
            # Revenue distribution analysis
            revenue_std = data['total_sales'].std()
            coefficient_of_variation = revenue_std / avg_transaction
            metrics['revenue_consistency'] = 1 / (1 + coefficient_of_variation)  # Higher = more consistent
            
            print(f"📊 Revenue Consistency Score: {metrics['revenue_consistency']:.3f}")
        
        # Regional performance
        if 'region' in data.columns and 'total_sales' in data.columns:
            regional_performance = data.groupby('region')['total_sales'].agg(['sum', 'mean', 'count'])
            
            best_region = regional_performance['sum'].idxmax()
            worst_region = regional_performance['sum'].idxmin()
            
            metrics['best_performing_region'] = best_region
            metrics['worst_performing_region'] = worst_region
            metrics['regional_variance'] = regional_performance['sum'].var()
            
            print(f"🏆 Best Region: {best_region}")
            print(f"⚠️  Worst Region: {worst_region}")
        
        # Customer metrics
        if 'customer_segment' in data.columns:
            segment_performance = data.groupby('customer_segment')['total_sales'].agg(['sum', 'mean', 'count'])
            top_segment = segment_performance['sum'].idxmax()
            
            metrics['top_customer_segment'] = top_segment
            print(f"👥 Top Customer Segment: {top_segment}")
        
        results['metrics_calculated'] = metrics
        
        # Recommendations
        if metrics.get('revenue_consistency', 0) < 0.5:
            results['recommendations'].append("High revenue variability - consider more consistent pricing strategies")
        
        if 'regional_variance' in metrics and metrics['regional_variance'] > metrics.get('total_revenue', 0) * 0.1:
            results['recommendations'].append("Significant regional performance gaps - focus on underperforming regions")
        
        return results
    
    def _calculate_marketing_metrics(self, data, results):
        """Calculate marketing-specific metrics"""
        
        metrics = {}
        
        # Conversion funnel metrics
        if all(col in data.columns for col in ['impressions', 'clicks', 'conversions']):
            total_impressions = data['impressions'].sum()
            total_clicks = data['clicks'].sum()
            total_conversions = data['conversions'].sum()
            
            ctr = (total_clicks / total_impressions) * 100 if total_impressions > 0 else 0
            conversion_rate = (total_conversions / total_clicks) * 100 if total_clicks > 0 else 0
            
            metrics['click_through_rate'] = ctr
            metrics['conversion_rate'] = conversion_rate
            metrics['total_impressions'] = total_impressions
            metrics['total_conversions'] = total_conversions
            
            print(f"👆 Click-Through Rate: {ctr:.2f}%")
            print(f"🎯 Conversion Rate: {conversion_rate:.2f}%")
            
            # Benchmark against industry standards
            industry_ctr_benchmark = 2.5  # Industry average
            industry_conv_benchmark = 10.0  # Industry average
            
            results['benchmarks'] = {
                'ctr_vs_industry': ctr / industry_ctr_benchmark if industry_ctr_benchmark > 0 else 0,
                'conversion_vs_industry': conversion_rate / industry_conv_benchmark if industry_conv_benchmark > 0 else 0
            }
        
        # ROI metrics
        if all(col in data.columns for col in ['cost', 'revenue']):
            total_cost = data['cost'].sum()
            total_revenue = data['revenue'].sum()
            
            roi = ((total_revenue - total_cost) / total_cost) * 100 if total_cost > 0 else 0
            roas = total_revenue / total_cost if total_cost > 0 else 0
            
            metrics['return_on_investment'] = roi
            metrics['return_on_ad_spend'] = roas
            
            print(f"💰 ROI: {roi:.1f}%")
            print(f"💰 ROAS: {roas:.2f}x")
        
        results['metrics_calculated'] = metrics
        
        # Recommendations
        if metrics.get('click_through_rate', 0) < 2.0:
            results['recommendations'].append("Low CTR - optimize ad creative and targeting")
        
        if metrics.get('conversion_rate', 0) < 8.0:
            results['recommendations'].append("Low conversion rate - improve landing page experience")
        
        if metrics.get('return_on_ad_spend', 0) < 3.0:
            results['recommendations'].append("Low ROAS - review campaign efficiency and targeting")
        
        return results
    
    def _calculate_hr_metrics(self, data, results):
        """Calculate HR-specific metrics"""
        
        metrics = {}
        
        # Performance metrics
        if 'performance_score' in data.columns:
            avg_performance = data['performance_score'].mean()
            performance_distribution = data['performance_score'].value_counts().sort_index()
            
            high_performers = len(data[data['performance_score'] >= 8])
            total_employees = len(data)
            high_performer_rate = (high_performers / total_employees) * 100
            
            metrics['average_performance_score'] = avg_performance
            metrics['high_performer_percentage'] = high_performer_rate
            
            print(f"📊 Average Performance Score: {avg_performance:.2f}/10")
            print(f"⭐ High Performers: {high_performer_rate:.1f}%")
        
        # Compensation analysis
        if 'salary' in data.columns and 'department' in data.columns:
            dept_compensation = data.groupby('department')['salary'].agg(['mean', 'median', 'std'])
            
            highest_paid_dept = dept_compensation['mean'].idxmax()
            salary_equity_score = 1 - (dept_compensation['std'].sum() / dept_compensation['mean'].sum())
            
            metrics['highest_compensated_department'] = highest_paid_dept
            metrics['compensation_equity_score'] = salary_equity_score
            
            print(f"💰 Highest Compensated Dept: {highest_paid_dept}")
            print(f"⚖️  Compensation Equity Score: {salary_equity_score:.3f}")
        
        results['metrics_calculated'] = metrics
        
        # Recommendations
        if metrics.get('high_performer_percentage', 0) < 20:
            results['recommendations'].append("Low high-performer rate - review performance management strategies")
        
        if metrics.get('compensation_equity_score', 1) < 0.8:
            results['recommendations'].append("Compensation inequity detected - review salary structure")
        
        return results
    
    def _calculate_general_metrics(self, data, results):
        """Calculate general data quality and distribution metrics"""
        
        metrics = {}
        
        # Data quality metrics
        total_records = len(data)
        missing_data_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        
        metrics['total_records'] = total_records
        metrics['data_completeness'] = 100 - missing_data_percentage
        
        print(f"📊 Total Records: {total_records:,}")
        print(f"✅ Data Completeness: {100 - missing_data_percentage:.1f}%")
        
        # Distribution analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            skewness_scores = data[numeric_cols].skew()
            highly_skewed = skewness_scores[abs(skewness_scores) > 2]
            
            metrics['data_skewness'] = skewness_scores.to_dict()
            metrics['highly_skewed_columns'] = highly_skewed.to_dict()
            
            if len(highly_skewed) > 0:
                print(f"⚠️  Highly skewed columns: {list(highly_skewed.index)}")
        
        results['metrics_calculated'] = metrics
        
        if missing_data_percentage > 10:
            results['recommendations'].append(f"High missing data rate ({missing_data_percentage:.1f}%) - investigate data quality issues")
        
        return results

def test_custom_tools():
    """Test all custom analytics tools"""
    print("🧪 Testing Custom Analytics Tools...")
    
    # Create sample data for testing
    np.random.seed(42)
    test_data = pd.DataFrame({
        'sales': np.random.normal(1000, 200, 100),
        'marketing_spend': np.random.normal(50, 10, 100),
        'customer_satisfaction': np.random.normal(7.5, 1, 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
        'month': np.random.choice(range(1, 13), 100),
        'year': 2024
    })
    
    # Test each tool
    tools = [
        AdvancedCorrelationTool(),
        SmartClusteringTool(),
        TimeSeriesInsightTool(),
        BusinessMetricsTool()
    ]
    
    results = {}
    
    for tool in tools:
        print(f"\n{'='*60}")
        print(f"Testing {tool.name}")
        print(f"{'='*60}")
        
        try:
            result = tool.analyze(test_data)
            results[tool.name] = result
            
            if 'error' not in result:
                print(f"✅ {tool.name} completed successfully")
            else:
                print(f"⚠️ {tool.name} returned error: {result['error']}")
                
        except Exception as e:
            print(f"❌ {tool.name} failed with error: {str(e)}")
            results[tool.name] = {'error': str(e)}
    
    print(f"\n{'='*60}")
    print("🎉 Custom tools testing complete!")
    print(f"✅ Successfully tested: {len([r for r in results.values() if 'error' not in r])} tools")
    print(f"⚠️ Tools with issues: {len([r for r in results.values() if 'error' in r])} tools")
    
    return results

if __name__ == "__main__":
    test_custom_tools()