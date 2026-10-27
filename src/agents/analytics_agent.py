"""
Basic Analytics Agent - Foundation for RL Learning
This agent can perform different types of analysis on business data
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

class AnalyticsAgent:
    """
    Basic analytics agent that can perform various analysis types
    This will later be enhanced with reinforcement learning
    """
    
    def __init__(self, name="Analytics_Agent"):
        self.name = name
        self.available_actions = [
            'descriptive_stats',
            'correlation_analysis', 
            'time_series_analysis',
            'regional_comparison',
            'product_performance',
            'clustering_analysis',
            'trend_analysis'
        ]
        
        # Track what analyses have been performed
        self.analysis_history = []
        
    def load_data(self, data_path):
        """Load business data for analysis"""
        try:
            self.data = pd.read_csv(data_path)
            self.data_path = data_path
            print(f"✅ Loaded data: {self.data.shape}")
            print(f"   Columns: {list(self.data.columns)}")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def descriptive_stats(self):
        """Perform basic descriptive statistics"""
        print(f"\n=== {self.name}: Descriptive Statistics ===")
        
        # Numerical columns summary
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats = self.data[numeric_cols].describe()
            print("📊 Numerical Summary:")
            print(stats.round(2))
        
        # Categorical columns summary  
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print("\n📋 Categorical Summary:")
            for col in categorical_cols[:3]:  # Show first 3 categorical columns
                value_counts = self.data[col].value_counts().head()
                print(f"  {col}: {dict(value_counts)}")
        
        # Record this analysis
        self.analysis_history.append('descriptive_stats')
        
        return {
            'action': 'descriptive_stats',
            'insights': f"Dataset has {self.data.shape[0]} rows, {len(numeric_cols)} numeric columns",
            'numeric_summary': stats.to_dict() if len(numeric_cols) > 0 else None
        }
    
    def correlation_analysis(self):
        """Analyze correlations between numerical variables"""
        print(f"\n=== {self.name}: Correlation Analysis ===")
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            print("❌ Need at least 2 numerical columns for correlation")
            return None
        
        # Calculate correlations
        corr_matrix = self.data[numeric_cols].corr()
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.3:  # Only show moderate+ correlations
                    corr_pairs.append((col1, col2, corr_val))
        
        # Sort by absolute correlation strength
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        print("🔗 Strong Correlations Found:")
        for col1, col2, corr in corr_pairs[:5]:  # Top 5
            direction = "positive" if corr > 0 else "negative"
            print(f"  {col1} ↔ {col2}: {corr:.3f} ({direction})")
        
        self.analysis_history.append('correlation_analysis')
        
        return {
            'action': 'correlation_analysis',
            'insights': f"Found {len(corr_pairs)} significant correlations",
            'top_correlations': corr_pairs[:5]
        }
    
    def regional_comparison(self):
        """Compare performance across regions (if region column exists)"""
        print(f"\n=== {self.name}: Regional Comparison ===")
        
        region_col = None
        for col in self.data.columns:
            if 'region' in col.lower():
                region_col = col
                break
        
        if not region_col:
            print("❌ No region column found")
            return None
        
        # Find a numerical column to analyze by region
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        target_col = None
        
        # Prefer sales, revenue, or performance related columns
        for col in numeric_cols:
            if any(keyword in col.lower() for keyword in ['sales', 'revenue', 'total', 'amount']):
                target_col = col
                break
        
        if not target_col:
            target_col = numeric_cols[0] if len(numeric_cols) > 0 else None
        
        if not target_col:
            print("❌ No numerical column found for comparison")
            return None
        
        # Perform regional analysis
        regional_stats = self.data.groupby(region_col)[target_col].agg([
            'count', 'mean', 'median', 'std', 'sum'
        ]).round(2)
        
        print(f"📍 {target_col} by {region_col}:")
        print(regional_stats)
        
        # Find best and worst performing regions
        best_region = regional_stats['mean'].idxmax()
        worst_region = regional_stats['mean'].idxmin()
        
        print(f"\n🏆 Best performing region: {best_region}")
        print(f"⚠️  Lowest performing region: {worst_region}")
        
        self.analysis_history.append('regional_comparison')
        
        return {
            'action': 'regional_comparison',
            'insights': f"Regional analysis of {target_col}: {best_region} performs best",
            'regional_stats': regional_stats.to_dict(),
            'best_region': best_region,
            'worst_region': worst_region
        }
    
    def get_analysis_suggestions(self):
        """Suggest what analysis to do next based on data characteristics"""
        suggestions = []
        
        # Check data characteristics
        has_date = any('date' in col.lower() for col in self.data.columns)
        has_region = any('region' in col.lower() for col in self.data.columns)
        has_product = any('product' in col.lower() for col in self.data.columns)
        numeric_cols = len(self.data.select_dtypes(include=[np.number]).columns)
        
        # Suggest based on data features
        if has_date and 'time_series_analysis' not in self.analysis_history:
            suggestions.append('time_series_analysis')
        
        if has_region and 'regional_comparison' not in self.analysis_history:
            suggestions.append('regional_comparison')
            
        if numeric_cols >= 2 and 'correlation_analysis' not in self.analysis_history:
            suggestions.append('correlation_analysis')
            
        if 'descriptive_stats' not in self.analysis_history:
            suggestions.append('descriptive_stats')
        
        return suggestions
    
    def perform_analysis(self, action):
        """Perform a specific analysis action"""
        if action not in self.available_actions:
            print(f"❌ Unknown action: {action}")
            return None
        
        # Route to appropriate analysis method
        if action == 'descriptive_stats':
            return self.descriptive_stats()
        elif action == 'correlation_analysis':
            return self.correlation_analysis()
        elif action == 'regional_comparison':
            return self.regional_comparison()
        else:
            print(f"⚠️  {action} not implemented yet")
            return None

# Test function
def test_analytics_agent():
    """Test the basic analytics agent"""
    print("=== Testing Basic Analytics Agent ===\n")
    
    # Create agent
    agent = AnalyticsAgent("Test_Agent")
    
    # Test with sales data
    print("1. Testing with Sales Data:")
    if agent.load_data('data/sales_data.csv'):
        
        # Get suggestions
        suggestions = agent.get_analysis_suggestions()
        print(f"📝 Suggested analyses: {suggestions}")
        
        # Perform first few analyses
        for action in suggestions[:3]:
            result = agent.perform_analysis(action)
            if result:
                print(f"✅ {action} completed")
        
        print(f"\n📋 Analysis history: {agent.analysis_history}")
    
    print("\n🎉 Basic agent test complete!")

if __name__ == "__main__":
    test_analytics_agent()