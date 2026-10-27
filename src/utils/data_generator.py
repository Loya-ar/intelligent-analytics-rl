"""
Business Data Generator for Analytics Assistant
Creates realistic business datasets for RL training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker

fake = Faker()

class BusinessDataGenerator:
    """Generates realistic business datasets for analytics learning"""
    
    def __init__(self, seed=42):
        """Initialize with random seed for reproducible data"""
        np.random.seed(seed)
        random.seed(seed)
        Faker.seed(seed)
    
    def generate_sales_data(self, n_records=10000):
        """Generate realistic sales transaction data"""
        print(f"Generating {n_records} sales records...")
        
        # Date range: last 2 years
        start_date = datetime.now() - timedelta(days=730)
        dates = [start_date + timedelta(days=x) for x in range(730)]
        
        # Regions and products
        regions = ['North', 'South', 'East', 'West', 'Central']
        products = ['Product_A', 'Product_B', 'Product_C', 'Product_D', 'Product_E']
        sales_reps = [fake.name() for _ in range(20)]
        
        data = []
        for _ in range(n_records):
            # Add seasonal trends and regional preferences
            date = random.choice(dates)
            region = random.choice(regions)
            product = random.choice(products)
            sales_rep = random.choice(sales_reps)
            
            # Base price with regional and seasonal variations
            base_price = {
                'Product_A': 100, 'Product_B': 150, 'Product_C': 200, 
                'Product_D': 75, 'Product_E': 300
            }[product]
            
            # Regional multiplier
            regional_multiplier = {
                'North': 1.2, 'South': 0.9, 'East': 1.1, 
                'West': 1.0, 'Central': 0.95
            }[region]
            
            # Seasonal effect (higher sales in Q4)
            month = date.month
            seasonal_multiplier = 1.3 if month in [11, 12] else (1.1 if month in [3, 4] else 1.0)
            
            # Final calculations
            quantity = np.random.poisson(3) + 1  # 1-10 items typically
            unit_price = base_price * regional_multiplier * seasonal_multiplier * np.random.uniform(0.9, 1.1)
            total_sales = quantity * unit_price
            
            # Add some customer demographics
            customer_segment = random.choice(['Enterprise', 'SMB', 'Individual'])
            customer_age = np.random.normal(45, 15) if customer_segment == 'Individual' else None
            
            data.append({
                'date': date,
                'region': region,
                'product': product,
                'sales_rep': sales_rep,
                'customer_segment': customer_segment,
                'customer_age': customer_age,
                'quantity': quantity,
                'unit_price': round(unit_price, 2),
                'total_sales': round(total_sales, 2),
                'month': month,
                'quarter': f"Q{(month-1)//3 + 1}",
                'year': date.year
            })
        
        df = pd.DataFrame(data)
        print(f"✅ Sales data generated: {df.shape}")
        return df
    
    def generate_marketing_data(self, n_records=5000):
        """Generate marketing campaign performance data"""
        print(f"Generating {n_records} marketing records...")
        
        campaigns = ['Email_Campaign', 'Social_Media', 'Google_Ads', 'Display_Ads', 'Content_Marketing']
        channels = ['Email', 'Facebook', 'Google', 'Instagram', 'LinkedIn', 'Website']
        
        data = []
        for _ in range(n_records):
            campaign = random.choice(campaigns)
            channel = random.choice(channels)
            
            # Realistic metrics with correlations
            impressions = np.random.exponential(1000) + 100
            clicks = np.random.binomial(int(impressions), 0.03)  # 3% average CTR
            conversions = np.random.binomial(clicks, 0.15)  # 15% conversion rate
            
            cost = impressions * np.random.uniform(0.01, 0.05)  # Cost per impression
            revenue = conversions * np.random.uniform(50, 200)  # Revenue per conversion
            
            data.append({
                'campaign': campaign,
                'channel': channel,
                'impressions': int(impressions),
                'clicks': clicks,
                'conversions': conversions,
                'cost': round(cost, 2),
                'revenue': round(revenue, 2),
                'ctr': round(clicks/impressions * 100, 2) if impressions > 0 else 0,
                'conversion_rate': round(conversions/clicks * 100, 2) if clicks > 0 else 0,
                'roas': round(revenue/cost, 2) if cost > 0 else 0
            })
        
        df = pd.DataFrame(data)
        print(f"✅ Marketing data generated: {df.shape}")
        return df
    
    def generate_hr_data(self, n_records=2000):
        """Generate HR/employee performance data"""
        print(f"Generating {n_records} HR records...")
        
        departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations']
        job_levels = ['Junior', 'Mid', 'Senior', 'Lead', 'Manager']
        
        data = []
        for _ in range(n_records):
            dept = random.choice(departments)
            level = random.choice(job_levels)
            
            # Generate correlated performance metrics
            base_salary = {
                'Junior': 60000, 'Mid': 80000, 'Senior': 100000, 
                'Lead': 120000, 'Manager': 140000
            }[level]
            
            dept_multiplier = {
                'Engineering': 1.2, 'Sales': 1.1, 'Marketing': 1.0,
                'HR': 0.9, 'Finance': 1.1, 'Operations': 0.95
            }[dept]
            
            salary = base_salary * dept_multiplier * np.random.uniform(0.9, 1.3)
            performance_score = np.random.normal(7.5, 1.5)  # 1-10 scale
            years_experience = np.random.gamma(2, 2)  # Realistic experience distribution
            
            data.append({
                'employee_id': f"EMP_{random.randint(1000, 9999)}",
                'department': dept,
                'job_level': level,
                'salary': round(salary, 0),
                'performance_score': round(max(1, min(10, performance_score)), 1),
                'years_experience': round(years_experience, 1),
                'training_hours': np.random.poisson(20),
                'satisfaction_score': round(np.random.normal(7, 1.5), 1),
                'promotion_eligible': random.choice([True, False])
            })
        
        df = pd.DataFrame(data)
        print(f"✅ HR data generated: {df.shape}")
        return df

def create_sample_datasets():
    """Create all sample datasets and save to files"""
    print("=== Creating Sample Business Datasets ===\n")
    
    generator = BusinessDataGenerator()
    
    # Generate datasets
    sales_df = generator.generate_sales_data(10000)
    marketing_df = generator.generate_marketing_data(5000)
    hr_df = generator.generate_hr_data(2000)
    
    # Save to CSV files
    print("\nSaving datasets...")
    sales_df.to_csv('data/sales_data.csv', index=False)
    marketing_df.to_csv('data/marketing_data.csv', index=False)
    hr_df.to_csv('data/hr_data.csv', index=False)
    
    print("✅ All datasets saved to data/ folder")
    
    # Show previews
    print("\n=== Dataset Previews ===")
    print("\n1. Sales Data Preview:")
    print(sales_df.head(3))
    print(f"   Shape: {sales_df.shape}")
    
    print("\n2. Marketing Data Preview:")
    print(marketing_df.head(3))
    print(f"   Shape: {marketing_df.shape}")
    
    print("\n3. HR Data Preview:")
    print(hr_df.head(3))
    print(f"   Shape: {hr_df.shape}")
    
    print("\n🎉 Sample datasets ready for analytics assistant!")
    return sales_df, marketing_df, hr_df

if __name__ == "__main__":
    create_sample_datasets()