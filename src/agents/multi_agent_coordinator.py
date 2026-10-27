"""
Multi-Agent Analytics System
Different agents specialize in different types of analysis
They learn to coordinate and share insights
"""

import numpy as np
import pandas as pd
import random
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

class BaseAnalyticsAgent:
    """Base class for specialized analytics agents"""
    
    def __init__(self, name, specialty, learning_rate=0.1):
        self.name = name
        self.specialty = specialty
        self.learning_rate = learning_rate
        
        # RL components
        self.action_values = defaultdict(float)
        self.action_counts = defaultdict(int)
        self.total_reward = 0
        self.episode_count = 0
        
        # Multi-agent communication
        self.message_queue = deque(maxlen=10)  # Store messages from other agents
        self.shared_insights = {}
        
        print(f"🤖 {self.name} ({self.specialty}) initialized")
    
    def extract_context(self, data):
        """Extract relevant context for this agent's specialty"""
        context = {}
        
        # Common context features
        context['data_size'] = 'large' if len(data) > 5000 else 'small'
        context['n_numeric'] = len(data.select_dtypes(include=[np.number]).columns)
        context['n_categorical'] = len(data.select_dtypes(include=['object']).columns)
        
        # Add specialty-specific context
        if self.specialty == 'statistical':
            context['has_correlations'] = context['n_numeric'] >= 2
        elif self.specialty == 'geographic':
            context['has_region'] = any('region' in col.lower() for col in data.columns)
        elif self.specialty == 'temporal':
            context['has_date'] = any('date' in col.lower() or 'time' in col.lower() for col in data.columns)
        
        return '_'.join([f"{k}:{v}" for k, v in sorted(context.items())])
    
    def receive_message(self, sender, message_type, content):
        """Receive communication from another agent"""
        message = {
            'sender': sender,
            'type': message_type,
            'content': content,
            'timestamp': len(self.message_queue)
        }
        self.message_queue.append(message)
        print(f"📨 {self.name} received {message_type} from {sender}")
    
    def send_message(self, recipient, message_type, content):
        """Send message to another agent"""
        recipient.receive_message(self.name, message_type, content)
        print(f"📤 {self.name} sent {message_type} to {recipient.name}")
    
    def calculate_reward(self, action_result):
        """Calculate reward based on analysis quality"""
        if action_result is None:
            return -0.5
        
        base_reward = 1.0
        
        # Specialty-specific bonuses
        if self.specialty == 'statistical' and 'correlations' in str(action_result):
            base_reward += 0.5
        elif self.specialty == 'geographic' and 'region' in str(action_result):
            base_reward += 0.5
        elif self.specialty == 'temporal' and 'trend' in str(action_result):
            base_reward += 0.5
        
        # Collaboration bonus - if we used insights from other agents
        if len(self.message_queue) > 0:
            base_reward += 0.2  # Bonus for coordination
        
        return base_reward
    
    def update_q_value(self, context, reward):
        """Update Q-value for this context"""
        old_q = self.action_values[context]
        new_q = old_q + self.learning_rate * (reward - old_q)
        
        self.action_values[context] = new_q
        self.action_counts[context] += 1
        self.total_reward += reward
        
        print(f"📈 {self.name} Q-value: {old_q:.3f} → {new_q:.3f} (reward: {reward:.2f})")
    
    def should_act(self, context, threshold=0.5):
        """Decide whether to perform analysis based on learned Q-values"""
        q_value = self.action_values[context]
        confidence = q_value + random.uniform(-0.1, 0.1)  # Small randomness
        
        # If we have no experience, encourage exploration
        if context not in self.action_counts or self.action_counts[context] == 0:
            confidence += 0.6  # Exploration bonus for new contexts
        
        decision = confidence > threshold
        print(f"🤔 {self.name} decision: {'ACT' if decision else 'PASS'} (confidence: {confidence:.3f})")
        return decision

class StatisticalAgent(BaseAnalyticsAgent):
    """Agent specialized in statistical analysis"""
    
    def __init__(self, name="Statistical_Agent"):
        super().__init__(name, "statistical")
    
    def analyze(self, data):
        """Perform statistical analysis"""
        print(f"\n=== {self.name}: Statistical Analysis ===")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return None
        
        # Calculate correlations
        corr_matrix = data[numeric_cols].corr()
        
        # Find strong correlations
        strong_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corrs.append((col1, col2, corr_val))
        
        print(f"🔍 Found {len(strong_corrs)} strong correlations")
        for col1, col2, corr in strong_corrs[:3]:
            direction = "positive" if corr > 0 else "negative"
            print(f"  {col1} ↔ {col2}: {corr:.3f} ({direction})")
        
        result = {
            'agent': self.name,
            'analysis_type': 'statistical',
            'correlations': strong_corrs,
            'insights': f"Found {len(strong_corrs)} significant correlations"
        }
        
        # Share insights with other agents (we'll implement this)
        self.shared_insights['correlations'] = strong_corrs
        
        return result

class GeographicAgent(BaseAnalyticsAgent):
    """Agent specialized in geographic/regional analysis"""
    
    def __init__(self, name="Geographic_Agent"):
        super().__init__(name, "geographic")
    
    def analyze(self, data):
        """Perform geographic analysis"""
        print(f"\n=== {self.name}: Geographic Analysis ===")
        
        # Find region column
        region_col = None
        for col in data.columns:
            if 'region' in col.lower() or 'location' in col.lower():
                region_col = col
                break
        
        if region_col is None:
            print("❌ No geographic column found")
            return None
        
        # Find target metric (prefer sales/revenue)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        target_col = None
        for col in numeric_cols:
            if any(keyword in col.lower() for keyword in ['sales', 'revenue', 'total']):
                target_col = col
                break
        
        if target_col is None:
            target_col = numeric_cols[0] if len(numeric_cols) > 0 else None
        
        if target_col is None:
            return None
        
        # Perform regional analysis
        regional_stats = data.groupby(region_col)[target_col].agg(['mean', 'sum', 'count'])
        
        best_region = regional_stats['mean'].idxmax()
        worst_region = regional_stats['mean'].idxmin()
        
        print(f"📍 Regional {target_col} analysis:")
        print(f"  🏆 Best: {best_region} (avg: {regional_stats.loc[best_region, 'mean']:.2f})")
        print(f"  ⚠️  Worst: {worst_region} (avg: {regional_stats.loc[worst_region, 'mean']:.2f})")
        
        result = {
            'agent': self.name,
            'analysis_type': 'geographic',
            'best_region': best_region,
            'worst_region': worst_region,
            'regional_stats': regional_stats.to_dict(),
            'insights': f"Regional analysis: {best_region} outperforms {worst_region}"
        }
        
        self.shared_insights['regional_performance'] = {
            'best': best_region,
            'worst': worst_region
        }
        
        return result

class TemporalAgent(BaseAnalyticsAgent):
    """Agent specialized in time-series and temporal analysis"""
    
    def __init__(self, name="Temporal_Agent"):
        super().__init__(name, "temporal")
    
    def analyze(self, data):
        """Perform temporal analysis"""
        print(f"\n=== {self.name}: Temporal Analysis ===")
        
        # Find date/time columns
        date_cols = [col for col in data.columns 
                    if 'date' in col.lower() or 'time' in col.lower()]
        
        if not date_cols:
            # Try month/year columns
            if 'month' in data.columns and 'year' in data.columns:
                print("🗓️ Using month/year for temporal analysis")
                
                # Simple seasonal analysis
                monthly_avg = data.groupby('month').agg({
                    col: 'mean' for col in data.select_dtypes(include=[np.number]).columns
                }).round(2)
                
                # Find target metric
                target_col = None
                for col in monthly_avg.columns:
                    if 'sales' in col.lower() or 'revenue' in col.lower():
                        target_col = col
                        break
                
                if target_col:
                    best_month = monthly_avg[target_col].idxmax()
                    worst_month = monthly_avg[target_col].idxmin()
                    
                    print(f"📈 Seasonal analysis of {target_col}:")
                    print(f"  🏆 Best month: {best_month} (avg: {monthly_avg.loc[best_month, target_col]:.2f})")
                    print(f"  ⚠️  Worst month: {worst_month} (avg: {monthly_avg.loc[worst_month, target_col]:.2f})")
                    
                    result = {
                        'agent': self.name,
                        'analysis_type': 'temporal',
                        'best_period': f"Month {best_month}",
                        'worst_period': f"Month {worst_month}",
                        'seasonal_pattern': monthly_avg[target_col].to_dict(),
                        'insights': f"Seasonal analysis: Month {best_month} shows peak performance"
                    }
                    
                    self.shared_insights['seasonality'] = {
                        'best_month': best_month,
                        'worst_month': worst_month
                    }
                    
                    return result
        
        print("❌ No suitable temporal columns found")
        return None

class MultiAgentCoordinator:
    """Coordinates multiple specialized agents"""
    
    def __init__(self):
        self.agents = {
            'statistical': StatisticalAgent(),
            'geographic': GeographicAgent(), 
            'temporal': TemporalAgent()
        }
        self.coordination_history = []
    
    def coordinate_analysis(self, data, max_agents=3):
        """Coordinate multiple agents to analyze data"""
        print(f"\n🎯 Multi-Agent Coordinator: Starting coordinated analysis")
        print(f"📊 Data shape: {data.shape}")
        
        results = {}
        active_agents = []
        
        # Let each agent decide if they should participate
        for agent_type, agent in self.agents.items():
            context = agent.extract_context(data)
            
            if agent.should_act(context, threshold=0.1):  # Much lower threshold for participation
                active_agents.append(agent)
        
        if not active_agents:
            print("❌ No agents decided to participate")
            return None
        
        print(f"✅ Active agents: {[a.name for a in active_agents]}")
        
        # Execute analyses in order of specialization relevance
        for agent in active_agents:
            print(f"\n🔄 Executing {agent.name}...")
            
            try:
                result = agent.analyze(data)
                
                if result:
                    results[agent.specialty] = result
                    
                    # Calculate reward and update Q-values
                    context = agent.extract_context(data)
                    reward = agent.calculate_reward(result)
                    agent.update_q_value(context, reward)
                    
                    # Send insights to other agents
                    for other_agent in active_agents:
                        if other_agent != agent and result:
                            agent.send_message(
                                other_agent, 
                                'insight_share', 
                                result.get('insights', 'Analysis completed')
                            )
                    
                else:
                    # Negative reward for failed analysis
                    context = agent.extract_context(data)
                    agent.update_q_value(context, -0.5)
            
            except Exception as e:
                print(f"❌ {agent.name} encountered error: {e}")
                continue
        
        # Synthesize results
        synthesis = self.synthesize_results(results)
        
        self.coordination_history.append({
            'active_agents': [a.name for a in active_agents],
            'participating_agents': [a.name for a in active_agents],
            'results': results,
            'synthesis': synthesis
        })
        
        return {
            'individual_results': results,
            'synthesis': synthesis,
            'participating_agents': [a.name for a in active_agents]
        }
    
    def synthesize_results(self, results):
        """Combine insights from multiple agents"""
        print(f"\n🧠 Synthesizing insights from {len(results)} agents...")
        
        synthesis = {
            'key_findings': [],
            'recommendations': [],
            'cross_agent_insights': []
        }
        
        # Extract key findings
        for agent_type, result in results.items():
            if result and 'insights' in result:
                synthesis['key_findings'].append(f"{agent_type}: {result['insights']}")
        
        # Look for cross-agent patterns
        if 'statistical' in results and 'geographic' in results:
            stat_result = results['statistical']
            geo_result = results['geographic']
            
            if stat_result and geo_result:
                synthesis['cross_agent_insights'].append(
                    f"Regional leader {geo_result.get('best_region', 'N/A')} may benefit from correlation analysis"
                )
        
        # Generate recommendations
        if len(results) >= 2:
            synthesis['recommendations'].append("Multi-dimensional analysis reveals complex patterns")
            synthesis['recommendations'].append("Consider integrated strategy addressing multiple factors")
        
        print("✅ Synthesis complete")
        return synthesis
    
    def show_agent_performance(self):
        """Display learning progress for all agents"""
        print("\n=== Multi-Agent Learning Summary ===")
        
        for agent_type, agent in self.agents.items():
            print(f"\n🤖 {agent.name}:")
            print(f"   Total reward: {agent.total_reward:.2f}")
            print(f"   Episodes: {len(agent.action_values)}")
            
            if agent.action_values:
                avg_q = np.mean(list(agent.action_values.values()))
                print(f"   Average Q-value: {avg_q:.3f}")
            
            print(f"   Messages received: {len(agent.message_queue)}")

def test_multi_agent_system():
    """Test the multi-agent coordination system"""
    print("=== Testing Multi-Agent Analytics System ===\n")
    
    # Create coordinator
    coordinator = MultiAgentCoordinator()
    
    # Load sample data
    try:
        sales_data = pd.read_csv('data/sales_data.csv')
        print(f"✅ Loaded sales data: {sales_data.shape}")
    except:
        print("❌ Could not load sales data")
        return
    
    # Run coordinated analysis (Episode 1)
    print("\n" + "="*50)
    print("🔄 Multi-Agent Episode 1")
    result1 = coordinator.coordinate_analysis(sales_data)
    
    # Run second episode to see learning
    print("\n" + "="*50)
    print("🔄 Multi-Agent Episode 2")
    result2 = coordinator.coordinate_analysis(sales_data)
    
    # Show learning progress
    coordinator.show_agent_performance()
    
    # Show final insights
    if result2:
        print("\n=== Final Coordinated Insights ===")
        for finding in result2['synthesis']['key_findings']:
            print(f"• {finding}")
        
        for rec in result2['synthesis']['recommendations']:
            print(f"💡 {rec}")
    
    print("\n🎉 Multi-agent system test complete!")

if __name__ == "__main__":
    test_multi_agent_system()