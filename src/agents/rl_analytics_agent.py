"""
Reinforcement Learning Analytics Agent
Uses contextual bandits to learn optimal analysis strategies
"""

import numpy as np
import pandas as pd
import random
from collections import defaultdict
import json
# Import fix - try different import methods
try:
    from src.agents.analytics_agent import AnalyticsAgent
except ImportError:
    try:
        from analytics_agent import AnalyticsAgent
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from src.agents.analytics_agent import AnalyticsAgent

class ContextualBanditAgent(AnalyticsAgent):
    """
    Analytics agent that learns through contextual bandit RL
    Learns which analyses are most valuable for different data contexts
    """
    
    def __init__(self, name="RL_Analytics_Agent", learning_rate=0.1, exploration_rate=0.3):
        super().__init__(name)
        
        # RL parameters
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        
        # Contextual bandit components
        self.action_values = defaultdict(lambda: defaultdict(float))  # Q(context, action)
        self.action_counts = defaultdict(lambda: defaultdict(int))    # Count(context, action)
        self.context_features = []
        
        # Learning history
        self.learning_history = []
        self.total_reward = 0
        self.episode_count = 0
        
        print(f"🤖 {self.name} initialized with RL capabilities")
    
    def extract_context(self, data=None):
        """
        Extract context features from the current dataset
        Context helps the agent understand what type of data it's analyzing
        """
        if data is None:
            data = self.data
        
        context = {}
        
        # Data size context
        n_rows = len(data)
        if n_rows < 1000:
            context['data_size'] = 'small'
        elif n_rows < 10000:
            context['data_size'] = 'medium'
        else:
            context['data_size'] = 'large'
        
        # Data type context
        numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(data.select_dtypes(include=['object']).columns)
        
        context['numeric_ratio'] = 'high' if numeric_cols > categorical_cols else 'low'
        
        # Business domain context (based on column names)
        col_names = ' '.join(data.columns).lower()
        if 'sales' in col_names or 'revenue' in col_names:
            context['domain'] = 'sales'
        elif 'marketing' in col_names or 'campaign' in col_names:
            context['domain'] = 'marketing'
        elif 'employee' in col_names or 'salary' in col_names:
            context['domain'] = 'hr'
        else:
            context['domain'] = 'general'
        
        # Temporal context
        has_date = any('date' in col.lower() for col in data.columns)
        context['has_temporal'] = 'yes' if has_date else 'no'
        
        # Geographic context
        has_region = any('region' in col.lower() for col in data.columns)
        context['has_geographic'] = 'yes' if has_region else 'no'
        
        # Convert to string for use as dictionary key
        context_key = '_'.join([f"{k}:{v}" for k, v in sorted(context.items())])
        
        return context, context_key
    
    def select_action_ucb(self, context_key, available_actions):
        """
        Select action using Upper Confidence Bound (UCB) algorithm
        Balances exploration vs exploitation
        """
        if not available_actions:
            return None
        
        # If we haven't tried all actions in this context, try unexplored ones
        unexplored = [a for a in available_actions 
                     if self.action_counts[context_key][a] == 0]
        
        if unexplored:
            action = random.choice(unexplored)
            print(f"🔍 Exploring new action: {action}")
            return action
        
        # UCB calculation: Q(s,a) + c * sqrt(ln(total_trials) / n(s,a))
        total_trials = sum(self.action_counts[context_key].values())
        ucb_values = {}
        
        for action in available_actions:
            q_value = self.action_values[context_key][action]
            count = max(1, self.action_counts[context_key][action])  # Avoid division by zero
            
            # UCB confidence bonus
            confidence = np.sqrt(2 * np.log(total_trials + 1) / count)
            ucb_values[action] = q_value + confidence
        
        # Select action with highest UCB value
        best_action = max(ucb_values.items(), key=lambda x: x[1])[0]
        
        print(f"🎯 UCB selected: {best_action} (UCB: {ucb_values[best_action]:.3f})")
        return best_action
    
    def select_action_epsilon_greedy(self, context_key, available_actions):
        """
        Select action using epsilon-greedy strategy
        Simple exploration vs exploitation
        """
        if not available_actions:
            return None
        
        # Exploration: random action
        if random.random() < self.exploration_rate:
            action = random.choice(available_actions)
            print(f"🎲 Random exploration: {action}")
            return action
        
        # Exploitation: best known action
        if context_key in self.action_values:
            best_action = max(self.action_values[context_key].items(),
                            key=lambda x: x[1])[0]
            print(f"⚡ Exploiting best: {best_action}")
            return best_action
        else:
            # No experience with this context, explore randomly
            action = random.choice(available_actions)
            print(f"🆕 New context, exploring: {action}")
            return action
    
    def calculate_reward(self, action, result):
        """
        Calculate reward for an analysis action
        Higher reward = more valuable insights discovered
        """
        if result is None:
            return -1.0  # Penalty for failed analysis
        
        base_reward = 1.0
        bonus_reward = 0.0
        
        # Reward based on insights discovered
        insights = result.get('insights', '')
        
        # Bonus for finding strong correlations
        if action == 'correlation_analysis' and result.get('top_correlations'):
            strong_corrs = [c for c in result['top_correlations'] if abs(c[2]) > 0.5]
            bonus_reward += len(strong_corrs) * 0.5
        
        # Bonus for regional insights
        if action == 'regional_comparison' and result.get('best_region'):
            bonus_reward += 0.3
        
        # Bonus for comprehensive descriptive stats
        if action == 'descriptive_stats' and result.get('numeric_summary'):
            bonus_reward += 0.2
        
        total_reward = base_reward + bonus_reward
        
        print(f"💰 Reward for {action}: {total_reward:.2f} (base: {base_reward}, bonus: {bonus_reward:.2f})")
        return total_reward
    
    def update_q_values(self, context_key, action, reward):
        """
        Update Q-values using simple Q-learning update rule
        Q(s,a) = Q(s,a) + α * (reward - Q(s,a))
        """
        old_q = self.action_values[context_key][action]
        
        # Q-learning update
        new_q = old_q + self.learning_rate * (reward - old_q)
        self.action_values[context_key][action] = new_q
        self.action_counts[context_key][action] += 1
        
        print(f"📈 Q-value updated: {action} {old_q:.3f} → {new_q:.3f}")
    
    def learn_from_analysis(self, context_key, action, result):
        """Complete learning cycle: action → result → reward → update"""
        # Calculate reward
        reward = self.calculate_reward(action, result)
        
        # Update Q-values
        self.update_q_values(context_key, action, reward)
        
        # Track learning history
        self.learning_history.append({
            'episode': self.episode_count,
            'context': context_key,
            'action': action,
            'reward': reward,
            'q_value': self.action_values[context_key][action]
        })
        
        self.total_reward += reward
        return reward
    
    def analyze_with_learning(self, data_path, max_analyses=5):
        """
        Perform analyses while learning from results
        Main RL loop: observe → act → get reward → learn
        """
        print(f"\n🚀 {self.name} starting learning episode {self.episode_count + 1}")
        
        # Load data and extract context
        if not self.load_data(data_path):
            return None
        
        context, context_key = self.extract_context()
        print(f"📊 Context: {context}")
        
        episode_rewards = []
        
        for step in range(max_analyses):
            print(f"\n--- Step {step + 1}/{max_analyses} ---")
            
            # Get available actions (exclude already performed)
            available_actions = [a for a in self.available_actions 
                               if a not in self.analysis_history]
            
            if not available_actions:
                print("✅ All available analyses completed")
                break
            
            # Select action using RL strategy
            action = self.select_action_ucb(context_key, available_actions)
            
            if action is None:
                break
            
            # Perform analysis
            result = self.perform_analysis(action)
            
            # Learn from the result
            reward = self.learn_from_analysis(context_key, action, result)
            episode_rewards.append(reward)
        
        self.episode_count += 1
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        
        print(f"\n📊 Episode {self.episode_count} Summary:")
        print(f"   Total reward: {sum(episode_rewards):.2f}")
        print(f"   Average reward: {avg_reward:.2f}")
        print(f"   Analyses performed: {len(episode_rewards)}")
        
        return {
            'episode': self.episode_count,
            'context': context,
            'total_reward': sum(episode_rewards),
            'average_reward': avg_reward,
            'actions_taken': len(episode_rewards)
        }
    
    def show_learning_progress(self):
        """Display learning progress and Q-values"""
        print("\n=== Learning Progress ===")
        print(f"Total episodes: {self.episode_count}")
        print(f"Total reward: {self.total_reward:.2f}")
        
        if self.learning_history:
            recent_rewards = [h['reward'] for h in self.learning_history[-10:]]
            print(f"Recent average reward: {np.mean(recent_rewards):.2f}")
        
        print("\n📊 Learned Q-Values:")
        for context_key, actions in self.action_values.items():
            print(f"  Context: {context_key[:50]}...")
            for action, q_value in sorted(actions.items(), key=lambda x: x[1], reverse=True):
                count = self.action_counts[context_key][action]
                print(f"    {action}: {q_value:.3f} (tried {count} times)")

def test_rl_agent():
    """Test the reinforcement learning agent"""
    print("=== Testing RL Analytics Agent ===\n")
    
    # Create RL agent
    agent = ContextualBanditAgent("RL_Test_Agent", exploration_rate=0.4)
    
    # Train on sales data
    print("Training on sales data...")
    result1 = agent.analyze_with_learning('data/sales_data.csv', max_analyses=3)
    
    # Reset for new episode with same data type
    agent.analysis_history = []  # Reset analysis history
    print("\n" + "="*50)
    result2 = agent.analyze_with_learning('data/sales_data.csv', max_analyses=3)
    
    # Show learning progress
    agent.show_learning_progress()
    
    print("\n🎉 RL agent testing complete!")

if __name__ == "__main__":
    test_rl_agent()