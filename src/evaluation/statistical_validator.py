"""
Statistical Validation Framework for RL Agent Performance
Provides rigorous statistical analysis of learning effectiveness
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu, ks_2samp
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

class StatisticalValidator:
    """
    Comprehensive statistical validation of RL agent learning performance
    """
    
    def __init__(self, significance_level=0.05, save_dir="results/statistical_analysis"):
        self.alpha = significance_level
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"📊 Statistical Validator initialized (α = {significance_level})")
        print(f"📁 Results will be saved to: {save_dir}")
    
    def validate_learning_improvement(self, coordinator):
        """
        Test if agents show statistically significant learning improvement
        """
        print("\n📈 STATISTICAL VALIDATION OF LEARNING IMPROVEMENT")
        print("=" * 60)
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'significance_level': self.alpha,
            'agents': {},
            'system_level': {}
        }
        
        for agent_name, agent in coordinator.agents.items():
            if not hasattr(agent, 'learning_history') or len(agent.learning_history) < 5:
                print(f"⚠️ {agent_name}: Insufficient data for statistical testing (need ≥5 episodes)")
                continue
            
            print(f"\n🤖 Analyzing {agent_name}:")
            print("-" * 30)
            
            rewards = [episode['reward'] for episode in agent.learning_history]
            q_values = [episode['q_value'] for episode in agent.learning_history]
            episodes = [episode['episode'] for episode in agent.learning_history]
            
            agent_results = self._analyze_agent_learning(agent_name, rewards, q_values, episodes)
            validation_results['agents'][agent_name] = agent_results
        
        # System-level analysis
        if len(validation_results['agents']) > 0:
            system_results = self._analyze_system_performance(coordinator, validation_results['agents'])
            validation_results['system_level'] = system_results
        
        # Save results
        results_path = os.path.join(self.save_dir, 'statistical_validation_report.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            # default= converts NumPy scalars (np.bool_, np.float64) that json can't encode
            json.dump(validation_results, f, indent=2,
                      default=lambda o: o.item() if hasattr(o, 'item') else str(o))
        
        print(f"\n💾 Statistical validation results saved to: {results_path}")
        
        return validation_results
    
    def _analyze_agent_learning(self, agent_name, rewards, q_values, episodes):
        """
        Perform comprehensive statistical analysis for individual agent
        """
        n_episodes = len(rewards)
        
        # Split data into early vs late episodes for comparison
        split_point = n_episodes // 2
        early_rewards = rewards[:split_point]
        late_rewards = rewards[split_point:]
        
        early_q_values = q_values[:split_point]
        late_q_values = q_values[split_point:]
        
        results = {
            'episodes_analyzed': n_episodes,
            'reward_analysis': {},
            'q_value_analysis': {},
            'trend_analysis': {},
            'learning_indicators': {}
        }
        
        # Reward Analysis
        print(f"📊 Reward Analysis:")
        
        early_mean = np.mean(early_rewards)
        late_mean = np.mean(late_rewards)
        
        # Paired t-test for reward improvement
        if len(early_rewards) == len(late_rewards):
            t_stat, p_value = ttest_rel(late_rewards, early_rewards)
            test_type = "Paired t-test"
        else:
            t_stat, p_value = mannwhitneyu(late_rewards, early_rewards, alternative='greater')
            test_type = "Mann-Whitney U test"
        
        improvement = ((late_mean - early_mean) / abs(early_mean)) * 100 if early_mean != 0 else 0
        is_significant = p_value < self.alpha
        
        print(f"  Early episodes mean: {early_mean:.3f}")
        print(f"  Late episodes mean: {late_mean:.3f}")  
        print(f"  Improvement: {improvement:+.1f}%")
        print(f"  {test_type}: t={t_stat:.3f}, p={p_value:.4f}")
        print(f"  Significant improvement: {'✅ YES' if is_significant else '❌ NO'}")
        
        results['reward_analysis'] = {
            'early_mean': early_mean,
            'late_mean': late_mean,
            'improvement_percentage': improvement,
            'test_statistic': t_stat,
            'p_value': p_value,
            'significant_improvement': is_significant,
            'test_type': test_type
        }
        
        # Q-Value Analysis
        print(f"\n📈 Q-Value Analysis:")
        
        early_q_mean = np.mean(early_q_values)
        late_q_mean = np.mean(late_q_values)
        
        # Test for Q-value convergence/improvement
        if len(early_q_values) == len(late_q_values):
            q_t_stat, q_p_value = ttest_rel(late_q_values, early_q_values)
        else:
            q_t_stat, q_p_value = mannwhitneyu(late_q_values, early_q_values, alternative='greater')
        
        q_improvement = ((late_q_mean - early_q_mean) / abs(early_q_mean)) * 100 if early_q_mean != 0 else 0
        q_significant = q_p_value < self.alpha
        
        print(f"  Early Q-values mean: {early_q_mean:.3f}")
        print(f"  Late Q-values mean: {late_q_mean:.3f}")
        print(f"  Q-value improvement: {q_improvement:+.1f}%")
        print(f"  Statistical significance: {'✅ YES' if q_significant else '❌ NO'}")
        
        results['q_value_analysis'] = {
            'early_mean': early_q_mean,
            'late_mean': late_q_mean,
            'improvement_percentage': q_improvement,
            'p_value': q_p_value,
            'significant_improvement': q_significant
        }
        
        # Trend Analysis (Linear regression on episode vs reward)
        print(f"\n📊 Trend Analysis:")
        
        slope, intercept, r_value, trend_p_value, std_err = stats.linregress(episodes, rewards)
        
        trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        trend_significant = trend_p_value < self.alpha
        
        print(f"  Learning trend: {trend_direction} (slope={slope:.4f})")
        print(f"  Trend strength: R²={r_value**2:.3f}")
        print(f"  Trend significance: {'✅ YES' if trend_significant else '❌ NO'} (p={trend_p_value:.4f})")
        
        results['trend_analysis'] = {
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': trend_p_value,
            'direction': trend_direction,
            'significant_trend': trend_significant
        }
        
        # Learning Quality Indicators
        reward_variance = np.var(rewards)
        q_value_variance = np.var(q_values)
        
        # Stability score (lower variance in later episodes indicates convergence)
        late_reward_variance = np.var(late_rewards)
        early_reward_variance = np.var(early_rewards)
        stability_improvement = (early_reward_variance - late_reward_variance) / early_reward_variance if early_reward_variance > 0 else 0
        
        results['learning_indicators'] = {
            'reward_variance': reward_variance,
            'q_value_variance': q_value_variance,
            'stability_improvement': stability_improvement,
            'learning_consistency': 1 / (1 + reward_variance) if reward_variance > 0 else 1
        }
        
        print(f"\n💡 Learning Quality Indicators:")
        print(f"  Learning consistency: {results['learning_indicators']['learning_consistency']:.3f}")
        print(f"  Stability improvement: {stability_improvement:+.1f}%")
        
        return results
    
    def _analyze_system_performance(self, coordinator, agent_results):
        """
        Analyze system-level learning performance
        """
        print(f"\n🎯 SYSTEM-LEVEL ANALYSIS:")
        print("-" * 30)
        
        # Count agents with significant improvement
        significant_agents = sum(1 for results in agent_results.values() 
                               if results.get('reward_analysis', {}).get('significant_improvement', False))
        
        total_agents = len(agent_results)
        
        # System learning success rate
        learning_success_rate = (significant_agents / total_agents) * 100 if total_agents > 0 else 0
        
        # Combined system performance metrics
        all_improvements = [results['reward_analysis']['improvement_percentage'] 
                          for results in agent_results.values() 
                          if 'reward_analysis' in results]
        
        system_avg_improvement = np.mean(all_improvements) if all_improvements else 0
        
        # Coordination effectiveness
        coordination_episodes = len(coordinator.coordination_history)
        successful_episodes = len([ep for ep in coordinator.coordination_history if ep.get('results')])
        coordination_success_rate = (successful_episodes / coordination_episodes) * 100 if coordination_episodes > 0 else 0
        
        print(f"📊 Agents with significant learning: {significant_agents}/{total_agents} ({learning_success_rate:.1f}%)")
        print(f"📊 Average system improvement: {system_avg_improvement:+.1f}%")
        print(f"📊 Coordination success rate: {coordination_success_rate:.1f}%")
        
        # Statistical significance of system-level learning
        if len(all_improvements) >= 3:
            # One-sample t-test against null hypothesis of no improvement (0%)
            t_stat, p_value = stats.ttest_1samp(all_improvements, 0)
            system_learning_significant = p_value < self.alpha and t_stat > 0
            
            print(f"📊 System-wide learning significance: {'✅ YES' if system_learning_significant else '❌ NO'} (p={p_value:.4f})")
        else:
            system_learning_significant = False
            p_value = None
            print(f"⚠️ Insufficient agents for system-level significance testing")
        
        system_results = {
            'agents_with_significant_learning': significant_agents,
            'total_agents': total_agents,
            'learning_success_rate': learning_success_rate,
            'average_improvement': system_avg_improvement,
            'coordination_success_rate': coordination_success_rate,
            'system_learning_significant': system_learning_significant,
            'system_p_value': p_value
        }
        
        return system_results
    
    def compare_learning_algorithms(self, results1, results2, algorithm1_name="Algorithm 1", algorithm2_name="Algorithm 2"):
        """
        Compare performance between different learning algorithms
        """
        print(f"\n🔬 ALGORITHM COMPARISON: {algorithm1_name} vs {algorithm2_name}")
        print("=" * 60)
        
        # Extract performance metrics
        perf1 = [agent['reward_analysis']['improvement_percentage'] 
                for agent in results1['agents'].values() 
                if 'reward_analysis' in agent]
        
        perf2 = [agent['reward_analysis']['improvement_percentage'] 
                for agent in results2['agents'].values() 
                if 'reward_analysis' in agent]
        
        if len(perf1) == 0 or len(perf2) == 0:
            print("❌ Insufficient data for comparison")
            return None
        
        # Statistical comparison
        if len(perf1) == len(perf2):
            # Paired comparison
            t_stat, p_value = ttest_rel(perf1, perf2)
            test_type = "Paired t-test"
        else:
            # Independent samples
            t_stat, p_value = stats.ttest_ind(perf1, perf2)
            test_type = "Independent t-test"
        
        mean1, mean2 = np.mean(perf1), np.mean(perf2)
        
        print(f"📊 {algorithm1_name}: {mean1:.1f}% average improvement")
        print(f"📊 {algorithm2_name}: {mean2:.1f}% average improvement")
        print(f"📊 {test_type}: t={t_stat:.3f}, p={p_value:.4f}")
        
        if p_value < self.alpha:
            better_algorithm = algorithm1_name if mean1 > mean2 else algorithm2_name
            print(f"✅ Significant difference: {better_algorithm} performs better")
        else:
            print(f"❌ No significant difference between algorithms")
        
        return {
            'algorithm1_performance': mean1,
            'algorithm2_performance': mean2,
            'test_statistic': t_stat,
            'p_value': p_value,
            'significant_difference': p_value < self.alpha,
            'better_algorithm': algorithm1_name if mean1 > mean2 else algorithm2_name if mean1 != mean2 else "No difference"
        }
    
    def generate_confidence_intervals(self, coordinator, confidence_level=0.95):
        """
        Generate confidence intervals for agent performance metrics
        """
        print(f"\n📊 CONFIDENCE INTERVALS ({confidence_level*100}% confidence)")
        print("=" * 50)
        
        alpha = 1 - confidence_level
        
        intervals = {}
        
        for agent_name, agent in coordinator.agents.items():
            if not hasattr(agent, 'learning_history') or len(agent.learning_history) < 3:
                continue
            
            rewards = [ep['reward'] for ep in agent.learning_history]
            n = len(rewards)
            
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards, ddof=1)
            
            # Calculate confidence interval
            t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
            margin_of_error = t_critical * (std_reward / np.sqrt(n))
            
            ci_lower = mean_reward - margin_of_error
            ci_upper = mean_reward + margin_of_error
            
            print(f"🤖 {agent_name}:")
            print(f"   Mean reward: {mean_reward:.3f}")
            print(f"   {confidence_level*100}% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
            
            intervals[agent_name] = {
                'mean': mean_reward,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'margin_of_error': margin_of_error
            }
        
        return intervals
    
    def create_statistical_summary_report(self, validation_results):
        """
        Create a comprehensive statistical summary report
        """
        print(f"\n📋 GENERATING STATISTICAL SUMMARY REPORT")
        print("=" * 50)
        
        report_lines = []
        report_lines.append("STATISTICAL VALIDATION SUMMARY REPORT")
        report_lines.append("=" * 45)
        report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Significance Level: α = {self.alpha}")
        report_lines.append("")
        
        # System Overview
        system = validation_results.get('system_level', {})
        report_lines.append("SYSTEM-LEVEL RESULTS:")
        report_lines.append("-" * 20)
        report_lines.append(f"Agents Analyzed: {system.get('total_agents', 0)}")
        report_lines.append(f"Agents with Significant Learning: {system.get('agents_with_significant_learning', 0)}")
        report_lines.append(f"Learning Success Rate: {system.get('learning_success_rate', 0):.1f}%")
        report_lines.append(f"Average System Improvement: {system.get('average_improvement', 0):+.1f}%")
        
        if system.get('system_learning_significant'):
            report_lines.append("System-wide Learning: ✅ SIGNIFICANT")
        else:
            report_lines.append("System-wide Learning: ❌ NOT SIGNIFICANT")
        
        report_lines.append("")
        
        # Agent-by-Agent Results
        report_lines.append("AGENT-LEVEL RESULTS:")
        report_lines.append("-" * 20)
        
        for agent_name, results in validation_results.get('agents', {}).items():
            report_lines.append(f"\n{agent_name}:")
            
            reward_analysis = results.get('reward_analysis', {})
            if reward_analysis:
                improvement = reward_analysis.get('improvement_percentage', 0)
                significant = reward_analysis.get('significant_improvement', False)
                p_value = reward_analysis.get('p_value', 1.0)
                
                report_lines.append(f"  Performance Improvement: {improvement:+.1f}%")
                report_lines.append(f"  Statistical Significance: {'✅ YES' if significant else '❌ NO'} (p={p_value:.4f})")
            
            trend_analysis = results.get('trend_analysis', {})
            if trend_analysis:
                trend = trend_analysis.get('direction', 'unknown')
                r_squared = trend_analysis.get('r_squared', 0)
                report_lines.append(f"  Learning Trend: {trend} (R²={r_squared:.3f})")
            
            indicators = results.get('learning_indicators', {})
            if indicators:
                consistency = indicators.get('learning_consistency', 0)
                report_lines.append(f"  Learning Consistency: {consistency:.3f}")
        
        report_lines.append("\n" + "=" * 45)
        
        # Statistical Interpretation
        report_lines.append("\nSTATISTICAL INTERPRETATION:")
        report_lines.append("-" * 25)
        
        significant_count = system.get('agents_with_significant_learning', 0)
        total_count = system.get('total_agents', 1)
        
        if significant_count / total_count >= 0.8:
            report_lines.append("✅ Strong evidence of effective learning across agents")
        elif significant_count / total_count >= 0.5:
            report_lines.append("✅ Moderate evidence of learning effectiveness")
        else:
            report_lines.append("⚠️ Limited evidence of systematic learning")
        
        avg_improvement = system.get('average_improvement', 0)
        if avg_improvement > 20:
            report_lines.append("✅ High magnitude of performance improvement")
        elif avg_improvement > 10:
            report_lines.append("✅ Moderate performance improvement observed")
        elif avg_improvement > 0:
            report_lines.append("✅ Positive learning trend detected")
        else:
            report_lines.append("⚠️ No clear improvement in performance")
        
        # Save report
        report_text = "\n".join(report_lines)
        print(report_text)
        
        report_path = os.path.join(self.save_dir, 'statistical_summary.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n💾 Statistical summary saved to: {report_path}")
        
        return report_text

def test_statistical_validation():
    """
    Test the statistical validation framework
    """
    print("🧪 Testing Statistical Validation Framework...")
    
    # We'll use the coordinator from our previous tests
    try:
        # Import the multi-agent system
        import sys
        import os
        sys.path.append('.')
        
        # Try to load a coordinator with learning history
        print("📊 Creating test coordinator with learning data...")
        
        # Create mock coordinator for demonstration
        class MockAgent:
            def __init__(self, name):
                self.name = name
                self.learning_history = [
                    {'episode': 1, 'reward': 1.0, 'q_value': 0.1},
                    {'episode': 2, 'reward': 1.2, 'q_value': 0.15},  
                    {'episode': 3, 'reward': 1.8, 'q_value': 0.25},
                    {'episode': 4, 'reward': 2.1, 'q_value': 0.35},
                    {'episode': 5, 'reward': 2.3, 'q_value': 0.42},
                    {'episode': 6, 'reward': 2.8, 'q_value': 0.51}
                ]
        
        class MockCoordinator:
            def __init__(self):
                self.agents = {
                    'statistical': MockAgent('Statistical_Agent'),
                    'geographic': MockAgent('Geographic_Agent')
                }
                self.coordination_history = [
                    {'results': {'stat': 'success'}},
                    {'results': {'geo': 'success'}},
                    {'results': {'both': 'success'}}
                ]
        
        coordinator = MockCoordinator()
        
        # Test statistical validation
        validator = StatisticalValidator(significance_level=0.05)
        
        print("\n🔬 Running statistical validation...")
        results = validator.validate_learning_improvement(coordinator)
        
        print("\n📊 Generating confidence intervals...")
        intervals = validator.generate_confidence_intervals(coordinator)
        
        print("\n📋 Creating summary report...")
        summary = validator.create_statistical_summary_report(results)
        
        print("\n✅ Statistical validation framework test complete!")
        return validator, results
        
    except Exception as e:
        print(f"❌ Error in statistical validation test: {str(e)}")
        return None, None

if __name__ == "__main__":
    test_statistical_validation()