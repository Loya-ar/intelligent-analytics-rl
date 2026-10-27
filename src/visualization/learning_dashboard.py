"""
Learning Visualization Dashboard
Creates comprehensive learning curves and performance metrics visualization
"""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import json
import os

class LearningVisualizer:
    """
    Comprehensive visualization suite for RL agent learning progress
    """
    
    def __init__(self, save_dir="results/visualizations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        print(f"📊 Learning Visualizer initialized. Saving to: {save_dir}")
    
    def plot_learning_curves(self, coordinator, save_plots=True):
        """
        Create comprehensive learning curves for all agents
        """
        print("📈 Generating learning curves...")
        
        # Extract learning data
        agent_data = {}
        for agent_name, agent in coordinator.agents.items():
            if hasattr(agent, 'learning_history') and agent.learning_history:
                episodes = [entry['episode'] for entry in agent.learning_history]
                rewards = [entry['reward'] for entry in agent.learning_history]
                q_values = [entry['q_value'] for entry in agent.learning_history]
                
                agent_data[agent_name] = {
                    'episodes': episodes,
                    'rewards': rewards, 
                    'q_values': q_values,
                    'cumulative_reward': np.cumsum(rewards),
                    'moving_avg_reward': self._moving_average(rewards, window=3)
                }
        
        if not agent_data:
            print("❌ No learning history found. Run training episodes first.")
            return None
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Learning Curves (Rewards)', 'Q-Value Evolution', 
                          'Cumulative Performance', 'Agent Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Plot 1: Episode Rewards
        for i, (agent_name, data) in enumerate(agent_data.items()):
            fig.add_trace(
                go.Scatter(
                    x=data['episodes'], 
                    y=data['rewards'],
                    mode='lines+markers',
                    name=f"{agent_name} Rewards",
                    line=dict(color=colors[i % len(colors)]),
                    marker=dict(size=6)
                ),
                row=1, col=1
            )
        
        # Plot 2: Q-Value Evolution  
        for i, (agent_name, data) in enumerate(agent_data.items()):
            fig.add_trace(
                go.Scatter(
                    x=data['episodes'],
                    y=data['q_values'],
                    mode='lines+markers',
                    name=f"{agent_name} Q-Values",
                    line=dict(color=colors[i % len(colors)], dash='dash'),
                    marker=dict(size=4),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Plot 3: Cumulative Performance
        for i, (agent_name, data) in enumerate(agent_data.items()):
            fig.add_trace(
                go.Scatter(
                    x=data['episodes'],
                    y=data['cumulative_reward'],
                    mode='lines',
                    name=f"{agent_name} Cumulative",
                    line=dict(color=colors[i % len(colors)], width=3),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Plot 4: Agent Performance Comparison (Bar Chart)
        agent_names = list(agent_data.keys())
        total_rewards = [sum(data['rewards']) for data in agent_data.values()]
        avg_q_values = [np.mean(data['q_values']) for data in agent_data.values()]
        
        fig.add_trace(
            go.Bar(
                x=agent_names,
                y=total_rewards,
                name="Total Reward",
                marker_color=colors[:len(agent_names)],
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="<b>Multi-Agent Reinforcement Learning Progress</b>",
            title_x=0.5,
            template="plotly_white",
            font=dict(size=12)
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Episode", row=1, col=1)
        fig.update_xaxes(title_text="Episode", row=1, col=2)
        fig.update_xaxes(title_text="Episode", row=2, col=1)
        fig.update_xaxes(title_text="Agent", row=2, col=2)
        
        fig.update_yaxes(title_text="Reward", row=1, col=1)
        fig.update_yaxes(title_text="Q-Value", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative Reward", row=2, col=1)
        fig.update_yaxes(title_text="Total Reward", row=2, col=2)
        
        if save_plots:
            # Save interactive plot
            fig.write_html(f"{self.save_dir}/learning_curves.html")
            fig.write_image(f"{self.save_dir}/learning_curves.png", width=1200, height=800)
            print(f"✅ Learning curves saved to {self.save_dir}")
        
        fig.show()
        return fig
    
    def plot_agent_specialization(self, coordinator, save_plots=True):
        """
        Visualize how agents specialized over time
        """
        print("🎯 Generating agent specialization analysis...")
        
        # Create specialization heatmap
        agent_context_performance = {}
        
        for agent_name, agent in coordinator.agents.items():
            agent_context_performance[agent_name] = {}
            
            for context, q_value in agent.action_values.items():
                # Simplify context names for readability
                simple_context = self._simplify_context(context)
                agent_context_performance[agent_name][simple_context] = q_value
        
        if not agent_context_performance:
            print("❌ No specialization data available")
            return None
        
        # Convert to DataFrame for heatmap
        df_data = []
        all_contexts = set()
        for agent_data in agent_context_performance.values():
            all_contexts.update(agent_data.keys())
        
        for agent_name, contexts in agent_context_performance.items():
            for context in all_contexts:
                df_data.append({
                    'Agent': agent_name,
                    'Context': context,
                    'Q_Value': contexts.get(context, 0)
                })
        
        df = pd.DataFrame(df_data)
        
        # Create pivot table for heatmap
        pivot_df = df.pivot(index='Agent', columns='Context', values='Q_Value')
        pivot_df = pivot_df.fillna(0)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=0, ax=ax, cbar_kws={'label': 'Q-Value'})
        
        ax.set_title('Agent Specialization Heatmap\n(Higher Q-Values = Better Performance)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Data Context', fontsize=12)
        ax.set_ylabel('Agent Type', fontsize=12)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{self.save_dir}/agent_specialization.png", dpi=300, bbox_inches='tight')
            print(f"✅ Specialization heatmap saved to {self.save_dir}")
        
        plt.show()
        return fig, ax
    
    def plot_coordination_metrics(self, coordinator, save_plots=True):
        """
        Visualize multi-agent coordination effectiveness
        """
        print("🤝 Generating coordination analysis...")
        
        if not coordinator.coordination_history:
            print("❌ No coordination history available")
            return None
        
        # Extract coordination data
        episodes = []
        participating_agents = []
        total_agents = []
        insights_generated = []
        collaboration_events = []
        
        for i, episode in enumerate(coordinator.coordination_history):
            episodes.append(i + 1)
            participating_agents.append(len(episode.get('participating_agents', episode.get('active_agents', []))))
            total_agents.append(len(coordinator.agents))
            
            # Count insights
            results = episode.get('results', {})
            insights = sum(1 for result in results.values() if result is not None)
            insights_generated.append(insights)
            
            # Count collaboration events (message exchanges)
            collaboration = 0
            for agent in coordinator.agents.values():
                collaboration += len(agent.message_queue)
            collaboration_events.append(collaboration)
        
        # Create coordination dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Agent Participation Over Time', 'Insights Generated', 
                          'Collaboration Efficiency', 'Success Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Agent Participation
        fig.add_trace(
            go.Scatter(x=episodes, y=participating_agents, mode='lines+markers',
                      name='Active Agents', line=dict(color='#1f77b4', width=3)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=episodes, y=total_agents, mode='lines',
                      name='Total Agents', line=dict(color='#ff7f0e', dash='dash')),
            row=1, col=1
        )
        
        # Plot 2: Insights Generated
        fig.add_trace(
            go.Bar(x=episodes, y=insights_generated, name='Insights',
                  marker_color='#2ca02c', showlegend=False),
            row=1, col=2
        )
        
        # Plot 3: Collaboration Efficiency
        efficiency = [p/t if t > 0 else 0 for p, t in zip(participating_agents, total_agents)]
        fig.add_trace(
            go.Scatter(x=episodes, y=efficiency, mode='lines+markers',
                      name='Efficiency', line=dict(color='#d62728', width=2),
                      showlegend=False),
            row=2, col=1
        )
        
        # Plot 4: Success Rate (assuming success = insights > 0)
        success_rate = [1 if i > 0 else 0 for i in insights_generated]
        cumulative_success = np.cumsum(success_rate) / np.arange(1, len(success_rate) + 1)
        
        fig.add_trace(
            go.Scatter(x=episodes, y=cumulative_success, mode='lines+markers',
                      name='Success Rate', line=dict(color='#9467bd', width=2),
                      showlegend=False),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            title_text="<b>Multi-Agent Coordination Metrics</b>",
            title_x=0.5,
            template="plotly_white"
        )
        
        # Update axes
        fig.update_xaxes(title_text="Episode")
        fig.update_yaxes(title_text="Number of Agents", row=1, col=1)
        fig.update_yaxes(title_text="Insights Count", row=1, col=2)
        fig.update_yaxes(title_text="Participation Ratio", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Success Rate", row=2, col=2)
        
        if save_plots:
            fig.write_html(f"{self.save_dir}/coordination_metrics.html")
            fig.write_image(f"{self.save_dir}/coordination_metrics.png", width=1000, height=600)
            print(f"✅ Coordination metrics saved to {self.save_dir}")
        
        fig.show()
        return fig
    
    def generate_performance_report(self, coordinator, save_report=True):
        """
        Generate comprehensive performance statistics
        """
        print("📊 Generating performance report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'agents': {},
            'system_metrics': {},
            'coordination_stats': {}
        }
        
        # Agent-specific metrics
        for agent_name, agent in coordinator.agents.items():
            agent_stats = {
                'total_reward': agent.total_reward,
                'episodes_participated': len(agent.action_values),
                'average_q_value': np.mean(list(agent.action_values.values())) if agent.action_values else 0,
                'max_q_value': max(agent.action_values.values()) if agent.action_values else 0,
                'contexts_learned': len(agent.action_values),
                'messages_received': len(agent.message_queue) if hasattr(agent, 'message_queue') else 0
            }
            report['agents'][agent_name] = agent_stats
        
        # System-wide metrics
        total_system_reward = sum(agent.total_reward for agent in coordinator.agents.values())
        total_contexts = sum(len(agent.action_values) for agent in coordinator.agents.values())
        
        report['system_metrics'] = {
            'total_system_reward': total_system_reward,
            'total_contexts_learned': total_contexts,
            'coordination_episodes': len(coordinator.coordination_history),
            'average_reward_per_episode': total_system_reward / max(1, len(coordinator.coordination_history))
        }
        
        # Coordination statistics
        if coordinator.coordination_history:
            participating_agents = [len(ep['participating_agents']) for ep in coordinator.coordination_history]
            insights_per_episode = []
            
            for episode in coordinator.coordination_history:
                results = episode.get('results', {})
                insights = sum(1 for result in results.values() if result is not None)
                insights_per_episode.append(insights)
            
            report['coordination_stats'] = {
                'avg_participating_agents': np.mean(participating_agents),
                'avg_insights_per_episode': np.mean(insights_per_episode),
                'total_insights': sum(insights_per_episode),
                'coordination_efficiency': np.mean(participating_agents) / len(coordinator.agents)
            }
        
        if save_report:
            report_path = f"{self.save_dir}/performance_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            print(f"✅ Performance report saved to {report_path}")
        
        return report
    
    def _moving_average(self, data, window=3):
        """Calculate moving average"""
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    def _simplify_context(self, context):
        """Simplify context string for readability"""
        # Extract key information from context
        parts = context.split('_')
        simplified = []
        
        for part in parts:
            if 'data_size:' in part:
                simplified.append(part.split(':')[1].title())
            elif 'domain:' in part:
                simplified.append(part.split(':')[1].title())
            elif 'has_region:True' in part:
                simplified.append('Regional')
            elif 'has_date:True' in part:
                simplified.append('Temporal')
        
        return ' '.join(simplified) if simplified else 'General'

def test_visualization():
    """Test the visualization system"""
    print("🧪 Testing Learning Visualization System...")
    
    # Import our multi-agent system
    try:
        from src.agents.multi_agent_coordinator import MultiAgentCoordinator
        import pandas as pd
        
        # Create and train coordinator
        coordinator = MultiAgentCoordinator()
        
        # Run a few training episodes for demonstration
        datasets = ['data/sales_data.csv', 'data/marketing_data.csv']
        
        for i, data_path in enumerate(datasets):
            try:
                data = pd.read_csv(data_path)
                print(f"\n🔄 Training episode {i+1} on {data_path}")
                
                # Reset analysis history for new episode
                for agent in coordinator.agents.values():
                    if hasattr(agent, 'analysis_history'):
                        agent.analysis_history = []
                
                result = coordinator.coordinate_analysis(data, max_agents=3)
                
            except FileNotFoundError:
                print(f"⚠️ {data_path} not found, skipping...")
                continue
        
        # Create visualizer and generate all plots
        visualizer = LearningVisualizer()
        
        # Generate visualizations
        visualizer.plot_learning_curves(coordinator)
        visualizer.plot_agent_specialization(coordinator)  
        visualizer.plot_coordination_metrics(coordinator)
        
        # Generate performance report
        report = visualizer.generate_performance_report(coordinator)
        
        print("\n📊 Performance Summary:")
        for agent_name, stats in report['agents'].items():
            print(f"  {agent_name}: {stats['total_reward']:.2f} total reward")
        
        print(f"🎯 System Total Reward: {report['system_metrics']['total_system_reward']:.2f}")
        print(f"🤝 Coordination Efficiency: {report['coordination_stats'].get('coordination_efficiency', 0):.2%}")
        
        print("\n✅ Visualization system test complete!")
        
    except ImportError:
        print("❌ Could not import multi-agent system. Run it from the repository root: python -m src.visualization.learning_dashboard")

if __name__ == "__main__":
    test_visualization()