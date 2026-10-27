"""
System Architecture Diagram Generator
Creates professional architecture diagrams for the RL Analytics System
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import os

class ArchitectureDiagramGenerator:
    """
    Generates professional system architecture diagrams
    """
    
    def __init__(self, save_dir="results/diagrams"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Color scheme
        self.colors = {
            'agent': '#4A90E2',
            'coordinator': '#F5A623', 
            'data': '#7ED321',
            'rl': '#D0021B',
            'communication': '#9013FE',
            'output': '#50E3C2',
            'background': '#F8F9FA',
            'text': '#2C3E50'
        }
        
        print(f"🏗️ Architecture Diagram Generator initialized. Saving to: {save_dir}")
    
    def create_system_overview(self, save_diagram=True):
        """
        Create high-level system architecture overview
        """
        print("🏗️ Creating system overview diagram...")
        
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 12)
        ax.set_aspect('equal')
        
        # Title
        ax.text(8, 11.5, 'Intelligent Analytics Assistant\nSystem Architecture', 
                ha='center', va='center', fontsize=20, fontweight='bold',
                color=self.colors['text'])
        
        # Input Layer
        self._draw_box(ax, 1, 9, 3, 1.5, "Business Data\nSources", 
                      self.colors['data'], "Sales, Marketing,\nHR Datasets")
        
        # RL Engine Layer  
        self._draw_box(ax, 6, 8.5, 4, 2.5, "Reinforcement Learning\nEngine", 
                      self.colors['rl'], "• Contextual Bandits\n• Multi-Agent RL\n• Q-Learning")
        
        # Agent Layer
        agents = [
            ("Statistical\nAgent", 2, 6),
            ("Geographic\nAgent", 6, 6), 
            ("Temporal\nAgent", 10, 6)
        ]
        
        for name, x, y in agents:
            self._draw_box(ax, x, y, 2.5, 1.5, name, self.colors['agent'])
        
        # Coordinator
        self._draw_box(ax, 6, 3.5, 4, 1.5, "Multi-Agent\nCoordinator", 
                      self.colors['coordinator'], "Orchestrates agent\ncollaboration")
        
        # Communication System
        self._draw_box(ax, 12, 6, 3, 1.5, "Communication\nProtocol", 
                      self.colors['communication'], "Message passing\nInsight sharing")
        
        # Output Layer
        self._draw_box(ax, 6, 0.5, 4, 1.5, "Analytics Insights\n& Recommendations", 
                      self.colors['output'], "Synthesized results\nBusiness intelligence")
        
        # Draw connections
        connections = [
            # Data to RL Engine
            ((4, 9.75), (6, 9.75)),
            # RL Engine to Agents
            ((8, 8.5), (3.25, 7.5)),
            ((8, 8.5), (7.25, 7.5)), 
            ((8, 8.5), (11.25, 7.5)),
            # Agents to Coordinator
            ((3.25, 6), (7, 5)),
            ((7.25, 6), (8, 5)),
            ((11.25, 6), (9, 5)),
            # Agents to Communication
            ((12.5, 6.75), (12, 6.75)),
            # Coordinator to Output
            ((8, 3.5), (8, 2))
        ]
        
        for start, end in connections:
            self._draw_arrow(ax, start, end, self.colors['text'])
        
        # Add legend
        legend_elements = [
            ('Data Sources', self.colors['data']),
            ('RL Components', self.colors['rl']),
            ('Agents', self.colors['agent']),
            ('Coordination', self.colors['coordinator']),
            ('Communication', self.colors['communication']),
            ('Output', self.colors['output'])
        ]
        
        for i, (label, color) in enumerate(legend_elements):
            y_pos = 10.5 - i * 0.3
            ax.add_patch(plt.Rectangle((13.5, y_pos-0.1), 0.3, 0.2, 
                                     facecolor=color, alpha=0.7))
            ax.text(14, y_pos, label, va='center', fontsize=10)
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        plt.tight_layout()
        
        if save_diagram:
            plt.savefig(f"{self.save_dir}/system_architecture.png", 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.savefig(f"{self.save_dir}/system_architecture.pdf", 
                       bbox_inches='tight', facecolor='white')
            print(f"✅ System overview saved to {self.save_dir}")
        
        plt.show()
        return fig, ax
    
    def create_rl_algorithm_diagram(self, save_diagram=True):
        """
        Create detailed RL algorithm flow diagram
        """
        print("🧠 Creating RL algorithm flow diagram...")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        
        # Title
        ax.text(7, 9.5, 'Multi-Agent Reinforcement Learning Flow', 
                ha='center', va='center', fontsize=18, fontweight='bold')
        
        # Algorithm steps
        steps = [
            ("1. Context\nExtraction", 2, 8, "Extract data\ncharacteristics"),
            ("2. Action\nSelection", 6, 8, "UCB Algorithm\nExplore/Exploit"),
            ("3. Agent\nExecution", 10, 8, "Perform analysis\nGenerate insights"),
            ("4. Reward\nCalculation", 12, 6, "Evaluate analysis\nquality"),
            ("5. Q-Value\nUpdate", 10, 4, "Update learning\nQ(s,a) += α(r-Q)"),
            ("6. Communication", 6, 4, "Share insights\nCoordinate agents"),
            ("7. Synthesis", 2, 4, "Combine results\nGenerate recommendations")
        ]
        
        for i, (title, x, y, desc) in enumerate(steps):
            color = self.colors['rl'] if i % 2 == 0 else self.colors['agent']
            self._draw_box(ax, x, y, 2.5, 1.5, title, color, desc)
        
        # Draw flow arrows
        flow = [
            ((4.5, 8), (6, 8)),      # 1 -> 2
            ((8.5, 8), (10, 8)),     # 2 -> 3
            ((11.25, 7), (11.25, 7)), # 3 -> 4 (down)
            ((12, 6), (11.25, 5.5)), # 4 -> 5
            ((10, 4.75), (8.5, 4.75)), # 5 -> 6
            ((6, 4.75), (4.5, 4.75)), # 6 -> 7
            ((2, 5.5), (2, 7)),      # 7 -> 1 (feedback loop)
        ]
        
        for start, end in flow:
            self._draw_arrow(ax, start, end, self.colors['text'], width=2)
        
        # Add mathematical formulations
        math_box_x, math_box_y = 0.5, 2
        ax.add_patch(FancyBboxPatch((math_box_x, math_box_y), 6, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='white', edgecolor=self.colors['rl'], linewidth=2))
        
        math_text = "Key Equations:\n"
        math_text += "UCB: Q(s,a) + c√(ln(t)/N(s,a))\n"  
        math_text += "Q-Learning: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]"
        
        ax.text(math_box_x + 3, math_box_y + 0.75, math_text, 
                ha='center', va='center', fontsize=10, 
                fontfamily='monospace', color=self.colors['text'])
        
        # Add context types
        context_box_x, context_box_y = 8, 2
        ax.add_patch(FancyBboxPatch((context_box_x, context_box_y), 5, 1.5,
                                  boxstyle="round,pad=0.1",
                                  facecolor='white', edgecolor=self.colors['data'], linewidth=2))
        
        context_text = "Context Features:\n"
        context_text += "• Data size (small/medium/large)\n"
        context_text += "• Domain type (sales/marketing/hr)\n" 
        context_text += "• Temporal/Geographic indicators"
        
        ax.text(context_box_x + 2.5, context_box_y + 0.75, context_text,
                ha='center', va='center', fontsize=10, color=self.colors['text'])
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        
        if save_diagram:
            plt.savefig(f"{self.save_dir}/rl_algorithm_flow.png", 
                       dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ RL algorithm diagram saved to {self.save_dir}")
        
        plt.show()
        return fig, ax
    
    def create_agent_interaction_diagram(self, save_diagram=True):
        """
        Create agent interaction and communication diagram
        """
        print("🤝 Creating agent interaction diagram...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        
        # Title
        ax.text(6, 9.5, 'Multi-Agent Communication & Coordination', 
                ha='center', va='center', fontsize=16, fontweight='bold')
        
        # Agent positions (triangle formation)
        agents = [
            ("Statistical\nAgent", 6, 7.5, "Correlation analysis\nStatistical modeling"),
            ("Geographic\nAgent", 3, 4.5, "Regional analysis\nLocation-based insights"),
            ("Temporal\nAgent", 9, 4.5, "Time series analysis\nSeasonal patterns")
        ]
        
        # Draw agents
        agent_positions = {}
        for name, x, y, desc in agents:
            self._draw_box(ax, x, y, 2.5, 1.5, name, self.colors['agent'], desc)
            agent_positions[name.replace('\n', ' ')] = (x, y)
        
        # Coordinator in center
        self._draw_box(ax, 5, 1.5, 3, 1.5, "Multi-Agent\nCoordinator", 
                      self.colors['coordinator'], "Orchestrates\ncollaboration")
        
        # Communication lines between agents (bidirectional)
        comm_lines = [
            (agent_positions["Statistical Agent"], agent_positions["Geographic Agent"]),
            (agent_positions["Statistical Agent"], agent_positions["Temporal Agent"]),
            (agent_positions["Geographic Agent"], agent_positions["Temporal Agent"])
        ]
        
        for (x1, y1), (x2, y2) in comm_lines:
            # Draw bidirectional arrow
            self._draw_double_arrow(ax, (x1, y1), (x2, y2), self.colors['communication'])
            
        # Connections to coordinator
        coordinator_pos = (6.5, 3)
        for name, (x, y) in agent_positions.items():
            self._draw_arrow(ax, (x, y-0.75), coordinator_pos, self.colors['text'])
        
        # Message types legend
        legend_x, legend_y = 0.5, 7
        ax.add_patch(FancyBboxPatch((legend_x, legend_y), 3, 2,
                                  boxstyle="round,pad=0.1",
                                  facecolor='white', edgecolor=self.colors['communication'], 
                                  linewidth=2))
        
        legend_text = "Message Types:\n"
        legend_text += "• Insight sharing\n"
        legend_text += "• Analysis requests\n"
        legend_text += "• Context updates\n"
        legend_text += "• Performance feedback"
        
        ax.text(legend_x + 1.5, legend_y + 1, legend_text,
                ha='center', va='center', fontsize=10, color=self.colors['text'])
        
        # Coordination protocol
        protocol_x, protocol_y = 8.5, 7
        ax.add_patch(FancyBboxPatch((protocol_x, protocol_y), 3, 2,
                                  boxstyle="round,pad=0.1", 
                                  facecolor='white', edgecolor=self.colors['coordinator'],
                                  linewidth=2))
        
        protocol_text = "Coordination Steps:\n"
        protocol_text += "1. Context assessment\n" 
        protocol_text += "2. Agent selection\n"
        protocol_text += "3. Task allocation\n"
        protocol_text += "4. Result synthesis"
        
        ax.text(protocol_x + 1.5, protocol_y + 1, protocol_text,
                ha='center', va='center', fontsize=10, color=self.colors['text'])
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        
        if save_diagram:
            plt.savefig(f"{self.save_dir}/agent_interactions.png", 
                       dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Agent interaction diagram saved to {self.save_dir}")
        
        plt.show()
        return fig, ax
    
    def _draw_box(self, ax, x, y, width, height, title, color, subtitle=""):
        """Draw a styled box with text"""
        # Main box
        box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                           boxstyle="round,pad=0.1", 
                           facecolor=color, alpha=0.7,
                           edgecolor='white', linewidth=2)
        ax.add_patch(box)
        
        # Title text
        ax.text(x, y+0.2, title, ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white')
        
        # Subtitle text
        if subtitle:
            ax.text(x, y-0.3, subtitle, ha='center', va='center',
                   fontsize=8, color='white', alpha=0.9)
    
    def _draw_arrow(self, ax, start, end, color, width=1.5):
        """Draw an arrow between two points"""
        arrow = patches.FancyArrowPatch(start, end,
                                      arrowstyle='->', mutation_scale=20,
                                      linewidth=width, color=color, alpha=0.8)
        ax.add_patch(arrow)
    
    def _draw_double_arrow(self, ax, start, end, color):
        """Draw a bidirectional arrow"""
        # Calculate perpendicular offset for double arrow
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)
        
        if length > 0:
            # Normalize and create perpendicular
            ux, uy = dx/length, dy/length
            offset = 0.1
            px, py = -uy * offset, ux * offset
            
            # Draw two arrows
            start1 = (start[0] + px, start[1] + py)
            end1 = (end[0] + px, end[1] + py)
            start2 = (start[0] - px, start[1] - py) 
            end2 = (end[0] - px, end[1] - py)
            
            self._draw_arrow(ax, start1, end1, color, width=1)
            self._draw_arrow(ax, end2, start2, color, width=1)

def generate_all_diagrams():
    """Generate all architecture diagrams"""
    print("🏗️ Generating all architecture diagrams...")
    
    generator = ArchitectureDiagramGenerator()
    
    # Generate all diagrams
    print("\n1. Creating system overview...")
    generator.create_system_overview()
    
    print("\n2. Creating RL algorithm flow...")
    generator.create_rl_algorithm_diagram()
    
    print("\n3. Creating agent interaction diagram...")
    generator.create_agent_interaction_diagram()
    
    print(f"\n✅ All diagrams generated and saved to {generator.save_dir}")
    print("📁 Generated files:")
    print("   • system_architecture.png/.pdf")
    print("   • rl_algorithm_flow.png") 
    print("   • agent_interactions.png")

if __name__ == "__main__":
    generate_all_diagrams()