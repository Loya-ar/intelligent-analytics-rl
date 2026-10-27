"""
Memory & Persistence System for RL Analytics Agents
Enables agents to remember learning across sessions
"""

import pickle
import json
import os
from datetime import datetime
import numpy as np
from collections import defaultdict
import sqlite3

class MemoryManager:
    """
    Manages persistent storage and retrieval of agent learning data
    """
    
    def __init__(self, memory_dir="data/memory", db_file="agent_memory.db"):
        self.memory_dir = memory_dir
        self.db_path = os.path.join(memory_dir, db_file)
        
        # Create memory directory
        os.makedirs(memory_dir, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        print(f"💾 Memory Manager initialized. Storage: {memory_dir}")
    
    def _init_database(self):
        """Initialize SQLite database for learning history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Agent learning table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_learning (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL,
                    episode INTEGER,
                    context TEXT,
                    action TEXT,
                    reward REAL,
                    q_value REAL,
                    timestamp TEXT,
                    session_id TEXT
                )
            ''')
            
            # Agent state table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_state (
                    agent_name TEXT PRIMARY KEY,
                    total_reward REAL,
                    episode_count INTEGER,
                    action_values TEXT,
                    action_counts TEXT,
                    last_updated TEXT
                )
            ''')
            
            # Coordination history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS coordination_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id TEXT,
                    participating_agents TEXT,
                    results TEXT,
                    timestamp TEXT,
                    session_id TEXT
                )
            ''')
            
            conn.commit()
            print("💾 Database initialized successfully")
    
    def save_agent_state(self, agent, session_id=None):
        """Save complete agent state to persistent storage"""
        
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Serialize complex data structures
        action_values_json = json.dumps(dict(agent.action_values))
        action_counts_json = json.dumps(dict(agent.action_counts))
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert or update agent state
            cursor.execute('''
                INSERT OR REPLACE INTO agent_state
                (agent_name, total_reward, episode_count, action_values, action_counts, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                agent.name,
                agent.total_reward,
                len(agent.learning_history) if hasattr(agent, 'learning_history') else 0,
                action_values_json,
                action_counts_json,
                datetime.now().isoformat()
            ))
            
            # Save learning history
            if hasattr(agent, 'learning_history'):
                for entry in agent.learning_history:
                    cursor.execute('''
                        INSERT INTO agent_learning
                        (agent_name, episode, context, action, reward, q_value, timestamp, session_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        agent.name,
                        entry.get('episode', 0),
                        entry.get('context', ''),
                        entry.get('action', ''),
                        entry.get('reward', 0),
                        entry.get('q_value', 0),
                        datetime.now().isoformat(),
                        session_id
                    ))
            
            conn.commit()
        
        print(f"💾 Saved {agent.name} state to persistent memory")
        return session_id
    
    def load_agent_state(self, agent):
        """Load agent state from persistent storage"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Load agent state
            cursor.execute('''
                SELECT total_reward, episode_count, action_values, action_counts, last_updated
                FROM agent_state WHERE agent_name = ?
            ''', (agent.name,))
            
            result = cursor.fetchone()
            
            if result:
                total_reward, episode_count, action_values_json, action_counts_json, last_updated = result
                
                # Restore agent state
                agent.total_reward = total_reward
                agent.action_values = defaultdict(float, json.loads(action_values_json))
                agent.action_counts = defaultdict(int, json.loads(action_counts_json))
                
                # Load learning history
                cursor.execute('''
                    SELECT episode, context, action, reward, q_value, timestamp
                    FROM agent_learning WHERE agent_name = ?
                    ORDER BY timestamp
                ''', (agent.name,))
                
                learning_entries = cursor.fetchall()
                
                if not hasattr(agent, 'learning_history'):
                    agent.learning_history = []
                
                for episode, context, action, reward, q_value, timestamp in learning_entries:
                    agent.learning_history.append({
                        'episode': episode,
                        'context': context,
                        'action': action,
                        'reward': reward,
                        'q_value': q_value,
                        'timestamp': timestamp
                    })
                
                print(f"💾 Loaded {agent.name} state from memory (last updated: {last_updated})")
                print(f"   Restored: {total_reward:.2f} total reward, {len(learning_entries)} learning episodes")
                
                return True
            else:
                print(f"💾 No saved state found for {agent.name}")
                return False
    
    def save_coordination_history(self, coordinator, session_id=None):
        """Save coordination episodes to persistent storage"""
        
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for i, episode in enumerate(coordinator.coordination_history):
                episode_id = f"{session_id}_ep{i+1}"
                
                participating_agents = json.dumps(episode['participating_agents'])
                results = json.dumps(episode.get('results', {}), default=str)  # Handle numpy types
                
                cursor.execute('''
                    INSERT INTO coordination_history
                    (episode_id, participating_agents, results, timestamp, session_id)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    episode_id,
                    participating_agents,
                    results,
                    datetime.now().isoformat(),
                    session_id
                ))
            
            conn.commit()
        
        print(f"💾 Saved {len(coordinator.coordination_history)} coordination episodes")
        return session_id
    
    def export_learning_data(self, format='json', filename=None):
        """Export all learning data to various formats"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"learning_export_{timestamp}"
        
        with sqlite3.connect(self.db_path) as conn:
            
            if format.lower() == 'json':
                # Export as JSON
                export_data = {
                    'export_timestamp': datetime.now().isoformat(),
                    'agents': {},
                    'coordination_episodes': []
                }
                
                # Get all agent data
                cursor = conn.cursor()
                cursor.execute('SELECT DISTINCT agent_name FROM agent_learning')
                agents = [row[0] for row in cursor.fetchall()]
                
                for agent_name in agents:
                    cursor.execute('''
                        SELECT episode, context, action, reward, q_value, timestamp
                        FROM agent_learning WHERE agent_name = ?
                        ORDER BY timestamp
                    ''', (agent_name,))
                    
                    learning_history = []
                    for row in cursor.fetchall():
                        learning_history.append({
                            'episode': row[0],
                            'context': row[1],
                            'action': row[2],
                            'reward': row[3],
                            'q_value': row[4],
                            'timestamp': row[5]
                        })
                    
                    export_data['agents'][agent_name] = {
                        'learning_history': learning_history
                    }
                
                # Get coordination data
                cursor.execute('SELECT episode_id, participating_agents, results, timestamp FROM coordination_history ORDER BY timestamp')
                for row in cursor.fetchall():
                    export_data['coordination_episodes'].append({
                        'episode_id': row[0],
                        'participating_agents': json.loads(row[1]),
                        'results': json.loads(row[2]),
                        'timestamp': row[3]
                    })
                
                # Save JSON file
                json_path = os.path.join(self.memory_dir, f"{filename}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2)
                
                print(f"💾 Exported learning data to {json_path}")
                return json_path
            
            elif format.lower() == 'csv':
                # Export as CSV
                import pandas as pd
                
                # Agent learning data
                learning_df = pd.read_sql_query('''
                    SELECT agent_name, episode, context, action, reward, q_value, timestamp, session_id
                    FROM agent_learning ORDER BY timestamp
                ''', conn)
                
                csv_path = os.path.join(self.memory_dir, f"{filename}_learning.csv")
                learning_df.to_csv(csv_path, index=False)
                
                # Coordination data
                coordination_df = pd.read_sql_query('''
                    SELECT episode_id, participating_agents, timestamp, session_id
                    FROM coordination_history ORDER BY timestamp
                ''', conn)
                
                coord_csv_path = os.path.join(self.memory_dir, f"{filename}_coordination.csv")
                coordination_df.to_csv(coord_csv_path, index=False)
                
                print(f"💾 Exported learning data to CSV files:")
                print(f"   Learning: {csv_path}")
                print(f"   Coordination: {coord_csv_path}")
                
                return [csv_path, coord_csv_path]
    
    def get_learning_statistics(self):
        """Get comprehensive learning statistics"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            stats = {
                'timestamp': datetime.now().isoformat(),
                'agents': {},
                'system': {}
            }
            
            # Agent statistics
            cursor.execute('SELECT DISTINCT agent_name FROM agent_learning')
            agents = [row[0] for row in cursor.fetchall()]
            
            for agent_name in agents:
                cursor.execute('''
                    SELECT 
                        COUNT(*) as episode_count,
                        AVG(reward) as avg_reward,
                        SUM(reward) as total_reward,
                        MIN(q_value) as min_q_value,
                        MAX(q_value) as max_q_value,
                        AVG(q_value) as avg_q_value
                    FROM agent_learning WHERE agent_name = ?
                ''', (agent_name,))
                
                result = cursor.fetchone()
                if result:
                    stats['agents'][agent_name] = {
                        'episode_count': result[0],
                        'avg_reward': round(result[1], 3) if result[1] else 0,
                        'total_reward': round(result[2], 3) if result[2] else 0,
                        'min_q_value': round(result[3], 3) if result[3] else 0,
                        'max_q_value': round(result[4], 3) if result[4] else 0,
                        'avg_q_value': round(result[5], 3) if result[5] else 0
                    }
            
            # System statistics
            cursor.execute('SELECT COUNT(*) FROM coordination_history')
            total_episodes = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT session_id) FROM agent_learning')
            total_sessions = cursor.fetchone()[0]
            
            stats['system'] = {
                'total_coordination_episodes': total_episodes,
                'total_learning_sessions': total_sessions,
                'agents_tracked': len(agents)
            }
            
            return stats
    
    def clear_memory(self, confirm=False):
        """Clear all persistent memory (use with caution)"""
        
        if not confirm:
            print("⚠️  Memory clear requires confirmation. Use clear_memory(confirm=True)")
            return False
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM agent_learning')
            cursor.execute('DELETE FROM agent_state')
            cursor.execute('DELETE FROM coordination_history')
            
            conn.commit()
        
        print("🗑️  All persistent memory cleared")
        return True

def test_memory_system():
    """Test the memory and persistence system"""
    print("🧪 Testing Memory & Persistence System...")
    
    # Create memory manager
    memory_manager = MemoryManager()
    
    # Create a mock agent for testing
    class MockAgent:
        def __init__(self, name):
            self.name = name
            self.total_reward = 15.5
            self.action_values = defaultdict(float, {'context1': 0.8, 'context2': 1.2})
            self.action_counts = defaultdict(int, {'context1': 3, 'context2': 2})
            self.learning_history = [
                {'episode': 1, 'context': 'context1', 'action': 'analyze', 'reward': 1.5, 'q_value': 0.15},
                {'episode': 2, 'context': 'context2', 'action': 'analyze', 'reward': 2.0, 'q_value': 0.35}
            ]
    
    # Test saving and loading
    test_agent = MockAgent("Test_Agent")
    
    print("\n1. Testing agent state saving...")
    session_id = memory_manager.save_agent_state(test_agent)
    
    print("\n2. Testing agent state loading...")
    new_agent = MockAgent("Test_Agent")
    new_agent.total_reward = 0  # Reset to test loading
    
    success = memory_manager.load_agent_state(new_agent)
    
    if success:
        print(f"✅ Loaded agent state: {new_agent.total_reward} reward")
        print(f"✅ Loaded action values: {dict(new_agent.action_values)}")
        print(f"✅ Loaded learning history: {len(new_agent.learning_history)} episodes")
    
    print("\n3. Testing learning statistics...")
    stats = memory_manager.get_learning_statistics()
    print(f"📊 Statistics: {stats}")
    
    print("\n4. Testing data export...")
    export_path = memory_manager.export_learning_data(format='json')
    
    print("\n✅ Memory system test complete!")
    return memory_manager

if __name__ == "__main__":
    test_memory_system()