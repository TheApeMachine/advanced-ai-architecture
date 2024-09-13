import torch
from sklearn.cluster import KMeans

class StrategyPlanner:
    def __init__(self, memory_bank):
        self.memory_bank = memory_bank
        self.current_strategy = None

    def plan_strategy(self, task_description):
        relevant_memories = self.memory_bank.retrieve_relevant_memories(task_description)
        
        # Cluster memories to identify different strategies
        kmeans = KMeans(n_clusters=min(3, len(relevant_memories)))
        clusters = kmeans.fit_predict([mem.numpy() for mem in relevant_memories])
        
        # Select the cluster with the highest average success score
        cluster_scores = {}
        for i, cluster in enumerate(clusters):
            if cluster not in cluster_scores:
                cluster_scores[cluster] = []
            cluster_scores[cluster].append(self.memory_bank.success_scores[i])
        
        best_cluster = max(cluster_scores, key=lambda c: sum(cluster_scores[c]) / len(cluster_scores[c]))
        
        self.current_strategy = torch.tensor(kmeans.cluster_centers_[best_cluster])
        return self.current_strategy

    def update_strategy(self, outcome):
        if outcome > 0.5:  # Assuming outcome is a success score between 0 and 1
            self.memory_bank.add_memory(self.current_strategy, outcome)
        else:
            self.current_strategy += torch.randn_like(self.current_strategy) * 0.1
