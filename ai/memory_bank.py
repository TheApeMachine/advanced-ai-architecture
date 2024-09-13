import numpy as np
import torch.nn.functional as F

class MemoryBank:
    def __init__(self, capacity=1000, memory_dimension=10):
        self.capacity = capacity
        self.memory_dimension = memory_dimension
        self.memories = np.zeros((capacity, memory_dimension), dtype=np.float32)
        self.success_scores = np.zeros(capacity, dtype=np.float32)
        self.current_index = 0
        self.is_full = False

    def add_memory(self, memory, success_score):
        self.memories[self.current_index] = memory
        self.success_scores[self.current_index] = success_score
        self.current_index = (self.current_index + 1) % self.capacity
        if self.current_index == 0:
            self.is_full = True

    def retrieve_relevant_memories(self, query, k=5):
        if not self.is_full:
            memories = self.memories[:self.current_index]
            scores = self.success_scores[:self.current_index]
        else:
            memories = self.memories
            scores = self.success_scores
        # Compute similarities
        query = query / np.linalg.norm(query)
        memories_norm = memories / np.linalg.norm(memories, axis=1, keepdims=True)
        similarities = np.dot(memories_norm, query)
        top_k_indices = np.argsort(-similarities)[:k]
        return memories[top_k_indices]