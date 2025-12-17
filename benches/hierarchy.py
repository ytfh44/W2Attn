import torch
import random

class HierarchyGenerator:
    def __init__(self, vocab_size=1000, branching_factor=4, max_depth=4):
        self.vocab_size = vocab_size
        # 0 is root. Remaining nodes are pool.
        self.pool = list(range(1, vocab_size))
        random.shuffle(self.pool)
        
        self.edges = {} # child -> parent
        self.children = {0: []} # parent -> [children]
        self.depths = {0: 0}
        self.active_nodes = [0]
        
        # BFS Queue for tree building
        # (node_id)
        queue = [0]
        
        while queue and self.pool:
            parent = queue.pop(0)
            if self.depths[parent] >= max_depth:
                continue
            
            # Decide how many children this node gets
            num_kids = random.randint(1, branching_factor)
            for _ in range(num_kids):
                if not self.pool:
                    break
                child = self.pool.pop()
                
                self.edges[child] = parent
                if parent not in self.children:
                    self.children[parent] = []
                self.children[parent].append(child)
                # Init child entry
                self.children[child] = []
                
                self.depths[child] = self.depths[parent] + 1
                self.active_nodes.append(child)
                queue.append(child)
                
        print(f"Generated Tree with {len(self.active_nodes)} nodes. Max Depth: {max(self.depths.values())}")

    def is_ancestor(self, node_a, node_b):
        # Is B an ancestor of A?
        curr = node_a
        # Use set for infinite loop protection (DAG safety, though it's a tree)
        seen = {curr}
        while curr in self.edges:
            curr = self.edges[curr]
            if curr == node_b:
                return True
            if curr in seen:
                break
            seen.add(curr)
        return False

    def get_batch(self, batch_size, device):
        # x: [batch, 2] -> [Child, Potential_Ancestor]
        # y: [batch] -> 1 (Yes) or 0 (No)
        
        x_list = []
        y_list = []
        
        half = batch_size // 2
        
        # Positives (Child, Ancestor)
        for _ in range(half):
            child = random.choice(self.active_nodes)
            # Find ancestors
            ancestors = []
            curr = child
            while curr in self.edges:
                curr = self.edges[curr]
                ancestors.append(curr)
                
            if ancestors:
                anc = random.choice(ancestors)
                x_list.append([child, anc])
                y_list.append(1)
            else:
                # Root or orphan, no ancestors, treat as negative case fallback
                rand_node = random.choice(self.active_nodes)
                x_list.append([child, rand_node])
                y_list.append(0)

        # Negatives (Child, Random)
        for _ in range(batch_size - len(x_list)):
            child = random.choice(self.active_nodes)
            rand_node = random.choice(self.active_nodes)
            
            # Verify it's not actually an ancestor by luck
            if self.is_ancestor(child, rand_node):
                y_list.append(1)
            else:
                y_list.append(0)
            x_list.append([child, rand_node])
            
        x = torch.tensor(x_list, dtype=torch.long, device=device)
        y = torch.tensor(y_list, dtype=torch.long, device=device)
        return x, y
