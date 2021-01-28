class Node:
    def __init__(self, name):
        self.frequency = 0
        self.isLeaf = True
        self.children = {}
        self.name = name
    
    def addChild(self, name: int, child):
        self.isLeaf = False
        self.children[name] = child


class Trie:
    def __init__(self, db: list):
        self.db = db
        self.root_node = Node(-1)

    
    def buildTrie(self, candidates: list):
        for cl in candidates:
            self.create_branch(self.root_node, cl, 0)
    

    def create_branch(self, current_node: Node, cl: list, idx):
        if(idx==len(cl)):
            return

        if(cl[idx] in current_node.children.keys()):
            self.create_branch(current_node.children[cl[idx]], cl, idx+1)
        else:
            current_node.addChild(cl[idx], Node(cl[idx]))
            self.create_branch(current_node.children[cl[idx]], cl, idx+1)

    def print_res(self, node: Node):
        if(len(node.children)==0):
            return
        else:
            print(len(node.children))
            for c, nn, in node.children.items():
                print(c, "parent: ", node.name, "frequency: ", nn.frequency, "leaf: ", nn.isLeaf)
                print(nn.children)
                self.print_res(nn)

    def assign_frequency(self):
        for trx in self.db:
            self.traverse(self.root_node, set(trx))

    def traverse(self, current_node: Node, trx):
        if current_node.isLeaf:
            current_node.frequency +=1
            return

        for child in current_node.children.keys():
            if child in trx:
                self.traverse(current_node.children[child], trx)    

    def get_candidate_freq(self, candidate):
        return self.single_traverse(self.root_node, candidate, 0)

    def single_traverse(self, current_node, candidate, idx):
        if current_node.isLeaf:
            return current_node.frequency
        return self.single_traverse(current_node.children[candidate[idx]], candidate, idx+1)
        