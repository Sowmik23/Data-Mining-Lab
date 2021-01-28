import sys
import timeit
import itertools
from apriori import load_dataset, find_frequent1_itemsets, get_process_memory

class Node:
    def __init__(self, name: int, parent):
        self.frequency = 0
        self.isLeaf = True
        self.children = {}
        self.name = name
        self.parent = parent
    
    def addChild(self, name: int, child):
        self.isLeaf = False
        self.children[name] = child
        child.frequency = 1


class FPTree:
    def __init__(self, db: list, minSupport):
        self.db = db
        self.minSupport = minSupport
        self.root_node = Node(-1, None)
        self.ordered1_itemset = self.get_ordered1_itemset()
        self.node_link = {k: [] for k in self.ordered1_itemset}


    def get_ordered1_itemset(self):
        L1 = find_frequent1_itemsets(self.db, self.minSupport)
        sorted_L1 = sorted(L1, key=L1.get, reverse=True)

        return sorted_L1
    
    def buildFPTree(self):
        for trx in self.db:
            if len(trx)==0:
                continue
            trx = set(trx)
            _trx = []
            for i in self.ordered1_itemset:
                if i in trx:
                    _trx.append(i)
            self.create_branch(self.root_node, _trx, 0)
    
    def create_branch(self, current_node: Node, trx: list, idx):
        if idx==len(trx):
            return 
        if trx[idx] in current_node.children.keys():
            next_node = current_node.children[trx[idx]]
            next_node.frequency +=1
            self.create_branch(next_node, trx, idx+1)
        else:
            child = Node(trx[idx], current_node)
            current_node.addChild(trx[idx], child)
            self.node_link[child.name].append(child)
            self.create_branch( current_node.children[trx[idx]], trx, idx+1)

    def _print(self, node: Node):
        if len(node.children)==0:
            return
        else:
            for c, nn in node.children.items():
                print(c, "parent: ", nn,parent.name, "paretn_freq: ", nn.parent_frequency, "frequency: ", nn.frequency)

                self._print(nn)
            
    def get_conditional_db(self):
        conditional_db = dict()
        conditional_db = {k: [] for k in self.ordered1_itemset}

        for i in reversed(self.ordered1_itemset):
            for j in self.node_link[i]:
                cond_trx = self.get_predecessors(j, [])

                for l in range(j.frequency):
                    conditional_db[i].append(cond_trx)
        
        return conditional_db


    def get_predecessors(self, current_node: Node, pred_list: list):
        if current_node.parent==self.root_node:
            return pred_list

        pred_list.append(current_node.parent.name)
        return self.get_predecessors(current_node.parent, pred_list)


cond_cnt = 0
def fp(item, cond_db: list, minSupport):
    l2 = [[item]]
    global cond_cnt

    fpt = FPTree(cond_db, minSupport)
    fpt.buildFPTree()

    for li in cond_db:
        if li!=[]:
            break
        else:
            cond_cnt +=1

    _cond_db_sub = fpt.get_conditional_db()

    for item, db in _cond_db_sub.items():
        l = fp(item, db, minSupport)

        for i in l:
            i.append(item)
            l2.append(i)

    return l2


def fp_growth(db: list, minSupport):
    # print(minSupport)
    freq = 0
    fp_tree = FPTree(db, minSupport)
    fp_tree.buildFPTree()

    proj_db_dict = fp_tree.get_conditional_db()

    for item1, cond_db in proj_db_dict.items():
        ls = fp(item1, cond_db, minSupport)
        freq += len(ls)
    
    print("Total number of frequent patterns: ", freq)




if __name__=="__main__":
    db = load_dataset(str(sys.argv[1]))
    minSupport = float(sys.argv[2])

    print("Dataset: ", str(sys.argv[1]))
    print("Min support: ", minSupport, "%")
    minSupportCount = (len(db)*minSupport)//100
    print("Min support count: ", minSupportCount)

    start_time = timeit.default_timer()
    mem_before = get_process_memory()

    print("Running...")
    fp_growth(db, minSupportCount)

    mem_after = get_process_memory()
    end_time = timeit.default_timer()

    print("Time Used: ", end_time-start_time,"seconds")
    print("Memory before: {:,}bytes, After: {:,}bytes, Consumed: {:,}bytes;".format(
            mem_before, mem_after, mem_after - mem_before,))