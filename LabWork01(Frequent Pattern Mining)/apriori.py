import sys, os, psutil
import timeit
import collections
import itertools
from trie import Trie

def load_dataset(file_name):
    # print("here we will load our dataset")
    dataset = [sorted(int(n) for n in i.strip().split())
                for i in open(file_name).readlines()]
    
    size = len(dataset)
    print("Size of the Dataset: ", size)

    total_len = 0

    for i in range(len(dataset)):
        total_len += len(dataset[i])

    avg_len = total_len/size
    print("Average Transaction length: ", avg_len)

    return dataset


def find_frequent1_itemsets(dataset, minSupport):
    frequency = dict(collections.Counter(
        itertools.chain.from_iterable(dataset)
    ))

    level1 = dict()

    for item, freq in frequency.items():
        if(freq>=minSupport):
            level1[item] = freq
    
    return level1


def has_infrequent_subset(candidate: tuple, L: list):
    for subset in list(itertools.combinations(candidate, len(candidate)-1)):
        if list(subset) not in L:
            return True
        
    return False



def apriori_gen(L: list, k):
    # self join
    l_next = list()

    for l1 in L:
        for l2 in L:
            if(len(set(l1) & set(l2))==k-1):
                l_next.append(sorted(list(set(l1) | set(l2))))

    # removing duplicates
    l_set = set(tuple(x) for x in l_next)
    l_k1  =[list(x) for x in l_set]

    l_k1.sort(key=lambda x: l_next.index(x))
    l_k1_tuple = [tuple(i) for i in l_k1]
    info = {'join': len(l_k1)}

    # prune 
    for c in l_k1_tuple:
        if has_infrequent_subset(c, L):
            l_k1.remove(list(c))

    info['prune'] = len(l_k1)

    return info, l_k1



def apriori(db: list, minSupport):
    minSupport = (len(db)*minSupport)//100

    levels = list()
    levels_info = list()

    level1 = find_frequent1_itemsets(db, minSupport)
    # print(level1)

    if bool(level1)==False:
        print("Minimum support threshold doesn't satifies 1-Itemset")
        return None

    _level1 = [[k] for k in level1.keys()]
    _level1 = sorted(_level1)

    # print(level1)
    # print(_level1)

    levels.append(_level1)
    levels_info.append({ 'join': len(_level1), 'prune': len(_level1)})

    # print(levels)
    # print(levels_info)

    while True:
        info, candidates = apriori_gen(levels[-1], len(levels[-1][0]))

        # print(info)
        # print(candidates)
        # break

        trie = Trie(db)
        trie.buildTrie(candidates)
        trie.assign_frequency()

        L = list()

        for itemset in candidates:
            if trie.get_candidate_freq(itemset)>=minSupport:
                L.append(sorted(itemset))

        if not L:
            break

        levels.append(L)
        levels_info.append(info)

    return levels_info, levels


def get_process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss



if __name__=="__main__":
    db = load_dataset(str(sys.argv[1]))
    minSupport = float(sys.argv[2])

    print("Dataset: ", str(sys.argv[1]))
    print("Min support: ", minSupport, "%")
    print("Min support count: ", (len(db)*minSupport)//100)

    start_time = timeit.default_timer()
    mem_before = get_process_memory()
    print("Running...")

    info, L = apriori(db, minSupport)
    
    end_time = timeit.default_timer()
    mem_after = get_process_memory()

    total_pattern = 0
    total_join = 0
    total_prune = 0
    
    print('Level', '    After Join', '  After Pruning', '   Frequent Itemset')
    if L is not None:
        for i in range(len(L)):
            print()
            print((i+1), info[i]['join'], info[i]['prune'], len(L[i]), sep='\t\t')

            total_pattern += len(L[i])
            total_join += info[i]['join']
            total_prune += info[i]['prune']


    print('\nTotal Join: ', total_join, '\nTotal Prune: ', total_prune, '\nTotal pattern: ',total_pattern)
    print("Time Used: ", end_time-start_time,"seconds")
    print("Memory before: {:,}bytes, After: {:,}bytes, Consumed: {:,}bytes;".format(
            mem_before, mem_after, mem_after - mem_before,))