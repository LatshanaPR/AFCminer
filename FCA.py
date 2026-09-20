'''The below implementation uses incidence matrix as mentioned in the paper which has the node to node relation and node 
to attribute relation as well
for example if there are two nodes and each node has two attributes gender(male,female) and year(2006,2007)
   1  2 (M,2006) (M,2007) (F,2006) (F,2007)
1  1  1    1        0         0        0
2  1  1    0        0         0        1

This is how the matrix is represented in a dictionary of dictionary
'''
#the following fucntion implementation is not directly given in the paper so its implemented with the idea of intent
def derive_intent(extent, candidate_intents, C,NEIGHBORS):
    if not extent:
        return candidate_intents
    # Bitwise intersection in C instead of double Python loops
    res = candidate_intents
    for x in extent:
        res = res & NEIGHBORS[x]
    return res
#the following function is not directly given in the paper so its implemented with the idea of extent
def derive_extent(intent, candidate_extents, C,NEIGHBORS):
    if not intent:
        return candidate_extents
    res = candidate_extents
    for b in intent:
        res = res & NEIGHBORS[b]
    return res
#the following function iteratively finds the next valid concept from the previous basic concept
#it takes the initial basic conocept whose extent is of single node for example node 2's extent was {1,2,3}
#so now this function takes the intersection of two nodes extent then now it explores two nodes
#then it takes intersection of the concepts generated through two nodes intersection which can give 4 and 3 node concepts
#Its an BFS technically for which the visited is conceptset and the everytime the next level of the tree is stored in next_concept for exploring
# lets see how level by level its explored
# level 1:(1) (2) (3) (4)
# level 2:(1,2) (1,3) (1,4) (2,3) (2,4) (3,4)
# level 3:(1,2,3) (1,2,4) (1,3,4)
#But not all the combination is explored as two nodes may never share any node and the path can be pruned
#for example if 1,2 doesnt have common node then 1,2,3 and 1,2,4 cant exist and as we use conceptset as visited 
#we wont explore same path second time
def AddConcept(C,conceptset,V,attributes,NEIGHBORS):
    #this cur is the same variable conceptset' in the paper and this stores the current level of nodes
    cur_concept=set(conceptset)
    #the nex is the same variable conceptset'' in the paper and is used to store next level of exploration
    next_concept=set()
    #using two loops to get pair of concepts from the cur
    count=0
    while cur_concept:
        print(f"Level {count}: {len(cur_concept)} concepts")
        cur_list = list(cur_concept)
        #the following is done to optimize the run time
        node_to_concepts={}
        for idx, concept in enumerate(cur_list):
            _,y,_=concept
            for node in y:
                if node not in node_to_concepts:
                    node_to_concepts[node]=[]
                node_to_concepts[node].append(idx)
        candidate_pairs = set()
        for node,indices in node_to_concepts.items():
            n_len=len(indices)
            for i in range(n_len):
                idx1=indices[i]
                for j in range(i + 1, n_len):
                    idx2=indices[j]
                    if idx1<idx2:
                        candidate_pairs.add((idx1, idx2))
                    else:
                        candidate_pairs.add((idx2, idx1))
        for idx1, idx2 in candidate_pairs:
            x1, y1, B1 = cur_list[idx1]
            x2, y2, B2 = cur_list[idx2]
            if y1 <= y2 or y2 <= y1:continue
            y=y1 & y2
            if len(y)<2:continue
            full_objects=V
            extent = frozenset(derive_extent(y, full_objects, C,NEIGHBORS))
            intent = frozenset(derive_intent(extent, full_objects, C,NEIGHBORS))
            
            B = frozenset(derive_intent(extent, attributes, C,NEIGHBORS))
            if (extent,intent,B) not in conceptset:
                conceptset.add((extent,intent,B))
                next_concept.add((extent,intent,B))
        cur_concept=next_concept
        next_concept=set()
        count+=1
    return conceptset
#this function is used to find the initial basic concept
#lets understand this with an example now a graph has four nodes 0 1 2 3
#and the modified adjacency matrix is
#[1 0 1 1]
#[0 1 1 1]
#[1 1 1 0]
#[1 1 0 1]
#for j=2 we will first find the extent of 2 which is just the nodes 2 is connected to so extent={1,2,3}
#now then we find the intent for {1,2,3} which is {2} so one of the basic concept is ({1,2,3},{2})
def BasicConcept(C,V,attributes,NEIGHBORS):
    conceptset=set()
    full_objects=V
    
    #here two loops arent used as in the paper cause the two loops in paper is meant to 
    #understand the timecomplexity as V^2 but here its replaced by one loop.
    #The paper uses (cij(intent) , cij) but cij value is 1 or 0 based on the connectivity.
    # In the paper cij refers to extent of either i or j so its implemented with single loop but still its V^2 because
    # we are finding extent which time complexity is V.  
    for j in full_objects:
        intent = frozenset(derive_intent({j}, full_objects, C,NEIGHBORS))
        extent = frozenset(derive_extent(intent, full_objects, C,NEIGHBORS))
        #B is just used to notate whether a particular set of nodes have same attribute which is just used visually in lattice
        #for example if node 1,2,3 is the intent and its all male then B is {M} if all those have different attribute value
        #then its empty or null as represented in the paper. 
        B = frozenset(derive_intent(extent, attributes, C,NEIGHBORS))
        if (extent,intent,B) not in conceptset:
            conceptset.add((extent,intent,B))
    return conceptset


def ConceptBuilder(Matrix,V,node_attribute_set):
    NEIGHBORS = {
    u: frozenset(v for v in Matrix[u] if Matrix[u][v] == 1)
    for u in Matrix
    }
    #V and node_attribute_set is being used to implement the attributed concept generation from AFCMiner
    V=set(V)
    node_attribute_set=set(node_attribute_set)
    attributes=node_attribute_set-V 
    conceptset=BasicConcept(Matrix,V,attributes,NEIGHBORS)
    AddConcept(Matrix,conceptset,V,attributes,NEIGHBORS)
    return conceptset
