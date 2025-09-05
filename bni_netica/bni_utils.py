from bni_netica import *
from collections import deque

def findAllDConnectedNodes(bn, source_node, dest_node, o=None):
    o = dict(o or {})
    o.setdefault("arcTraversal", False)
    o["arcs"] = o["arcTraversal"] or o.get("arcs", False)
    o.setdefault("noSourceParents", False)
    o.setdefault("noSourceChildren", False)

    if isinstance(source_node, str):
        source_node = bn.node(source_node)
    if isinstance(dest_node, str):
        dest_node = bn.node(dest_node)

    PARENTS = 0b01
    CHILDREN = 0b10
    PARENTSANDCHILDREN = 0b11

    def has_loop(path, end):
        end_node = end[0]
        for n, _d in reversed(path):
            if end_node is n:
                return True
        return False

    # Map GeNIe BN to a lightweight DAG view
    class DagNode:
        __slots__ = ("id", "parents", "children", "intervene", "_src")
        def __init__(self, n):
            self.id = n.name()
            self.parents = list(n.parents())
            self.children = list(n.children())
            self.intervene = False
            self._src = n
        def hasEvidence(self):
            return self._src.finding() is not None
        def getDescendants(self):
            descendants = set()
            to_visit = deque([self])
            while to_visit:
                nx = to_visit.popleft()
                for c in nx.children:
                    if c not in descendants:
                        descendants.add(c)
                        to_visit.append(c)
            return list(descendants)
        def isParent(self, toNode):
            return toNode in self.children

    def make_dag(bn):
        dag = {n.name(): DagNode(n) for n in bn.nodes()}
        # rewire parents/children to DagNode refs
        for dn in dag.values():
            dn.parents = [dag[p.name()] for p in dn.parents]
            dn.children = [dag[c.name()] for c in dn.children]
        return dag

    dag = make_dag(bn)
    s = dag[source_node.name()]
    t = dag[dest_node.name()]

    downstream_ev = {
        n.id: sum(1 for d in n.getDescendants() if d.hasEvidence())
        for n in dag.values()
    }

    paths_q = deque([[(s, PARENTSANDCHILDREN)]])
    resolved = {}
    arcs = set()

    def node_dir_key(node_dir):
        return f"{node_dir[0].id}-{node_dir[1]}"

    def resolve_path(path, how_resolved=True):
        prev = None
        for curr in path:
            if prev is not None:
                par, child = prev[0], curr[0]
                swap = child.isParent(par)
                if swap:
                    par, child = child, par
                if o["arcTraversal"]:
                    arcs.add(f"{par.id}-{child.id}|{'up' if swap else 'down'}")
                else:
                    arcs.add(f"{par.id}-{child.id}")
            resolved[node_dir_key(curr)] = curr[0] if how_resolved else False
            prev = curr

    def node_dir_is_resolved(node_dir):
        return bool(resolved.get(node_dir_key(node_dir)))

    while paths_q:
        current_path = paths_q.popleft()
        if node_dir_is_resolved(current_path[-1]):
            resolve_path(current_path)
            continue

        current_node, dirmask = current_path[-1]
        if current_node is t:
            resolve_path(current_path)
        else:
            check_parents = (not current_node.intervene) and (not o["noSourceParents"] or current_node is not s)
            check_children = (not o["noSourceChildren"] or current_node is not s)

            if check_parents and (dirmask & PARENTS) == PARENTS:
                next_paths = [
                    current_path + [(p, PARENTSANDCHILDREN)]
                    for p in current_node.parents
                    if (not p.hasEvidence()) and (not has_loop(current_path, (p, PARENTSANDCHILDREN)))
                ]
                paths_q.extend(next_paths)

            if check_children and (dirmask & CHILDREN) == CHILDREN:
                def child_dir(c):
                    return PARENTS if c.hasEvidence() else (PARENTSANDCHILDREN if downstream_ev[c.id] else CHILDREN)
                next_paths = [
                    current_path + [(c, child_dir(c))]
                    for c in current_node.children
                    if not has_loop(current_path, (c, child_dir(c)))
                ]
                paths_q.extend(next_paths)

    if o["arcs"]:
        return list(arcs)
    uniq_nodes = {n for n in resolved.values() if n}
    return [bn.node(n.id) for n in uniq_nodes]

