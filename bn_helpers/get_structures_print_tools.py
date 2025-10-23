from bni_netica.bni_netica import Net
from pathlib import Path

def get_nets(is_debug=False):
    base_dir = Path(__file__).resolve().parent.parent
    bn_path = base_dir / "nets" / "collection"
    netDir = base_dir / "nets"
    
    net_list = []
    
    # Automatically load all BN files from the collection folder
    bn_extensions = ['.dne', '.neta']  # Supported BN file extensions
    for file_path in bn_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in bn_extensions:
            try:
                net = Net(str(file_path))
                net_list.append(net)
                if is_debug:
                    print(f"Loaded: {file_path.name}")
            except Exception as e:
                if is_debug:
                    print(f"Failed to load {file_path.name}: {e}")
    
    # Load additional specific networks from other locations
    additional_nets = [
        netDir / "outputs" / "common_cause_effect.neta",
        netDir / "NF_V1.dne"
    ]
    
    for net_path in additional_nets:
        if net_path.exists():
            try:
                net = Net(str(net_path))
                net_list.append(net)
                if is_debug:
                    print(f"Loaded: {net_path.name}")
            except Exception as e:
                if is_debug:
                    print(f"Failed to load {net_path.name}: {e}")
    
    return net_list

def printNet(net):
    for node in net.nodes():
        print(f"{node.name()} -> {[child.name() for child in node.children()]}")

def get_BN_structure(net):
    structure = ""
    for node in net.nodes():
        children_names = [child.name() for child in node.children()]
        structure += f"{node.name()} -> {children_names}\n"
    return structure

def get_BN_node_states(net):
    structure = ""
    for node in net.nodes():
        states = [s.name() for s in node.states()]
        structure += f"{node.name()} {states}\n"
    return structure

def printPath(path):
    ans = ""
    for node in path[:-1]:
        ans += f"{node} -> "
    ans += f"{path[-1]}"
    return ans

def getNetCPTStrings(net):
    cpt_strings = {}
    for node in net.nodes():
        cpt_data = node.cpt()
        if isinstance(cpt_data, list):
            # Handle list format - convert to string representation
            cpt_strings[node.name()] = str(cpt_data)
        else:
            # Handle dict format (if it exists)
            cpt_strings[node.name()] = "\n".join([f"{k}: {v}" for k, v in cpt_data.items()])
    return cpt_strings