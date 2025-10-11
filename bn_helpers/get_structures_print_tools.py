from bni_netica.bni_netica import Net
from pathlib import Path

def get_nets():
    base_dir = Path(__file__).resolve().parent.parent
    bn_path = base_dir / "nets" / "collection"
    netDir = base_dir / "nets"

    CancerNeapolitanNet = Net(str(bn_path / "CancerNeapolitan.neta"))
    ChestClinicNet = Net(str(bn_path / "ChestClinic.neta"))
    ClassifierNet = Net(str(bn_path / "Classifier.neta"))
    CoronaryRiskNet = Net(str(bn_path / "Coronary Risk.neta"))
    FireNet = Net(str(bn_path / "Fire.neta"))
    MendelGeneticsNet = Net(str(bn_path / "Mendel Genetics.neta"))
    RatsNet = Net(str(bn_path / "Rats.neta"))
    WetGrassNet = Net(str(bn_path / "Wet Grass.neta"))
    RatsNoisyOr = Net(str(bn_path / "Rats_NoisyOr.dne"))
    Derm = Net(str(bn_path / "Derm 7.9 A.dne"))
    CauseEffectNet = Net(str(netDir / "outputs" / "common_cause_effect.neta"))
    NF_V1_Net = Net(str(netDir / "NF_V1.dne"))

    net_list = [CancerNeapolitanNet, ChestClinicNet, ClassifierNet, CoronaryRiskNet,
                FireNet, MendelGeneticsNet, RatsNet, WetGrassNet, RatsNoisyOr, Derm, CauseEffectNet, NF_V1_Net]
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