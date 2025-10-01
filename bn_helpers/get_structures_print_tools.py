from bni_netica.bni_netica import Net

def get_nets():
    bn_path = "../nets/collection/"
    netDir = "../nets/"

    CancerNeapolitanNet = Net(bn_path+"CancerNeapolitan.neta")
    ChestClinicNet = Net(bn_path+"ChestClinic.neta")
    ClassifierNet = Net(bn_path+"Classifier.neta")
    CoronaryRiskNet = Net(bn_path+"Coronary Risk.neta")
    FireNet = Net(bn_path+"Fire.neta")
    MendelGeneticsNet = Net(bn_path+"Mendel Genetics.neta")
    RatsNet = Net(bn_path+"Rats.neta")
    WetGrassNet = Net(bn_path+"Wet Grass.neta")
    RatsNoisyOr = Net(bn_path+"Rats_NoisyOr.dne")
    Derm = Net(bn_path+"Derm 7.9 A.dne")
    CauseEffectNet = Net(netDir+"outputs/common_cause_effect.neta")
    NF_V1_Net = Net(netDir+"NF_V1.dne")

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