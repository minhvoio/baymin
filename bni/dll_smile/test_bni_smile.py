from __future__ import print_function
from bni_smile import *
import time

netDir = '../nets/'

os.makedirs(netDir+"outputs", exist_ok=True)

def mainTests():
	print("[Open NF_V1.dne]")
	myNet = Net(netDir+"NF_V1.xdsl")
	print(myNet.xdsl)
	print()

	print("[Set to None (garbage collect)]")
	# This will trigger the deletion
	myNet = None
	print()

	print("[Re-open NF_V1.dne]")
	myNet = Net(netDir+"NF_V1.dne")
	print()

	print("Net name:", myNet.name())
	print("[Change name to NFAV1]")
	myNet.name("NFAV1")
	print("Check new net name:", myNet.name())
	print()

	print("Net title:", myNet.title())
	print("[Change title to 'Native Fish A V1']")
	myNet.title("Native Fish A V1")
	print("Check new net title:", myNet.title())
	print()

	print("[Get RiverFlow node]")
	rf = myNet.node("RiverFlow")
	print(rf)
	
	print(f"[Print RiverFlow's node type (Node.NATURE_NODE=1)]")
	print(rf.type())

	print(f"[Print RiverFlow's node features (=Chance,Discrete)]")
	print(rf.features())

	print("RiverFlow number states:", rf.numberStates())
	print("RiverFlow node beliefs:", rf.beliefs())
	print("FishAbundance node beliefs:", myNet.node("FishAbundance").beliefs())
	# XXX Note that this will crash if there are nodes with one state (which can
	# happen when reading in a Netica .dne!)
	print("P(All Evidence):", myNet.findingsProbability())
	print()

	print("[Set RiverFlow = state0]")
	rf.finding(0)
	print("RiverFlow node beliefs|RiverFlow=state0:", rf.beliefs())
	print("FishAbundance node beliefs|RiverFlow=state0:", myNet.node("FishAbundance").beliefs())
	print("New P(Evidence):", myNet.findingsProbability())
	print()
	
	print("[Use likelihood sampling; then set RiverFlow = state0]")
	prevAlgorithm = myNet.updateAlgorithm()
	print("Previous algorithm:", prevAlgorithm)
	myNet.updateAlgorithm(Net.ALG_BN_LSAMPLING)
	print("Algorithm set to:", myNet.updateAlgorithm())
	rf.finding(0)
	print("RiverFlow node beliefs|RiverFlow=state0:", rf.beliefs())
	print("FishAbundance node beliefs|RiverFlow=state0:", myNet.node("FishAbundance").beliefs())
	print("New P(Evidence):", myNet.findingsProbability())
	print()
	myNet.updateAlgorithm(prevAlgorithm)

	print("[Clear findings]")
	myNet.node("RiverFlow").retractFindings()
	print("New P(Evidence):", myNet.findingsProbability())
	print()

	print("[Set RiverFlow = state0]")
	rf.finding(0)
	print("RiverFlow node beliefs|RiverFlow=state0:", rf.beliefs())
	print("[Clear all findings]")
	myNet.node("RiverFlow").retractFindings()
	print("RiverFlow node beliefs:", rf.beliefs())
	print()

	print("RiverFlow Virtual Evidence (Likelihoods):", rf.likelihoods())
	print("P(RiverFlow):", rf.beliefs())
	print()

	print("[Set RiverFlow Likelihoods = 0.3,0.2]")
	rf.likelihoods([0.3,0.2])
	print("New P(RiverFlow):", rf.beliefs())
	print("New Virtual Evidence:", rf.likelihoods())
	print()

	print("[Set RiverFlow Likelihoods = 0.4,0.2]")
	rf.likelihoods([0.4,0.2])
	print("New P(RiverFlow):", rf.beliefs())
	print("New Virtual Evidence:", rf.likelihoods())
	print()

	print("[Create node called TestA]")
	node = Node(myNet, "TestA")
	print()

	print("TestA states:", node.stateNames())
	print()
	
	print("[Add state called 'three']")
	node.addState('three')
	print("TestA states:", node.stateNames())
	print()

	print("[Rename state0 to 'one']")
	node.renameState(0, 'one')
	print("TestA states:", node.stateNames())
	print()

	print("[Rename all states to one,two,three]")
	node.renameStates(['one','two','three'])
	print("TestA states:", node.stateNames())
	print()

	print("[Reorder the states to three,one,two]")
	node.reorderStates(['three',0,1])
	print("New TestA states order:", node.stateNames())
	print()

	print("[Create node TestB with one state called 'a'")
	print(" (fails in GeNIe because doesn't allow it")
	print(" --- i.e. creates node with 2 states, wrongly named)]")
	node = Node(myNet, "TestB", ['a'])
	print("TestB states:", node.stateNames())
	print()

	print("[Create node TestC with 3 states, called a,b,c]")
	node = Node(myNet, "TestC", ['a','b','c'])
	print("TestC states:", node.stateNames())
	print()

	print("TestC CPT:", node.cpt1d())
	print()

	print("RiverFlow CPT:", rf.cpt1d())
	print()

	print("[Set TestC CPT with 0.3,0.3,0.3. Should give 1/3,1/3,1/3]")
	node.cpt1d([0.3,0.3,0.3])
	print("New TestC CPT:", node.cpt1d())
	print()

	print("[Create node TestD which is discrete, deterministic (truth table)]")
	nodeD = Node(myNet, "TestD", ['a','b'], Node.TRUTH_TABLE)
	print("Test D states & CPT:", nodeD.stateNames(), nodeD.cpt())
	print("Test D features:", nodeD.features())
	print()

	print("[Add RiverFlow as parent to TestC]")
	node.addParents(["RiverFlow"])
	print("[Add FishAbundance as child of TestC]")
	node.addChildren(["FishAbundance"])
	print()

	print("1D CPT:", node.cpt1d())
	node.cpt1d([0.1,0.3,0.7,0.2,0.1,0.1])
	print("[Set TestC CPT to [0.1,0.3,0.7,0.2,0.1,0.1] using 1D array]")
	print("New 1D CPT:", node.cpt1d())
	print()

	print("2D CPT:", node.cpt())
	node.cpt([[0.1,0.3,0.2],[0.2,0.4,0.4]])
	print("[Set TestC CPT to [[0.1,0.3,0.2],[0.2,0.4,0.4]] using 2D array]")
	print("New 2D CPT:", node.cpt())
	print("New 1D CPT:", node.cpt1d())
	print()

	print("[Run through all nodes, and print names and titles]")
	for node in myNet.nodes():
		print(node.name(), node.title())
	print()

	print("[Run through all parents of FishAbundance, print names and titles]")
	for node in myNet.node("FishAbundance").parents():
		print(node.name(), node.title())
	print()
	
	print("[Print FishAbundance comment]")
	print(myNet.node("FishAbundance").comment())
	print("[Change and print comment]")
	myNet.node("FishAbundance").comment("This is a new comment")
	print(myNet.node("FishAbundance").comment())
	print()

	print("RiverFlow's visual position (x, y):", rf.position())
	print("[Set RiverFlow's visual position to 120,400]")
	print("RiverFlow's visual position:", rf.position(120,400).position())
	print()

	print("RiverFlow's visual size (width, height):", rf.size())
	print("[Set RiverFlow's visual size to 100,100]")
	print("RiverFlow's visual size:", rf.size(100,100).size())
	print()

	fap = myNet.node("FishAbundance").parents()
	print("Combinations of parent states for FishAbundance: ", myNet.numberCombinations(fap))

	print("Parent state combinations for FishAbundance:")
	parentIndexes = [0]*len(fap)
	while 1:
		print([fap[i].state(pi).name() for i,pi in enumerate(parentIndexes)])
		if not myNet.nextCombination(parentIndexes, fap): break

	print("[Write net to file called <netDir>/outputs/NF_V1_test.dne]")
	myNet.write(netDir+"outputs/NF_V1_test.dne")
	myNet.write(netDir+"outputs/NF_V1_test.xdsl")
	print()
	
	print("[Learning parameters - outputs/NF_V1_learn_params.xdsl]")
	myNet.learn(netDir+"NF_V1.csv")
	myNet.write(netDir+"outputs/NF_V1_learn_params.xdsl")
	
	print('[Learning BS - outputs/NF_V1_learn_TAN.xdsl]')
	Net().learnStructure(netDir+'NF_V1.csv', type='TAN', classVar='FishAbundance').write(netDir+'/outputs/NF_V1_learn_TAN.xdsl')

	print('[Learning BS - outputs/NF_V1_learn_BS.xdsl]')
	Net().learnStructure(netDir+'NF_V1.csv', type='BS').write(netDir+'outputs/NF_V1_learn_BS.xdsl')

	print('[Learning PC - outputs/NF_V1_learn_PC.xdsl]')
	myNet = Net().learnStructure(netDir+'NF_V1.csv', type='PC', tiers=[['PesticideUse'],['PesticideInRiver']]).write(netDir+'output_NF_V1_learn_PC.xdsl')
	print('PC Matrix:')
	print('\n'.join(str(a) for a in myNet._pcMatrix))
	for arc in myNet._pcArcList:
		print(f'{arc[0].name()} -> {arc[1].name()} ({arc[2]})')
	myNet = Net().learnStructure(netDir+'NF_V1.csv', type='PC', tiers=[['PesticideInRiver'],['PesticideUse']]).write(netDir+'output_NF_V1_learn_PC.xdsl')
	print('PC Matrix (reversed priors):')
	print('\n'.join(str(a) for a in myNet._pcMatrix))
	for arc in myNet._pcArcList:
		print(f'{arc[0].name()} -> {arc[1].name()} ({arc[2]})')

def userProps():
	myNet = Net(netDir+"NF_V1.xdsl")
	print("[Set user properties on RiverFlow (using 'add')]")
	rf = myNet.node('RiverFlow')
	rf.user().add('Label', 'River flow label')
	rf.user().add('Description', 'This is a user property field')
	print()

	print("[Get all user properties of RiverFlow]")
	print(rf.user().getAll())

def specialTests():
	print("[Testing Netica5 .dne]")
	net = Net(netDir+"netica5parens.dne")
	
	net.write(netDir+"netica5parens-from-smile.dne")

def equationTests():
	print("[Test equations]")
	net = Net(netDir+'subscriptions.xdsl')
	print('[Get equation for node "Subscriptions"]')
	subsNode = net.node('Subscriptions')
	print(subsNode.equation())

	print('[Get the first 10 samples of the "Subscriptions" node]')
	print('First 10 Samples:',net.node('Subscriptions').samples()[:10])
	
	print('[Get the discretisation levels that bound the intervals]')
	print(subsNode.levels())
	print('[And get the names of the intervals defined by the levels]')
	print(subsNode.stateNames())
	print('[Change the names of the intervals and print]')
	subsNode.renameStates(['L','M','H'])
	print(subsNode.stateNames())

	print('[Set the levels to something different and print (remember, the equation doesn\'t change)]')
	subsNode.levels([3,6,9,12,15,18,21,24,27])
	print(subsNode.stateNames())
	print(subsNode.levels())
	print('[Shorter]')
	subsNode.levels([0,5,10])
	print(subsNode.stateNames())
	print(subsNode.levels())
	print('[Reset to 0,5,10,20]')
	subsNode.levels([0,5,10,20])
	subsNode.renameStates(['Low','Med','High'])
	
	print("[Create equation node]")
	emailsNode = net.addNode("NumberEmailsPerDay", nodeType=Node.EQUATION_NODE)
	emailsNode.setEquation("NumberEmailsPerDay=1")
	emailsNode.levels([0,5,10,15,20,25,30,35])
	print(emailsNode)
	print()
	
	emailsNode.setEquation("NumberEmailsPerDay=Uniform(0,35)")
	print("Uniform mean, SD:", emailsNode.mean(), emailsNode.sd())
	
	emailsNode.setEquation("NumberEmailsPerDay=Poisson(3)")
	print("[Poisson lambda=3]")
	print("Poisson mean, SD:", emailsNode.mean(), emailsNode.sd())
	
	emailsNode.setEquation("NumberEmailsPerDay=Poisson(Subscriptions/2)")
	print("[Poisson lambda=Subscriptions/2]")
	print("Poisson mean, SD:", emailsNode.mean(), emailsNode.sd())

	print("[Enter evidence (12 subscriptions)]")
	print("Prior subscriptions mean, SD, beliefs:", subsNode.mean(), subsNode.sd(), subsNode.beliefs())
	emailsNode.finding(value = 12)
	print('Finding:', emailsNode.finding())
	print("Poisson mean, SD:", emailsNode.mean(), emailsNode.sd())
	print("Posterior subscriptions mean, SD, beliefs:", subsNode.mean(), subsNode.sd(), subsNode.beliefs())

	print()
	print("[Enter larger finding (30 subscriptions)]")
	print("Note, this causes rounding issues, and invalidates probability (and hence mean/sd) calculations")
	emailsNode.finding(value = 30)
	print("Posterior subscriptions mean, SD, beliefs:", subsNode.mean(), subsNode.sd(), subsNode.beliefs())
	print("(Should be >20 subs, and prob of high should be 100%, not 11%)")

	
	net.write(netDir+'outputs/subscriptions_modified.xdsl')
	
	#print node.beliefs()
	#net.update()
	#print(node.mean())
	
	#print()

def utilityTests():
	print("[Decision/Utility tests]")
	net = Net()
	
	print("[Create nature node PesticideInRiver]")
	pestNode = net.addNode("PesticideInRiver", states = ["True", "False"])
	pestNode.cpt1d([0.3,0.7])
	
	print("[Create utility node as child of PesticideInRiver]")
	utilNode = net.addNode("Util", Node.UTILITY_NODE)
	utilNode.addParents([pestNode])
	
	print("[Set utility node table]")
	utilNode.utilities([
		-8, # PesticideInRiver=True
		-1  # PesticideInRiver=False
	])
	
	print("[Get utility node table]")
	print(utilNode.utilities())
	
	print("[Get expected value of node]")
	print(utilNode.expectedUtilities())
	
	print("[Add decision node UsePesticide as parent of PesticideInRiver]")
	usePestNode = net.addNode("UsePesticide", Node.DECISION_NODE, states = ["Yes","No"])
	#print usePestNode.options()
	usePestNode.addChildren(["PesticideInRiver"])
	
	print("[Set conditional probabilities on UsePesticide]")
	pestNode.cpt1d([
		0.7,0.3, # UsePesticide = True
		0.1,0.9  # UsePesticide = False
	])
	print(pestNode.beliefs())
	
	print(usePestNode.numberStates())
	print("[Get expected values for each option]")
	print(utilNode.expectedUtilities())
	
	print("[Create 2nd utility node as child of PesticideUse]")
	puUtil = net.addNode("PU_Util", Node.UTILITY_NODE)
	puUtil.addParents(['UsePesticide'])
	puUtil.utilities([-5,0])
	
	print('[Create MAU node as child of the other util nodes]')
	mauNode = net.addNode('Total_Util', Node.MAU_NODE)
	mauNode.equation('2*Util + PU_Util')
	
	print("[Get expected values for each option]")
	print(mauNode.expectedUtilities())
	
	print("[Show the indexing parents for the MAU node]")
	print([p.name() for p in mauNode.indexingParents()])
	
	net.write(netDir+'test-utility.xdsl')
	
def submodelTests():
	print('[Submodels]')
	
	net = Net()
	net.addNode('Treatment', states = ['Yes', 'No'])
	
	print('[Create a submodel]')
	humanBody = net.addSubmodel('HumanBody')
	healthyNode = humanBody.addNode('Healthy', states = ['Yes', 'No']).position(200,200)
	humanBody.position(400,400).size(250,250)
	
	print('[Create a submodel in a submodel]')
	lungs = humanBody.addSubmodel('Lungs').position(300,200)
	lungs.addNode('Infection', states = ['Yes', 'No']).position(200,200)
	print('[And another]')
	heart = humanBody.addSubmodel('Heart').position(380, 200)
	heart.addNode('Pumping', states = ['Yes','No'])
	print('[Print Heart\'s parent submodel name (i.e. HumanBody)]')
	print(heart.parentSubmodel().name())
	print('[Print Treatment and HumanBody\'s parent submodels (None) and Lungs parent submodel (should be submodel object)]')
	print('Treatment parent submodel:', net.node('Treatment').parentSubmodel())
	print('HumanBody parent submodel:', humanBody.parentSubmodel())
	print('Lungs parent submodel:', lungs.parentSubmodel())
	print('[Add Ventilation node to Heart, move to Lungs]')
	vent = heart.addNode('Ventilation', states = ['Yes','No']).position(380, 200)
	vent.parentSubmodel(lungs)
	vent.parentSubmodel('Lungs')
	print('[Print Ventilation parent submodel]')
	print(vent.parentSubmodel().name())
	print('[Add Alveoli submodel to Heart, move to Lungs, add ventilation to it]')
	alveoli = heart.addSubmodel('Alveoli')
	alveoli.parentSubmodel(lungs).position(380, 200)
	alveoli.parentSubmodel('Lungs')
	vent.parentSubmodel(alveoli)
	
	print('[List submodels (all, then submodel only)]')
	print([s.name() for s in net.submodels()])
	print([s.name() for s in net.submodels(submodelOnly=True)])
	
	print('[List nodes in net (all, then submodel only)]')
	print([n.name() for n in net.nodes()])
	print([n.name() for n in net.nodes(submodelOnly=True)])
	
	print('[List nodes in submodel (all, then submodel only)]')
	print([n.name() for n in net.getSubmodel('HumanBody').nodes()])
	print([n.name() for n in net.getSubmodel('HumanBody').nodes(submodelOnly=True)])
	
	print('[Delete submodel Heart]')
	net.getSubmodel('Heart').delete()
	print('Remaining submodels:', [n.name() for n in net.submodels()])
	
	
	
	net.write(netDir+'submodel1.xdsl')
	
	# Delete all nodes
	
	for node in net.nodes():
		node.delete()


# i.e. QGeNIe nodes
def demorganTests():
	net = Net()
	print('[Create a Demorgan node]')
	fa = net.addNode('FishAbundance', Node.DEMORGAN)
	print('Name:', fa.name())
	print('Type:', fa.type(), '(should print "9"; see Node constants in bni_smile.py)')

	print('[Set prior on FishAbundance to 0.2, and print]')
	fa.dmPriorBelief(0.2)
	print('FishAbundance prior belief:', fa.dmPriorBelief())

	print('[Create RiverFlow and PesticideInRiver as Demorgan and add as parents]')
	rf = net.addNode('RiverFlow', Node.DEMORGAN)
	pir = net.addNode('PesticideInRiver', Node.DEMORGAN)
	fa.addParents([rf, pir])

	print('[Set arc weights [0.3, 0.7] (which sets the arc weight from RiverFlow, then PesticideInRiver)]')
	print('Old weights:', fa.dmParentWeights())
	fa.dmParentWeights([0.3, 0.7])
	print('New weights:', fa.dmParentWeights())

	print('[Set type to RiverFlow=requirement and PesticideInRiver=barrier]')
	print('Old types:', fa.dmParentTypes(), '(should print "[2, 2]", which is cause, cause)')
	fa.dmParentTypes([Node.REQUIREMENT, Node.BARRIER])
	print('New types:', fa.dmParentTypes(), '(should print "[1, 3]", which is requirement, barrier)')

	print('[Write to .qdsl, just by specifying .qdsl extension]')
	net.write(netDir+'outputs/qgenie_test.qdsl')

# mainTests()
# userProps()
# specialTests()
# equationTests()
# utilityTests()
# submodelTests()
# demorganTests()

# print('\nDone. Hit Enter to quit')
# input()

# net = Net()

# net.addNode("Smoking", states=["True", "False"])
# net.addNode("LungCancer", states=["True", "False"])
# net.addNode("Bronchitis", states=["True", "False"])
# net.addNode("XRay", states=["True", "False"])
# net.addNode("Cough", states=["True", "False"])

# net.node("Smoking").addChildren(["LungCancer", "Bronchitis"])
# net.node("LungCancer").addChildren(["XRay", "Cough"])
# net.node("Bronchitis").addChildren(["Cough"])

# print("Network structure:")
# for node in net.nodes():
#     print(f"{node.name()} -> {[child.name() for child in node.children()]}")

# # print("CPT for Cough:")
# # print(net.node("Cough").cpt())

# for node in net.nodes():
# 	node.setRandom()
# 	print("CPT for", node.name(), ":")
# 	node.cpt()
# 	print(node.cpt())


# net.write(netDir+'outputs/smoking_lung_cancer.dne')


netDir = '../nets/'
filePath1 = "outputs/smoking_lung_cancer.dne"
filePath2 = "outputs/test2.dne"
filePath3 = "outputs/smoking_lung_cancer copy.dne"

myNet = Net(netDir+filePath2)
# print("Re-opened network structure:")
# for node in myNet.nodes():
# 		print(f"{node.name()} -> {[child.name() for child in node.children()]}")