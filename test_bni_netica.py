from __future__ import print_function
from bni_netica import *
import time

netDir = "../nets/"

def mainTests():
	print("[Open NF_V1.dne]")
	myNet = Net(netDir+"NF_V1.dne")
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
	print("[Change title to 'Native Fish V1']")
	myNet.title("Native Fish A V1")
	print("Check new net title:", myNet.title())
	print()

	print("[Get RiverFlow node]")
	rf = myNet.node("RiverFlow")
	print(rf)

	print("RiverFlow number states:", rf.numberStates())
	print("RiverFlow node beliefs:", rf.beliefs())
	print("FishAbundance node beliefs:", myNet.node("FishAbundance").beliefs())
	print("P(All Evidence):", myNet.findingsProbability())
	print()

	print("[Set RiverFlow = state0]")
	rf.finding(0)
	print("RiverFlow node beliefs|RiverFlow=state0:", rf.beliefs())
	print("FishAbundance node beliefs|RiverFlow=state0:", myNet.node("FishAbundance").beliefs())
	print("New P(Evidence):", myNet.findingsProbability())
	print()

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
	
	print("[Create continuous node TestD]")
	testD = myNet.addNode("TestD", [])
	print("Node type?", Node.typeName[testD.type()])
	print()
	
	newLevels = [0.2,0.6,1.8,3.4,9.7]
	print("[Set levels to {}]".format(newLevels))
	testD.levels(newLevels)
	print("Get TestD levels:", testD.levels())
	print()

	newLevels = [3,7,10]
	print("[Create continuous node TestE with levels {}]".format(newLevels))
	testE = myNet.addNode("TestE", newLevels)
	print("Node type?", Node.typeName[testE.type()])
	print("Levels", testE.levels())
	print()

	print("RiverFlow CPT:", rf.cpt1d())
	print()

	print("[Set TestC CPT with 0.3,0.3,0.3. Should give 1/3,1/3,1/3]")
	node.cpt1d([0.3,0.3,0.3])
	print("New TestC CPT:", node.cpt1d())
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

	print("RiverFlow's visual position:", rf.position())
	print("[Set RiverFlow's visual position to 120,400]")
	print("RiverFlow's visual position:", rf.position(120,400).position())
	print()

	fap = myNet.node("FishAbundance").parents()
	print("Combinations of parent states for FishAbundance: ", myNet.numberCombinations(fap))

	print("Parent state combinations for FishAbundance:")
	parentIndexes = [0]*len(fap)
	while 1:
		print([fap[i].state(pi).name() for i,pi in enumerate(parentIndexes)])
		if not myNet.nextCombination(parentIndexes, fap): break

	print("[Write net to file called output_NF_V1_test.dne]")
	myNet.write("output_NF_V1_test.dne")
	print()

	print("[Learning network]")
	myNet.experience(1000000)
	myNet.learn("../nets/NF_V1.csv", type = 'EM')
	myNet.write("../nets/output_NF_V1_test_learn.dne")
	
	#myNet.addNode('MyDecision', states = ['Choice1','Choice2'])
	# print('Utilities:', myNet.node('D').expectedUtils())

def utilsTests():
	netDir = "./nets/"
	from bni_netica.bni_netica import Net
	from bni_netica import bni_utils
	from bn_helpers.get_structures_print_tools import getNetCPTStrings
	from bn_helpers.get_structures_print_tools import printNet, get_BN_node_states
	myNet = Net(netDir+"NF_V1.dne")

	print('[d-connected nodes between PesticideUse]')
	nodes = myNet.node('PesticideUse').getRelated('d_connected, exclude_self')
	print([n.name() for n in nodes])

	print(getNetCPTStrings(myNet))

	anotherNetDir = "./nets/collection/"
	carNet = Net(anotherNetDir+"CarDiagnosis.neta")
	print('CarDiagnosis.neta:')
	printNet(carNet)
	node_states = get_BN_node_states(carNet)
	print(node_states)

	carNet.retractFindings()  # clean slate

	evidence = {
			# Electrical symptoms
			'HL': 'off',          # Headlights are off
			'BV': 'dead',         # Battery voltage is dead
			'BA': 'very_old',     # Battery is very old
			'CC': 'False',        # Car does not crank

			# Rule-out a fuse issue so the posterior points elsewhere
			'MF': 'okay',         # Main fuse okay

			# Starter suspected faulty in this scenario
			'SS': 'Faulty',       # Starter system faulty

			# Assume fuel/air are OK to isolate electrical & starter chain
			'FS': 'Okay',
			'AS': 'Okay',

			# Optional extra symptoms you might want to include:
			# 'PV': 'none',       # No voltage at plug
			# 'SQ': 'very_bad',   # Spark quality very bad
	}

	for node_name, state_name in evidence.items():
			carNet.node(node_name).finding(state_name)

	# 3) Query P(Car does NOT start)
	st_node = carNet.node('ST')                       # ST has states ['True','False']  (Car starts?)
	st_beliefs = st_node.beliefs()                 # posterior over ['True','False'] in state order
	st_states  = st_node.stateNames()              # ['True','False'] (names from the net)
	post = dict(zip(st_states, st_beliefs))        # {'True': p, 'False': q}

	print("Posterior over ST (Car starts):", post)
	print("P(Car DOESN'T start) = P(ST='False') =", post['False'])

	# If you want the joint probability of all findings too:
	print("P(all entered findings) =", carNet.findingsProbability())
	# print(getNetCPTStrings(anotherNet))

	# print('[d-connected nodes between PesticideUse and Rainfall]')

	# nodes = bni_utils.findAllDConnectedNodes(myNet, "PesticideUse", "Rainfall")
	# print([n.name() for n in nodes])

	# print('[d-connected nodes between PesticideUse and FishAbundance]')
	# nodes = bni_utils.findAllDConnectedNodes(myNet, "PesticideUse", "FishAbundance")
	# print([n.name() for n in nodes])

	# print('[d-connected nodes between PesticideUse and Rainfall | PesticideInRiver]')
	# myNet.node('PesticideInRiver').finding('Low')
	# nodes = bni_utils.findAllDConnectedNodes(myNet, "PesticideUse", "Rainfall")
	# print([n.name() for n in nodes])
	# myNet.retractFindings()

	# print('[d-connected nodes between PesticideUse and Rainfall | FishAbundance]')
	# myNet.node('FishAbundance').finding('Low')
	# arcs = bni_utils.findAllDConnectedNodes(myNet, "PesticideUse", "Rainfall", {"arcs": True})
	# print([arc for arc in arcs])

# mainTests()
utilsTests()

