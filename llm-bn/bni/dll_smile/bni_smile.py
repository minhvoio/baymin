from __future__ import print_function
from builtins import range
from builtins import bytes
import ctypes, re, sys, os, xml.etree.ElementTree as ET, pdb, math, random

# Add this onto the ElementTree class
def replaceNode(et, xpath, newNode):
	if type(newNode) is str:
		newNode = ET.fromstring(newNode)
	parent = et.find(xpath+'/..')
	node = et.find(xpath)
	i = list(parent).index(node)
	parent.remove(node)
	parent.insert(i, newNode)
ET.replaceNode = replaceNode

def parseLicense(fn):
	lic1 = ''
	lic2 = []
	with open(fn) as licFile:
		state = 'initial'
		for ln in licFile:
			if state == 'initial':
				if re.search(r'const char\* DSL_LIC1 =', ln):
					state = 'dsl_lic1'
			elif state == 'dsl_lic1':
				m = re.search(r'"(.*)"', ln)
				if m:
					lic1 += m.group(1)
				m = re.search('unsigned char DSL_LIC2', ln)
				if m:
					state = 'dsl_lic2'
			elif state == 'dsl_lic2':
				isEnd = re.search(r'};', ln)
				ln2 = ln.strip('}; \t\n')
				hexes = re.split(r',', ln2)
				hexes = [int(h[2:], 16) for h in hexes if h]
				lic2.extend(hexes)
				if isEnd:
					break
	
	return [lic1, lic2]
	
def patchDll(dllFn, lic1, lic2):
	licenseLabel = '0xf2,0x91,0x36,0xf3,0x46,0xb7,0xab,0xaa,0x77,0x5a,0xcb,0xea,0xf0,0xf5,0x15,0xce'
	licenseLabel = [int(h[2:],16) for h in licenseLabel.split(',')]
	with open(dllFn, 'rb') as inDll:
		with open(dllFn+'.out', 'wb') as outDll:
			inBytes = inDll.read()
			lic1Head = b'SMILE LICENSE'
			lic1Tail = b'END OF LICENSE SECTION'
			lic1Index = inBytes.index(lic1Head)
			lic1IndexEnd = inBytes.index(lic1Tail)+len(lic1Tail)
			lic1SectionLength = lic1IndexEnd - lic1Index
			#print("LIC1 LENGTH:",lic1SectionLength)
			
			lic2Index = inBytes.index(bytes(licenseLabel))-64
			lic2IndexEnd = lic2Index + 80
			
			outDll.write(inBytes[:lic1Index])
			lic1Bytes = bytes(lic1, 'ascii') + b'\x00'
			outDll.write(lic1Bytes)
			outDll.write(b' '*(lic1SectionLength-len(lic1Bytes)-len(lic1Tail)-1))
			outDll.write(lic1Tail+b'\x00')
			
			outDll.write(inBytes[lic1IndexEnd:lic2Index])
			
			outDll.write(bytes(lic2))
			outDll.write(bytes(licenseLabel))
			outDll.write(inBytes[lic2IndexEnd:])

def updateLicense():
	import shutil
	licFn = 'smile_license.h'
	bismileFns = ['bismile.dll', 'bismile64.dll']
	print("Trying license patch")
	if not os.path.exists(licFn):
		print('Can\'t find {}. Download it from bayesfusion.com and put it in this directory.'.format(licFn))
		sys.exit()
	[lic1,lic2] = parseLicense(licFn)
	for bismileFn in bismileFns:
		patchDll(bismileFn, lic1, lic2)
	# ctypes.windll.LoadLibrary('bismile64.dll.out')
	# print('OK!')
	
	shutil.copyfile('bismile.dll', 'bismile.dll.backup')
	shutil.copyfile('bismile64.dll', 'bismile64.dll.backup')
	shutil.copyfile('bismile.dll.out', 'bismile.dll')
	shutil.copyfile('bismile64.dll.out', 'bismile64.dll')
	os.unlink('bismile.dll.out')
	os.unlink('bismile64.dll.out')
	try:
		if sys.maxsize > 2**32:
			ctypes.windll.LoadLibrary('bismile64.dll')
		else:
			ctypes.windll.LoadLibrary('bismile.dll')
	except:
		print('Oh, the patch didn\'t seem to work. Reversing. :(')
		shutil.copyfile('bismile.dll.backup', 'bismile.dll')
		shutil.copyfile('bismile64.dll.backup', 'bismile64.dll')
		sys.exit()
	print("License update looks OK")
	#print("Seems like it worked. Check and delete backup files. Fingers crossed. :)")

d = os.path.abspath('.')
sys.path.append(d); os.environ['PATH'] += ';'+d
if hasattr(os, 'add_dll_directory'):
	os.add_dll_directory(os.path.dirname(__file__) or d)

if __name__ == '__main__':
	if len(sys.argv)==2 and sys.argv[1] == '--license':
		updateLicense()

# Try to auto-update license if smile_license.h is present
# (Disabling for now, as too problematic)
# if os.path.exists('smile_license.h'):
# 	updateLicense()
# 	os.unlink("smile_license.h")

if sys.platform == 'win32':
	if sys.maxsize > 2**32:
		g = ctypes.windll.LoadLibrary("bismile64.dll")
	else:
		g = ctypes.windll.LoadLibrary("bismile.dll")
elif sys.platform == 'linux':
	g = ctypes.cdll.LoadLibrary("./libbismile.so")
	
def isString(s):
	try:
		return isinstance(s, basestring)
	except:
		return isinstance(s, str)

def normalize(vec):
	total = 0
	newVec = vec[:]
	for i,v in enumerate(newVec):
		if v < 0:
			newVec[i] = 0
		else:
			total += v
	
	return [v/total for v in newVec]



# This is so that functions return pointer-like objects,
# rather than ints, which won't work on all platforms
# See: https://code.activestate.com/lists/python-list/702828/
class VOID(ctypes.Structure):
	pass
	
class c_simplechar_p(ctypes.c_char_p):
	@classmethod
	def from_param(cls, obj):
		return bytes(obj, 'ascii', errors='ignore')
	
	@classmethod
	def _check_retval_(cls, result):
		return result.value.decode('utf-8', 'replace') if result.value is not None else result.value
	
# Define the KeyValue struct
class KeyValue(ctypes.Structure):
    _fields_ = [("key", ctypes.c_char_p), ("value", ctypes.c_double)]

# Define the KeyValueArray struct
class KeyValueArray(ctypes.Structure):
    _fields_ = [("items", ctypes.POINTER(KeyValue)), ("size", ctypes.c_size_t)]

# [[BINDINGS START]] - DON'T MODIFY
g.test.restype = c_simplechar_p
g.test.argtypes = [c_simplechar_p]
#g.setLicense.restype = VOID
g.setLicense.argtypes = [c_simplechar_p, ctypes.c_ubyte]
g.new_intArray.restype = ctypes.POINTER(VOID)
g.new_intArray.argtypes = []
g.intArray_NumItems.restype = ctypes.c_int
g.intArray_NumItems.argtypes = [ctypes.POINTER(VOID)]
g.intArray_Items.restype = ctypes.POINTER(ctypes.c_int)
g.intArray_Items.argtypes = [ctypes.POINTER(VOID)]
g.doubleArray_NumItems.restype = ctypes.c_int
g.doubleArray_NumItems.argtypes = [ctypes.POINTER(VOID)]
g.doubleArray_Items.restype = ctypes.POINTER(ctypes.c_double)
g.doubleArray_Items.argtypes = [ctypes.POINTER(VOID)]
g.stringArray_Items.restype = ctypes.POINTER(c_simplechar_p)
g.stringArray_Items.argtypes = [ctypes.POINTER(VOID)]
g.new_network.restype = ctypes.POINTER(VOID)
g.new_network.argtypes = []
g.copy_network.restype = ctypes.POINTER(VOID)
g.copy_network.argtypes = [ctypes.POINTER(VOID)]
#g.delete_network.restype = VOID
g.delete_network.argtypes = [ctypes.POINTER(VOID)]
g.AddNode.restype = ctypes.c_int
g.AddNode.argtypes = [ctypes.POINTER(VOID), ctypes.c_int, c_simplechar_p]
g.DeleteNode.restype = ctypes.c_int
g.DeleteNode.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
g.AddArc.restype = ctypes.c_int
g.AddArc.argtypes = [ctypes.POINTER(VOID), ctypes.c_int, ctypes.c_int]
g.RemoveArc.restype = ctypes.c_int
g.RemoveArc.argtypes = [ctypes.POINTER(VOID), ctypes.c_int, ctypes.c_int]
g.FindNode.restype = ctypes.c_int
g.FindNode.argtypes = [ctypes.POINTER(VOID), c_simplechar_p]
g.HasNode.restype = ctypes.c_int
g.HasNode.argtypes = [ctypes.POINTER(VOID), c_simplechar_p]
g.GetNode.restype = ctypes.POINTER(VOID)
g.GetNode.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
g.GetAllNodes.restype = ctypes.c_int
g.GetAllNodes.argtypes = [ctypes.POINTER(VOID), ctypes.POINTER(VOID)]
g.GetParents.restype = ctypes.POINTER(VOID)
g.GetParents.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
#g.UpdateBeliefs.restype = VOID
g.UpdateBeliefs.argtypes = [ctypes.POINTER(VOID)]
g.ClearAllEvidence.restype = ctypes.c_int
g.ClearAllEvidence.argtypes = [ctypes.POINTER(VOID)]
g.CalcProbEvidence.restype = ctypes.c_double
g.CalcProbEvidence.argtypes = [ctypes.POINTER(VOID)]
#g.WriteToFile.restype = VOID
g.WriteToFile.argtypes = [ctypes.POINTER(VOID), c_simplechar_p]
#g.ReadFromFile.restype = VOID
g.ReadFromFile.argtypes = [ctypes.POINTER(VOID), c_simplechar_p]
#g.SetDefaultBNAlgorithm.restype = VOID
g.SetDefaultBNAlgorithm.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
g.GetDefaultBNAlgorithm.restype = ctypes.c_int
g.GetDefaultBNAlgorithm.argtypes = [ctypes.POINTER(VOID)]
g.GetNumberOfSamples.restype = ctypes.c_int
g.GetNumberOfSamples.argtypes = [ctypes.POINTER(VOID)]
g.SetNumberOfSamples.restype = ctypes.c_int
g.SetNumberOfSamples.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
g.GetNumberOfDiscretizationSamples.restype = ctypes.c_int
g.GetNumberOfDiscretizationSamples.argtypes = [ctypes.POINTER(VOID)]
#g.SetNumberOfDiscretizationSamples.restype = VOID
g.SetNumberOfDiscretizationSamples.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
g.net_Header.restype = ctypes.POINTER(VOID)
g.net_Header.argtypes = [ctypes.POINTER(VOID)]
g.node_Info.restype = ctypes.POINTER(VOID)
g.node_Info.argtypes = [ctypes.POINTER(VOID)]
g.node_Value.restype = ctypes.POINTER(VOID)
g.node_Value.argtypes = [ctypes.POINTER(VOID)]
g.node_ChangeType.restype = ctypes.c_int
g.node_ChangeType.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
g.nodeInfo_Header.restype = ctypes.POINTER(VOID)
g.nodeInfo_Header.argtypes = [ctypes.POINTER(VOID)]
g.nodeInfo_Screen.restype = ctypes.POINTER(VOID)
g.nodeInfo_Screen.argtypes = [ctypes.POINTER(VOID)]
g.nodeInfo_UserProperties.restype = ctypes.POINTER(VOID)
g.nodeInfo_UserProperties.argtypes = [ctypes.POINTER(VOID)]
g.nodeValue_GetMatrix.restype = ctypes.POINTER(VOID)
g.nodeValue_GetMatrix.argtypes = [ctypes.POINTER(VOID)]
g.nodeValue_GetType.restype = ctypes.c_int
g.nodeValue_GetType.argtypes = [ctypes.POINTER(VOID)]
g.nodeValue_IsValueValid.restype = ctypes.c_int
g.nodeValue_IsValueValid.argtypes = [ctypes.POINTER(VOID)]
g.nodeValue_IsRealEvidence.restype = ctypes.c_int
g.nodeValue_IsRealEvidence.argtypes = [ctypes.POINTER(VOID)]
g.nodeValue_GetEvidence.restype = ctypes.c_int
g.nodeValue_GetEvidence.argtypes = [ctypes.POINTER(VOID)]
g.nodeValue_SetEvidence.restype = ctypes.c_int
g.nodeValue_SetEvidence.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
g.nodeValue_GetVirtualEvidence.restype = ctypes.POINTER(ctypes.c_double)
g.nodeValue_GetVirtualEvidence.argtypes = [ctypes.POINTER(VOID)]
g.nodeValue_SetVirtualEvidence.restype = ctypes.c_int
g.nodeValue_SetVirtualEvidence.argtypes = [ctypes.POINTER(VOID), ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
g.nodeValue_ClearEvidence.restype = ctypes.c_int
g.nodeValue_ClearEvidence.argtypes = [ctypes.POINTER(VOID)]
g.nodeValue_GetIndexingParents.restype = ctypes.POINTER(VOID)
g.nodeValue_GetIndexingParents.argtypes = [ctypes.POINTER(VOID)]
g.node_Definition.restype = ctypes.POINTER(VOID)
g.node_Definition.argtypes = [ctypes.POINTER(VOID)]
g.nodeDefinition_GetType.restype = ctypes.c_int
g.nodeDefinition_GetType.argtypes = [ctypes.POINTER(VOID)]
g.nodeDefinition_GetMatrix.restype = ctypes.POINTER(VOID)
g.nodeDefinition_GetMatrix.argtypes = [ctypes.POINTER(VOID)]
g.nodeDefinition_GetDoubleDefinition.restype = ctypes.POINTER(VOID)
g.nodeDefinition_GetDoubleDefinition.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
#g.nodeDefinition_SetDoubleDefinition.restype = VOID
g.nodeDefinition_SetDoubleDefinition.argtypes = [ctypes.POINTER(VOID), ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
g.nodeDefinition_GetNumberOfOutcomes.restype = ctypes.c_int
g.nodeDefinition_GetNumberOfOutcomes.argtypes = [ctypes.POINTER(VOID)]
g.nodeDefinition_GetOutcomesNames.restype = ctypes.POINTER(VOID)
g.nodeDefinition_GetOutcomesNames.argtypes = [ctypes.POINTER(VOID)]
g.nodeDefinition_AddOutcome.restype = ctypes.c_int
g.nodeDefinition_AddOutcome.argtypes = [ctypes.POINTER(VOID), c_simplechar_p]
g.nodeDefinition_SetNumberOfOutcomes.restype = ctypes.c_int
g.nodeDefinition_SetNumberOfOutcomes.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
g.nodeDefinition_SetNumberOfOutcomesStr.restype = ctypes.c_int
g.nodeDefinition_SetNumberOfOutcomesStr.argtypes = [ctypes.POINTER(VOID), ctypes.c_int, ctypes.POINTER(c_simplechar_p)]
g.nodeDefinition_RenameOutcomes.restype = ctypes.c_int
g.nodeDefinition_RenameOutcomes.argtypes = [ctypes.POINTER(VOID), ctypes.c_int, ctypes.POINTER(c_simplechar_p)]
g.nodeDefinition_RemoveOutcome.restype = ctypes.c_int
g.nodeDefinition_RemoveOutcome.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
g.nodeDefinition_ChangeOrderOfOutcomes.restype = ctypes.c_int
g.nodeDefinition_ChangeOrderOfOutcomes.argtypes = [ctypes.POINTER(VOID), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
g.nodeDefinition_ChangeOrderOfOutcomesWithAddAndRemove.restype = ctypes.c_int
g.nodeDefinition_ChangeOrderOfOutcomesWithAddAndRemove.argtypes = [ctypes.POINTER(VOID), ctypes.c_int, ctypes.POINTER(c_simplechar_p), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
g.equation_SetEquation.restype = ctypes.c_int
g.equation_SetEquation.argtypes = [ctypes.POINTER(VOID), c_simplechar_p]
g.equation_GetDiscreteIntervals.restype = KeyValueArray
g.equation_GetDiscreteIntervals.argtypes = [ctypes.POINTER(VOID)]
g.equation_SetDiscreteIntervals.restype = ctypes.c_int
g.equation_SetDiscreteIntervals.argtypes = [ctypes.POINTER(VOID), KeyValueArray]
g.equation_ClearDiscreteIntervals.restype = ctypes.c_int
g.equation_ClearDiscreteIntervals.argtypes = [ctypes.POINTER(VOID)]
g.equation_GetBounds.restype = ctypes.POINTER(ctypes.c_double)
g.equation_GetBounds.argtypes = [ctypes.POINTER(VOID)]
g.equation_SetBounds.restype = ctypes.c_int
g.equation_SetBounds.argtypes = [ctypes.POINTER(VOID), ctypes.c_double, ctypes.c_double]
g.mau_SetExpression.restype = ctypes.c_int
g.mau_SetExpression.argtypes = [ctypes.POINTER(VOID), c_simplechar_p]
g.valEqEvaluation_GetMean.restype = ctypes.c_double
g.valEqEvaluation_GetMean.argtypes = [ctypes.POINTER(VOID)]
g.valEqEvaluation_GetStdDev.restype = ctypes.c_double
g.valEqEvaluation_GetStdDev.argtypes = [ctypes.POINTER(VOID)]
g.valEqEvaluation_GetSampleMean.restype = ctypes.c_double
g.valEqEvaluation_GetSampleMean.argtypes = [ctypes.POINTER(VOID)]
g.valEqEvaluation_GetSampleStdDev.restype = ctypes.c_double
g.valEqEvaluation_GetSampleStdDev.argtypes = [ctypes.POINTER(VOID)]
g.valEqEvaluation_GetSample.restype = ctypes.c_double
g.valEqEvaluation_GetSample.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
g.valEqEvaluation_GetNumberOfSamples.restype = ctypes.c_int
g.valEqEvaluation_GetNumberOfSamples.argtypes = [ctypes.POINTER(VOID)]
g.valEqEvaluation_GetSamples.restype = ctypes.POINTER(ctypes.c_double)
g.valEqEvaluation_GetSamples.argtypes = [ctypes.POINTER(VOID)]
#g.valEqEvaluation_SetEvidence.restype = VOID
g.valEqEvaluation_SetEvidence.argtypes = [ctypes.POINTER(VOID), ctypes.c_double]
g.valEqEvaluation_GetEvidence.restype = ctypes.c_double
g.valEqEvaluation_GetEvidence.argtypes = [ctypes.POINTER(VOID)]
g.dMatrix_GetSize.restype = ctypes.c_int
g.dMatrix_GetSize.argtypes = [ctypes.POINTER(VOID)]
g.dMatrix_GetItemsDouble.restype = ctypes.POINTER(ctypes.c_double)
g.dMatrix_GetItemsDouble.argtypes = [ctypes.POINTER(VOID)]
g.header_GetId.restype = c_simplechar_p
g.header_GetId.argtypes = [ctypes.POINTER(VOID)]
#g.header_SetId.restype = VOID
g.header_SetId.argtypes = [ctypes.POINTER(VOID), c_simplechar_p]
g.header_GetName.restype = c_simplechar_p
g.header_GetName.argtypes = [ctypes.POINTER(VOID)]
#g.header_SetName.restype = VOID
g.header_SetName.argtypes = [ctypes.POINTER(VOID), c_simplechar_p]
g.header_GetComment.restype = c_simplechar_p
g.header_GetComment.argtypes = [ctypes.POINTER(VOID)]
#g.header_SetComment.restype = VOID
g.header_SetComment.argtypes = [ctypes.POINTER(VOID), c_simplechar_p]
g.screenInfo_position.restype = ctypes.POINTER(VOID)
g.screenInfo_position.argtypes = [ctypes.POINTER(VOID)]
g.screenInfo_formatting.restype = ctypes.POINTER(ctypes.c_int)
g.screenInfo_formatting.argtypes = [ctypes.POINTER(VOID)]
#g.screenInfo_setFormatting.restype = VOID
g.screenInfo_setFormatting.argtypes = [ctypes.POINTER(VOID), ctypes.POINTER(ctypes.c_int)]
g.rectangle_center_X.restype = ctypes.c_int
g.rectangle_center_X.argtypes = [ctypes.POINTER(VOID)]
g.rectangle_center_Y.restype = ctypes.c_int
g.rectangle_center_Y.argtypes = [ctypes.POINTER(VOID)]
#g.rectangle_center_X_set.restype = VOID
g.rectangle_center_X_set.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
#g.rectangle_center_Y_set.restype = VOID
g.rectangle_center_Y_set.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
g.rectangle_width.restype = ctypes.c_int
g.rectangle_width.argtypes = [ctypes.POINTER(VOID)]
g.rectangle_height.restype = ctypes.c_int
g.rectangle_height.argtypes = [ctypes.POINTER(VOID)]
#g.rectangle_width_set.restype = VOID
g.rectangle_width_set.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
#g.rectangle_height_set.restype = VOID
g.rectangle_height_set.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
g.userProperties_AddProperty.restype = ctypes.c_int
g.userProperties_AddProperty.argtypes = [ctypes.POINTER(VOID), c_simplechar_p, c_simplechar_p]
g.userProperties_FindProperty.restype = ctypes.c_int
g.userProperties_FindProperty.argtypes = [ctypes.POINTER(VOID), c_simplechar_p]
g.userProperties_DeleteProperty.restype = ctypes.c_int
g.userProperties_DeleteProperty.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
g.userProperties_GetNumberOfProperties.restype = ctypes.c_int
g.userProperties_GetNumberOfProperties.argtypes = [ctypes.POINTER(VOID)]
g.userProperties_GetPropertyName.restype = c_simplechar_p
g.userProperties_GetPropertyName.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
g.userProperties_GetPropertyValue.restype = c_simplechar_p
g.userProperties_GetPropertyValue.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
#g.userProperties_Clear.restype = VOID
g.userProperties_Clear.argtypes = [ctypes.POINTER(VOID)]
g.submodel_CreateSubmodel.restype = ctypes.c_int
g.submodel_CreateSubmodel.argtypes = [ctypes.POINTER(VOID), ctypes.c_int, c_simplechar_p]
g.submodel_DeleteSubmodel.restype = ctypes.c_int
g.submodel_DeleteSubmodel.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
g.submodel_FindSubmodel.restype = ctypes.c_int
g.submodel_FindSubmodel.argtypes = [ctypes.POINTER(VOID), c_simplechar_p]
g.submodel_GetParent.restype = ctypes.c_int
g.submodel_GetParent.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
g.submodel_MoveSubmodel.restype = ctypes.c_int
g.submodel_MoveSubmodel.argtypes = [ctypes.POINTER(VOID), ctypes.c_int, ctypes.c_int]
g.submodel_GetSubmodel.restype = ctypes.POINTER(VOID)
g.submodel_GetSubmodel.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
g.submodel_Header.restype = ctypes.POINTER(VOID)
g.submodel_Header.argtypes = [ctypes.POINTER(VOID)]
g.submodel_ScreenInfo.restype = ctypes.POINTER(VOID)
g.submodel_ScreenInfo.argtypes = [ctypes.POINTER(VOID)]
g.node_GetSubmodel.restype = ctypes.c_int
g.node_GetSubmodel.argtypes = [ctypes.POINTER(VOID)]
g.node_SetSubmodel.restype = ctypes.c_int
g.node_SetSubmodel.argtypes = [ctypes.POINTER(VOID), c_simplechar_p]
g.submodel_GetIncludedSubmodels.restype = ctypes.POINTER(VOID)
g.submodel_GetIncludedSubmodels.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
g.submodel_GetIncludedNodes.restype = ctypes.POINTER(VOID)
g.submodel_GetIncludedNodes.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
g.submodel_GetDeepIncludedSubmodels.restype = ctypes.POINTER(VOID)
g.submodel_GetDeepIncludedSubmodels.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
g.submodel_GetDeepIncludedNodes.restype = ctypes.POINTER(VOID)
g.submodel_GetDeepIncludedNodes.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
g.learn_em.restype = ctypes.c_int
g.learn_em.argtypes = [ctypes.POINTER(VOID), c_simplechar_p, ctypes.c_float, ctypes.c_bool, ctypes.c_float, ctypes.c_int]
g.learn_emSimple.restype = ctypes.c_int
g.learn_emSimple.argtypes = [ctypes.POINTER(VOID), c_simplechar_p, ctypes.c_float]
g.learn_pc.restype = ctypes.POINTER(ctypes.c_int)
g.learn_pc.argtypes = [ctypes.POINTER(VOID), c_simplechar_p, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
g.learn_bs.restype = ctypes.c_int
g.learn_bs.argtypes = [ctypes.POINTER(VOID), c_simplechar_p, ctypes.c_int, ctypes.c_int]
g.learn_tan.restype = ctypes.c_int
g.learn_tan.argtypes = [ctypes.POINTER(VOID), c_simplechar_p, c_simplechar_p, ctypes.c_int]
g.Demorgan_SetParentWeights.restype = ctypes.c_int
g.Demorgan_SetParentWeights.argtypes = [ctypes.POINTER(VOID), ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
g.Demorgan_SetParentWeight.restype = ctypes.c_int
g.Demorgan_SetParentWeight.argtypes = [ctypes.POINTER(VOID), ctypes.c_int, ctypes.c_double]
g.Demorgan_GetParentWeight.restype = ctypes.c_double
g.Demorgan_GetParentWeight.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
g.Demorgan_GetParentWeights.restype = ctypes.POINTER(VOID)
g.Demorgan_GetParentWeights.argtypes = [ctypes.POINTER(VOID)]
g.Demorgan_SetParentTypes.restype = ctypes.c_int
g.Demorgan_SetParentTypes.argtypes = [ctypes.POINTER(VOID), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
g.Demorgan_SetParentType.restype = ctypes.c_int
g.Demorgan_SetParentType.argtypes = [ctypes.POINTER(VOID), ctypes.c_int, ctypes.c_int]
g.Demorgan_GetParentType.restype = ctypes.c_int
g.Demorgan_GetParentType.argtypes = [ctypes.POINTER(VOID), ctypes.c_int]
g.Demorgan_GetParentTypes.restype = ctypes.POINTER(VOID)
g.Demorgan_GetParentTypes.argtypes = [ctypes.POINTER(VOID)]
g.Demorgan_GetTemporalParentType.restype = ctypes.c_int
g.Demorgan_GetTemporalParentType.argtypes = [ctypes.POINTER(VOID), ctypes.c_int, ctypes.c_int]
g.Demorgan_SetTemporalParentType.restype = ctypes.c_int
g.Demorgan_SetTemporalParentType.argtypes = [ctypes.POINTER(VOID), ctypes.c_int, ctypes.c_int, ctypes.c_int]
g.Demorgan_GetTemporalParentWeight.restype = ctypes.c_double
g.Demorgan_GetTemporalParentWeight.argtypes = [ctypes.POINTER(VOID), ctypes.c_int, ctypes.c_int]
g.Demorgan_SetTemporalParentWeight.restype = ctypes.c_int
g.Demorgan_SetTemporalParentWeight.argtypes = [ctypes.POINTER(VOID), ctypes.c_int, ctypes.c_int, ctypes.c_double]
g.Demorgan_GetPriorBelief.restype = ctypes.c_double
g.Demorgan_GetPriorBelief.argtypes = [ctypes.POINTER(VOID)]
g.Demorgan_SetPriorBelief.restype = ctypes.c_int
g.Demorgan_SetPriorBelief.argtypes = [ctypes.POINTER(VOID), ctypes.c_double]
# [[BINDINGS END]] - DON'T MODIFY


# Constants
class Node:
	# These roles defined in nodedef.h (with names like DSL_DECISION)
	FEATURES = {
		'Decision': 1,
		'Chance': 2,
		'Deterministic': 4,
		'Utility': 8,
		'Discrete': 16,
		'CASTLogic': 32,
		'DeMorganLogic': 64,
		'NoisyMaxLogic': 128,
		'NoisyAdderLogic': 256,
		'ParentsContinuous': 512,
	}
	# MAU nodes are specific to GeNIe. They overlap with equation and utility nodes in Netica,
	# but not exactly for either. (In fact, GeNIe defines it as a utility node with continuous parents,
	# which would normally be utilities.)
	# Also defined in nodedef.h
	NATURE_NODE, CONSTANT_NODE, DECISION_NODE, UTILITY_NODE, DISCONNECTED_NODE, \
	EQUATION_NODE, MAU_NODE, TRUTH_TABLE, DEMORGAN = list(range(1,10))

	# QGeNIe arc types:
	INHIBITOR, REQUIREMENT, CAUSE, BARRIER = list(range(0, 4))
class Node(Node):
	NODE_TYPE_MAP = {
		Node.NATURE_NODE: 18, #DSL_CPT (=DSL_CHANCE|DSL_DISCRETE)
		Node.EQUATION_NODE: 4, # DSL_EQUATION (=DSL_DETERMINISTIC)
		Node.UTILITY_NODE: 8, # DSL_UTILITY
		Node.DECISION_NODE: 17, # DSL_LIST (=DSL_DECISION|DSL_DISCRETE)
		Node.MAU_NODE: 520, # DSL_MAU (=DSL_UTILITY|DSL_PARENTSCONTIN)
		Node.TRUTH_TABLE: 20, # DSL_TRUTHTABLE (=DSL_DETERMINISTIC|DSL_DISCRETE)
		Node.DEMORGAN: 82, # DSL_DEMORGAN (=DSL_CHANCE|DSL_DISCRETE|DSL_DEMORGANLOGIC)
	}
	NODE_TYPE_MAP_REV = {v:k for k,v in NODE_TYPE_MAP.items()}
class Net:
	ALG_BN_LAURITZEN, ALG_BN_HENRION, ALG_BN_PEARL, ALG_BN_LSAMPLING, \
		ALG_BN_SELFIMPORTANCE, ALG_BN_HEURISTICIMPORTANCE, ALG_BN_BACKSAMPLING, \
		ALG_BN_AISSAMPLING, ALG_BN_EPISSAMPLING, ALG_BN_LBP, ALG_BN_LAURITZEN_OLD, \
		ALG_BN_RELEVANCEDECOMP, ALG_BN_RELEVANCEDECOMP2, ALG_HBN_HEPIS, ALG_HBN_HLW, \
		ALG_HBN_HLBP, ALG_HBN_HLOGICSAMPLING = list(range(0,17))
 
def NYI():
	print("nyi")
	
class BNIError(Exception):
	def __init__(self, msg):
		self.msg = msg
	def __str__(self):
		return self.msg

class Net(Net):
	def __init__(self, fn = ""): # fn is a filename
		self.eNet = None # A pointer to the network as represented by the engine
		self._autoUpdate = True
		self.xdsl = None # The .xdsl XML
		
		self.needsUpdate = True
		self.forceUpdates = False
		
		# Caches
		self._nodeCache = {}
	
		self.eNet = g.new_network()
		if fn:
			# Just being unicode causes problems... Not sure how to
			# deal with a filename if it was genuinely unicode
			#fn2 = bytes(fn, 'ascii')
			fn2 = fn
			g.ReadFromFile(self.eNet, fn2)
			if fn.endswith(".xdsl"):
				self.xdsl = ET.parse(fn)
				self.xdsl.__class__.replaceNode = replaceNode
			elif fn.endswith(".dne"):
				# Fix up probs that GeNIe has with reading new Netica (>= V4) files
				contents = open(fn).read()
				# Strip out strings and comments (makes scanning for keywords simpler)
				# Strip out strings
				contents = re.sub(r'"(\\\\|\\"|[^"])*?"', '', contents)
				# Strip out comments
				contents = re.sub(r'//.*', '', contents)
				# Scan for nodes and their CPTs
				reg = re.compile(r'\b(node)\s+(\w+)\s*\{|\b(probs)\s*=([^;]*)')
				startPos = 0
				currentNode = None
				while 1:
					m = reg.search(contents, startPos)
					if m:
						if m.group(1)=="node":
							currentNode = m.group(2)
						elif m.group(3)=="probs":
							cptStr = m.group(4)
							# Flatten to 1D by splitting on commas/parentheses/spaces (any non-numeral symbol)
							cptStr = cptStr.strip("(), \t\r\n")
							cptStrs = re.split(r'[(),\s]+', cptStr)
							# Convert to floats
							cpt = [float(f) for f in cptStrs]
							# Update the in-memory CPT
							# Usually get err here?
							self.node(currentNode).cpt1d(cpt)
						startPos = m.end()
					else:
						break
	
	def __del__(self):
		g.delete_network(self.eNet)
	
	# |uniqueSet| is a list of strings or objects with .name() properties.
	# A name will be generated based on |name| that doesn't clash with anything
	# in |uniqueSet|.
	def makeValidName(self, name, uniqueSet = [], maxLength = 30):
		IDREGEX = r'([_a-zA-Z])([_0-9a-zA-Z]{0,29})' # NOTE: Not anchored at start,end
		def makeName(s):
			try:
				s = re.sub(r'^[^_a-zA-Z]', r'_\g<0>', re.sub(r'[^_0-9a-zA-Z]', '_', str(s)))[:maxLength]
				# GeNIe crash on _ at start? Yes, why?
				s = re.sub(r'^_+', '', s)
				# Make sure it starts with letter
				s = re.sub(r'^([^a-zA-Z])', r's\1', s)[:maxLength]
				# Make sure empty states also have a letter
				if not s:  s = 's'
				return s
			except:
				pdb.set_trace()
				return None
		
		hasNode = True
		newName = makeName(name)
		i = 1
		uniqueSet = [(n.name() if hasattr(n,'name') else n) for n in uniqueSet]
		while hasNode:
			if newName not in uniqueSet:
				hasNode = False
			else:
				newName = newName[:maxLength-len(str(i))] + str(i)
			i += 1
			if i > 1000:
				pdb.set_trace()
				return None
		
		return newName

	def submodels(self, submodelOnly = False, submodelId = 0):
		# if recursive:
			# submodelInts = g.submodel_GetDeepIncludedSubmodels(self.eNet, submodelId)
		# else:
			# submodelInts = g.submodel_GetIncludedSubmodels(self.eNet, submodelId)
		submodelInts = g.submodel_GetIncludedSubmodels(self.eNet, submodelId)
		n = g.intArray_NumItems(submodelInts)
		items = g.intArray_Items(submodelInts)
		
		submodels = []
		for i in range(n):
			submodelInt = items[i]
			submodels.append(Submodel(self, eId = submodelInt))
		
		if not submodelOnly:
			for submodel in submodels:
				submodels.extend( submodel.submodels(submodelOnly=submodelOnly) )
			
		return submodels
	
	def addSubmodel(self, submodelName):
		return Submodel(self, submodelName)
	
	def getSubmodel(self, submodelName = None, submodelId = None):
		submodelInt = submodelId
		if submodelId is None:
			submodelInt = g.submodel_FindSubmodel(self.eNet, submodelName)
		return Submodel(self, eId = submodelInt)
	
	# This is engine dependent
	def updateAlgorithm(self, algorithm = None):
		if algorithm is None:
			return g.GetDefaultBNAlgorithm(self.eNet)
		else:
			g.SetDefaultBNAlgorithm(self.eNet, algorithm)
			self.needsUpdate = True
		
		return self
	
	# autoUpdate can be turned off for performance. You then need
	# to manually use 'update' after changes and before reading beliefs.
	def autoUpdate(self, autoUpdate = None):
		if autoUpdate is None:
			return self._autoUpdate
		else:
			self._autoUpdate = autoUpdate
		
		return self

	def update(self, forceUpdates = False):
		if self.needsUpdate or forceUpdates or self.forceUpdates:
			#print("Updating")
			g.UpdateBeliefs(self.eNet)
			self.needsUpdate = False
		
		return self
		
	def _setSamples(self, samples, discreteSamples):
		g.SetNumberOfSamples(self.eNet, samples)
		g.SetNumberOfDiscretizationSamples(self.eNet, discreteSamples)
		self.needsUpdate = True
		
	def name(self, _name = None, check = False):
		header = g.net_Header(self.eNet)
		if _name is None:
			return g.header_GetId(header)
		else:
			if check:
				_name = self.net.makeValidName(_name)
			g.header_SetId(header, _name)
			
		return self
			
	def title(self, _title = None):
		header = g.net_Header(self.eNet)
		if _title is None:
			return g.header_GetName(header)
		else:
			g.header_SetName(header, _title)
			
		return self
	
	def comment(self, _comment = None):
		header = g.net_Header(self.eNet)
		if _comment is None:
			return g.header_GetComment(header)
		else:
			g.header_SetComment(header, _title)
			
		return self
			
	def write(self, fn):
		g.WriteToFile(self.eNet, fn)
		if re.search(r'\.dne$', fn):
			# If we're writing a .dne, fix the bounds
			with open(fn) as dneFile:
				maxX = max(n.position()[0] for n in self.nodes())+50
				maxY = max(n.position()[1] for n in self.nodes())+50
				toWrite = re.sub(r'(visual V1 \{\n)', r'\1	drawingbounds = ('+f'{maxX}, {maxY});\n', dneFile.read())
			with open(fn, 'w') as dneFile:
				dneFile.write(toWrite)
		
		return self
		
	def node(self, name):
		if name in self._nodeCache:  return self._nodeCache[name]
		
		nodeId = g.FindNode(self.eNet, name)
		if nodeId == -2:
			return None
		self._nodeCache[name] = Node(net = self, genieNodeId = nodeId)
		
		return self._nodeCache[name]

	def addNode(self, name, nodeType = None, states = None):
		self.needsUpdate = True
		return Node(self, name, nodeType = nodeType, states = states)
	
	def compile(self):
		# No compile phase in GeNIe. Might do something in future.
		pass
	
	def retractFindings(self):
		g.ClearAllEvidence(self.eNet)
		self.needsUpdate = True
		
	def findings(self, findings = None):
		if findings is None:
			findings = {}
			for node in self.nodes():
				findings[node.name()] = node.finding()
			
			return findings
		else:
			for nodeName,finding in findings.items():
				if finding is None:
					self.node(nodeName).retractFindings()
				else:
					self.node(nodeName).finding(finding)
		
		return self
	
	def nodes(self, submodelOnly = False, submodelId = 0):
		if submodelOnly or submodelId != 0:
			nodeInts = g.submodel_GetIncludedNodes(self.eNet, submodelId)
			n = g.intArray_NumItems(nodeInts)
			items = g.intArray_Items(nodeInts)
			
			nodes = []
			for i in range(n):
				nodeInt = items[i]
				nodes.append(Node(self, genieNodeId = nodeInt))
			
			if not submodelOnly:
				for submodel in self.getSubmodel(submodelId = submodelId).submodels():
					nodes.extend( submodel.nodes(submodelOnly = submodelOnly) )
			
			return nodes
			
		else:
			intArray = g.new_intArray()
			g.GetAllNodes(self.eNet, intArray)
			numItems = g.intArray_NumItems(intArray)
			items = g.intArray_Items(intArray)
			nodeList = []
			for i in range(numItems):
				nodeList.append(Node(self,genieNodeId=items[i]))
			return nodeList
			
	def findingsProbability(self):
		return g.CalcProbEvidence(self.eNet)
	
	# def arcWeight(parent, child):
		# self.xdsl.iter(

	# Not sure these belong here
	@classmethod
	def numberCombinations(cls, nodes):
		total = 1

		for i in range(len(nodes)):
			total *= nodes[i].numberStates()

		# Did something need this to be 0?
		return total # if len(nodes)>0 else 0

	# This function works for discrete nodes only, of course
	# nodeStates should be an array the size of len(nodes)
	@classmethod
	def nextCombination(cls, nodeStates, nodes, skip = []):
		# Flip skip array (which also makes it a dict)
		fs = {}
		for i,v in enumerate(skip): fs[v] = skip[i]
		skip = fs
		
		numNodes = len(nodeStates)
		for i in range(numNodes-1,-1,-1):
			if i in skip: continue
			nodeStates[i] += 1
			if nodeStates[i] >= nodes[i].numberStates():
				# Set the i^th node state to 0 and continue to next node
				nodeStates[i] = 0
			else:
				# More combinations to come
				return True
		# All node states have rolled back round to 0
		return False

	# This probably works more nicely than nextCombination for most use cases
	def CombinationIterator(net, nodes, returnType = 'indexes'):
		nodes = [net.node(s) if isinstance(s,str) else s for s in nodes]
		# Get the arity for each node
		nodeArities = [n.numberStates() for n in nodes]
		
		# |nodeStates| is our "odometer"
		# Initialise the odometer to all zeroes
		nodeStates = [0]*len(nodes)
		
		# Loop through every row in the CPT
		hasMore = True
		while hasMore:
			# We are using generators, so we can pause
			# and re-start the function. Each time, we
			# yield the current state of the odometer
			if returnType == 'indexes':
				yield nodeStates
			elif returnType == 'names':
				yield [node.state(s).name() for node,s in zip(nodes,nodeStates)]
			elif returnType == 'dict':
				yield {node.name():node.state(s).name() for node,s in zip(nodes,nodeStates)}
			elif returnType == 'dictTitle':
				yield {(node.title() or node.name()):node.state(s).name() for node,s in zip(nodes,nodeStates)}
			
			# On each loop, increment the odometer
			hasMore = False
			for i in range(len(nodeStates)-1,-1,-1):
				nodeStates[i] += 1
				# If we don't need to "Carry the 1", stop
				if nodeStates[i] < nodeArities[i]:
					hasMore = True
					break

				# Otherwise, carry the 1 and keep looping
				nodeStates[i] = 0

		
	# Learning -- SM: I don't actually know what learningRate and initialLearningPoint
	# are supposed to do.
	def learn(self, fileName, type = 'EM', experience = 1,
			relevance = False, learningRate = 0.6, initialLearningPoint = 1, randomize = None,
			uniformize = None):
		
		if uniformize:
			if uniformize == True:
				uniformize = self.nodes()
			for uNode in uniformize:
				if isinstance(uNode,str):
					uNode = self.node(uNode)
				if uNode is not None:
					uNode.setUniform()
		
		if randomize:
			if randomize == True:
				randomize = self.nodes()
			for rNode in randomize:
				if isinstance(rNode,str):
					rNode = self.node(rNode)
				if rNode is not None:
					rNode.setRandom()
					#rNode.experience(1)
		
		if type == 'EM':
			g.learn_em(self.eNet, fileName, experience,
				relevance, learningRate, initialLearningPoint)

		self.needsUpdate = True
		
		# Chaining
		return self
	
	# Use net._pcMatrix after running this with PC, if need to see specific arc types prior to DAG-ification
	def learnStructure(self, fileName, type = 'PC',
			maxAdjacency = 8, maxSearchTime = 0, significance = 0.05, # PC parameters
			numIterations = 20, maxParents = 5, # Bayesian Search parameters (there are more, but best use CaMML)
			classVar = '', matrixOnly = True, # + maxSearchTime. TAN parameters (classVar is of course required)
			tiers = None): # tiers -> [[<col1>,<col2>],[<col3>],...]  for PC only for now
		if type == 'PC':
			tierPriorSpecSize = (sum(len(t) for t in tiers)*2) if tiers else 0
			tierPriorSpec = (ctypes.c_int*tierPriorSpecSize)() if tiers else None
			import csv
			with open(fileName) as csvFile:
				inCsv = csv.reader(csvFile)
				headers = next(inCsv)
			colMap = {h:i for i,h in enumerate(headers)}
			if tiers:
				i = 0
				for tierI,tier in enumerate(tiers):
					for nodeName in tier:
						if nodeName not in colMap:
							print(nodeName, ' column not found')
							return
						tierPriorSpec[i] = colMap[nodeName]
						tierPriorSpec[i + 1] = tierI
						i += 2
			matrix = g.learn_pc(self.eNet, fileName, maxAdjacency, maxSearchTime, significance, tierPriorSpecSize, tierPriorSpec)
			matrixSize = matrix[0]
			matrix = matrix[1:matrixSize+1]
			numNodes = int(math.sqrt(matrixSize))
			pcMatrix = []
			st = ''
			pcArcList = []
			for i in range(numNodes):
				pcMatrix.append([])
				for j in range(numNodes):
					dependencyType = matrix[i*numNodes + j]
					pcMatrix[i].append(dependencyType)
					if dependencyType > 0:
						pcArcList.append([self.nodes()[i], self.nodes()[j], dependencyType])
					st += f'{dependencyType:2} '
				st += '\n'
			self._pcMatrix = pcMatrix
			self._pcArcList = pcArcList
		elif type == 'BS':
			print(g.learn_bs(self.eNet, fileName, numIterations, maxParents))
		elif type == 'TAN':
			print(g.learn_tan(self.eNet, fileName, classVar, maxSearchTime))
			
		self.needsUpdate = True
		
		return self

	# XXX-SM-2021-07-06: This needs work and to be cleaned up (and see the CAT/node.js version), but better
	# here than nowhere
	def sensitivityToFindings(self, targetNode, mi = True, expRankChange = True, nodes = None,
			targetAsSource = False):
		targetNode = self.node(targetNode)
		net = self
		# t = time.time()
		
		nodes = nodes or net.nodes()
		
		# Get marginals (with whatever current evidence is)
		marginals = {}
		# print('a', time.time() - t)
		for node in nodes:
			marginals[node.name()] = node.beliefs()
		
		# Store all node beliefs for every different state in target
		beliefsByTargetState = []
		# print('b', time.time() - t)
		for state in targetNode.states():
			state.setTrueFinding()
			beliefs = {}
			for node in nodes:
				# The below is wrong, as it requires the marginals to be recomputed as well
				# if node.name() != targetNode.name() and node.hasFinding():
					# saved = node.finding()
					# node.retractFindings()
					# beliefs[node.name()] = node.beliefs()
					# node.finding(saved)
				# else:
					# beliefs[node.name()] = node.beliefs()
				beliefs[node.name()] = node.beliefs()
			beliefsByTargetState.append(beliefs)
		# print('c', time.time() - t)
		targetNode.retractFindings()
		
		# Now, calculate the MI table
		miTable = []
		targetMarginals = marginals[targetNode.name()]
		targetCondProbs = {}
		for node in nodes:
			# joint * log ( joint / marginals)
			# For each prob in target marginal
			total = 0
			targetCondProbs[node.name()] = [[0 for s2 in targetNode.states()] for s in node.states()]
			for i,targetMarginalProb in enumerate(targetMarginals):
				nodeMarginal = marginals[node.name()]
				# And each prob in both marginal and conditional node beliefs
				for j,nodeProb in enumerate(beliefsByTargetState[i][node.name()]):
					jointProb = targetMarginalProb*nodeProb
					nodeMarginalProb = nodeMarginal[j]
					
					if jointProb * targetMarginalProb * nodeMarginalProb != 0:
						#print(node.name(), jointProb, targetMarginalProb, nodeMarginalProb)
						total += jointProb * math.log( jointProb / (targetMarginalProb * nodeMarginalProb), 2 )
						
						targetCondProb = jointProb / nodeMarginalProb
					else:
						targetCondProb = 0
					
					targetCondProbs[node.name()][j][i] = targetCondProb
					#.append([i, node.name(), j, targetCondProb])
			
			minExpRank = 10000000
			maxExpRank = -1
			minExpRankJ = -1
			maxExpRankJ = -1
			if targetAsSource:
				# Get, for every target state, the current node's beliefs (conditional on target state)
				nodeProbs = [nodeBels[node.name()] for nodeBels in beliefsByTargetState]
			else:
				# Get, for every node state, the target's beliefs (conditional on node state)
				nodeProbs = targetCondProbs[node.name()]
				
			for j,row in enumerate(nodeProbs):
				expRank = sum(i*p for i,p in enumerate(row))
				if expRank < minExpRank:
					minExpRank = expRank
					minExpRankJ = j
				if expRank > maxExpRank:
					maxExpRank = expRank
					maxExpRankJ = j
			miTable.append([node.name(), total, maxExpRank-minExpRank, minExpRankJ, maxExpRankJ, str(node.title().encode('ascii',errors='ignore'), 'ascii')])
		
		#print(json.dumps(targetCondProbs, indent='\t'))
		
		return sorted(miTable, key = lambda x: x[1], reverse=True)

	# Just a helper function for sensitivityToFindings
	# XXX: Doesn't belong here
	def sensTableToHtml(self, sensTable, colors = True):
		from htm import n
		table = n('table', n('tr', n('th', 'Node'), n('th', 'MI'), n('th', 'Max Exp Rank Change')))
		NAME = 0
		TITLE = 5
		MI = 1
		RANKCHANGE = 2
		
		minMi = 0
		maxMi = 0
		minRankChange = 0
		maxRankChange = 0
		for i,row in enumerate(sensTable):
			if i==0:  continue
			maxMi = max(row[MI], maxMi)
			print(row[RANKCHANGE])
			maxRankChange = max(row[RANKCHANGE], maxRankChange)
		print(maxRankChange)
		
		for row in sensTable:
			displayRow = row[:RANKCHANGE+1]
			displayRow[RANKCHANGE] = math.copysign(row[RANKCHANGE], row[4]-row[3])
			displayRow[NAME] = row[TITLE]
			tr = n('tr')
			for i,v in enumerate(displayRow):
				intensity = None
				if i==MI:
					intensity = (v-minMi)/(maxMi-minMi)
				elif i==RANKCHANGE:
					intensity = (abs(v)-minRankChange)/(maxRankChange-minRankChange)
				try:
					fmtVal = f'{float(v):.4f}'
				except:
					fmtVal = v
				tr.append(n('td', fmtVal, style=f'--intensity: {intensity}' if intensity is not None else ''))
			table.append(tr)
		
		return table

		

class Submodel:
	# parent is id of parent submodel (or None if no parent)
	def __init__(self, net, submodelName = None, parent = None, eId = None):
		self.net = None
		self.eId = None
	
		self.net = net
		parentInt = 0
		if parent is not None:
			parentInt = g.submodel_FindSubmodel(self.net.eNet, parent)
		if eId is not None:
			self.eId = eId
			# This acts as a kind of validation
			g.submodel_GetSubmodel(self.net.eNet, self.eId)
		else:
			self.eId = g.submodel_CreateSubmodel(self.net.eNet, parentInt, submodelName)
	
	# Some utility functions, unlikely to be useful outside
	def _gSubmodel(self):
		return g.submodel_GetSubmodel(self.net.eNet, self.eId)
		
	def _gSubmodelHdr(self):
		return g.submodel_Header(self._gSubmodel())
	
	def _gSubmodelScreenInfo(self):
		return g.submodel_ScreenInfo(self._gSubmodel())
		
	def delete(self):
		g.submodel_DeleteSubmodel(self.net.eNet, self.eId)
		self.net = None
		self.eId = None
	
	def name(self, name = None, check = False):
		hdr = self._gSubmodelHdr()
		
		if name is None:
			return g.header_GetId(hdr)
		else:
			if check:
				uniqueSet = [s.name() for s in self.net.submodels()]
				uniqueSet.remove(self.name())
				name = self.net.makeValidName(name, uniqueSet=uniqueSet)
			g.header_SetId(hdr, name)
		
		return self
	
	def title(self, title = None):
		hdr = self._gSubmodelHdr()
		
		if title is None:
			return g.header_GetName(hdr)
		else:
			g.header_SetName(hdr, title)
		
		return self
	
	# Get parent submodel (or set it)
	def parentSubmodel(self, parentSubmodel = None):
		if parentSubmodel is None:
			parentEId = g.submodel_GetParent(self.net.eNet, self.eId)
			# The only way to see we're at the top is to check for the parent's parent
			# XXX I'm not sure this is right
			if g.submodel_GetParent(self.net.eNet, parentEId) >= 0:
				return Submodel(self.net, eId = parentEId)
			return None
		else:
			submodelName = parentSubmodel if isString(parentSubmodel) else parentSubmodel.name()
			newSubmodelId = g.submodel_FindSubmodel(self.net.eNet, submodelName)
			g.submodel_MoveSubmodel(self.net.eNet, self.eId, newSubmodelId)
		
		return self
	
	def submodels(self, *args, **kwargs):
		return self.net.submodels(submodelId = self.eId, *args, **kwargs)
	
	def nodes(self, *args, **kwargs):
		return self.net.nodes(submodelId = self.eId, *args, **kwargs)
	
	def addSubmodel(self, name):
		return Submodel(self.net, name, self.name())
	
	def getSubmodel(self, *args, **kwargs):
		return self.net.getSubmodel(*args, **kwargs)
	
	def position(self, x = None, y = None):
		screenInfo = self._gSubmodelScreenInfo()
		position = g.screenInfo_position(screenInfo)
		if x:
			g.rectangle_center_X_set(position, x)
		if y:
			g.rectangle_center_Y_set(position, y)
		if x is None and y is None:
			return [g.rectangle_center_X(position), g.rectangle_center_Y(position)]
		else:
			return self

	def size(self, width = None, height = None):
		screenInfo = self._gSubmodelScreenInfo()
		position = g.screenInfo_position(screenInfo)
		if width:
			g.rectangle_width_set(position, width)
		if height:
			g.rectangle_height_set(position, height)
		if width is None and height is None:
			return [g.rectangle_width(position), g.rectangle_height(position)]
		else:
			return self

	def addNode(self, *args, **kwargs):
		node = self.net.addNode(*args, **kwargs)
		node.parentSubmodel(self.name())
		return node
		


class Node(Node):
	def __init__(self, net = None, name = None, states = None, nodeType = None, genieNodeId = None):
		self.net = None
		self.eId = None # The ID (integer) of this node in GeNIe (for the given net)
		self._nodeObjCache = None
		self._nodeHdrCache = None
		self._nodeDefCache = None
		self._nodeValCache = None
		self._nodeInfoCache = None
		self._screenInfoCache = None
		self._equationCache = {}
	
		self.net = net
		self.eId = genieNodeId
		self._states = None
		self._stateNames = None
		self._statesLookup = None
		if name is not None:
			if not self.checkValidName(name):
				raise BNIError("Node name "+repr(name)+" is not valid. Must have " +
					"first character as letter/underscore, "+
					"other characters as letter/number/underscore and max. 30 characters")
			if nodeType is None: nodeType = Node.NATURE_NODE
			self.eId = g.AddNode(self.net.eNet, Node.NODE_TYPE_MAP[nodeType], name)
		
		if states:
			for i in range(2,len(states)):
				self.addState(states[i])
			self.renameStates(states)
	
	def __eq__(self, other):
		return self.eId == other.eId
	def __hash__(self):
		return hash(self.eId)
	
	# Some utility functions, unlikely to be useful outside
	def _gNode(self):
		if self._nodeObjCache is not None:  return self._nodeObjCache
		self._nodeObjCache = g.GetNode(self.net.eNet, self.eId)
		return self._nodeObjCache
	
	def _gNodeHdr(self):
		if self._nodeHdrCache is not None: return self._nodeHdrCache
		# Not sure why it's so buried away in GeNIe
		myNodePtr = g.GetNode(self.net.eNet, self.eId)
		nodeInfoPtr = g.node_Info(myNodePtr)
		headerPtr = g.nodeInfo_Header(nodeInfoPtr)
		self._nodeHdrCache = headerPtr
		return headerPtr
	
	def _gNodeDef(self):
		if self._nodeDefCache is not None: return self._nodeDefCache
		nodePtr = self._gNode()
		self._nodeDefCache = g.node_Definition(nodePtr)
		return self._nodeDefCache
	
	def _gNodeVal(self):
		if self._nodeValCache is not None: return self._nodeValCache
		self._nodeValCache = g.node_Value(self._gNode())
		return self._nodeValCache
	
	def _gNodeInfo(self):
		if self._nodeInfoCache is not None: return self._nodeInfoCache
		self._nodeInfoCache =  g.node_Info(self._gNode())
		return self._nodeInfoCache
	
	def _gScreenInfo(self):
		if self._screenInfoCache is not None: return self._screenInfoCache
		self._screenInfoCache =  g.nodeInfo_Screen(self._gNodeInfo())
		return self._screenInfoCache
	
	def delete(self):
		if self.name() in self.net._nodeCache:
			del self.net._nodeCache[self.name()]
		g.DeleteNode(self.net.eNet, self.eId)
		self.net.needsUpdate = True
		self.net = None
		self.eId = None
	
	def checkValidName(self, name):
		# Netica convention: first character is letter/underscore,
		# other characters are letter/number/underscore
		# Max 30 characters
		if re.match(r'[a-zA-Z_][a-zA-Z0-9_]{,29}', name):
			return True
		else:
			return False
			
	def name(self, name = None, check = False):
		header = self._gNodeHdr()
		if name is None:
			return g.header_GetId(header)
		else:
			oldName = self.name()
			if check:
				uniqueSet = [n.name() for n in self.net.nodes()]
				uniqueSet.remove(self.name())
				name = self.net.makeValidName(name, uniqueSet=uniqueSet)
			g.header_SetId(header, name)
			if oldName in self.net._nodeCache:
				self.net._nodeCache[name] = self.net._nodeCache[oldName]
				del self.net._nodeCache[oldName]
			
		return self
		
	def title(self, _title = None):
		header = self._gNodeHdr()
		if _title is None:
			return g.header_GetName(header)
		else:
			g.header_SetName(header, _title)
			
		return self
		
	def comment(self, _comment = None):
		header = self._gNodeHdr()
		if _comment is None:
			return g.header_GetComment(header)
		else:
			g.header_SetComment(header, _comment)
			
		return self
	
	def format(self, color = None):
		# formatOptions are ints of [color, selColor, font, fontColor, borderThickness, borderColor]
		formatOptions = [-1 for i in range(6)]
		optionSet = False
		if color is not None:
			formatOptions[0] = color
			optionSet = True
		#if 
		if not optionSet:
			intPtr = g.screenInfo_formatting(self._gScreenInfo())
			numItems = 6
			
			formatItems = [intPtr[i] for i in range(numItems)]
			
			return formatItems
		else:
			g.screenInfo_setFormatting(self._gScreenInfo(), (ctypes.c_int*6)(*formatOptions))
		
		return self
	
	def features(self):
		nodeFeatures = g.nodeDefinition_GetType(self._gNodeDef())
		features = []
		for feature,featureMask in Node.FEATURES.items():
			if nodeFeatures & featureMask:
				features.append(feature)
		return features
	
	def hasFeature(self, features):
		return features in self.features()

	# Use the Node.TYPE constants
	def type(self, _type = None):
		if _type is None:
			return Node.NODE_TYPE_MAP_REV[g.nodeDefinition_GetType(self._gNodeDef())]
		else:
			g.node_ChangeType(self._gNode(), self.NODE_TYPE_MAP[_type])
		
		return self
	
	def parents(self):
		parentIds = g.GetParents(self.net.eNet, self.eId)
		parentIdItems = g.intArray_Items(parentIds)
		numItems = g.intArray_NumItems(parentIds)
		parents = []
		for i in range(numItems):
			parents.append(Node(self.net, genieNodeId=parentIdItems[i]))
		return parents

	# Slow impl.
	def children(self):
		allNodes = self.net.nodes()
		children = []
		for node in allNodes:
			if self.name() in [p.name() for p in node.parents()]:
				children.append(node)
		return children	
		
	def addParents(self, parents):
		"""
		Each element of parents can be an existing node name or node.
		"""
		for parent in parents:
			# Make sure each parent is a Node object
			if isString(parent):
				parent = self.net.node(parent)
			g.AddArc(self.net.eNet, parent.eId, self.eId)
		
		self.net.needsUpdate = True
		# Allow the call chain to continue
		return self
		
	def addChildren(self, children):
		"""
		Each element of children can be an existing node name or node.
		"""
		for child in children:
			# Make sure each parent is a Node object
			if isString(child):
				child = self.net.node(child)
			g.AddArc(self.net.eNet, self.eId, child.eId)

		self.net.needsUpdate = True
		# Allow the call chain to continue
		return self
	
	def removeParents(self, parents):
		"""
		Each element of parents can be an existing node name or node.
		"""
		for parent in parents:
			# Make sure each parent is a Node object
			if isString(parent):
				parent = self.net.node(parent)
			g.RemoveArc(self.net.eNet, parent.eId, self.eId)
		
		self.net.needsUpdate = True
		# Allow the call chain to continue
		return self
		
	def removeChildren(self, children):
		"""
		Each element of children can be an existing node name or node.
		"""
		for child in children:
			# Make sure each parent is a Node object
			if isString(child):
				child = self.net.node(child)
			g.RemoveArc(self.net.eNet, self.eId, child.eId)

		self.net.needsUpdate = True
		# Allow the call chain to continue
		return self
	
	def _clearStatesCache(self):
		self._stateNames = None
		self._statesLookup = None
		self._states = None
	
	def _setupStates(self, force = False):
		# set 'force' or erase _stateNames to clear cache
		if not force and self._stateNames: return
		
		self._stateNames = self.stateNames()
		self._statesLookup = dict((k,State(self,i)) for i,k in enumerate(self._stateNames))
		self._states = []
		for stateName in self._stateNames:
			self._states.append(self._statesLookup[stateName])
		
	def state(self, name, case = True):
		self._setupStates()
		
		if isinstance(name, State):
			name = name.name()
		# If int (assumed if not str), then just get state
		if not isString(name):
			return self._states[name]
		
		return self._statesLookup[name]
		
	def states(self):
		self._setupStates()
		
		return self._states
		
	def hasState(self, name):
		self._setupStates()
		
		return name in self._statesLookup
	
	def addState(self, name):
		nd = self._gNodeDef()
		g.nodeDefinition_AddOutcome(nd, name)
		
		self.net.needsUpdate = True
		# Chain
		return self
	
	def renameState(self, name, newName):
		nd = self._gNodeDef()
		stateNames = self.stateNames()
		state = self.state(name)
		stateNames[state.stateNum] = newName
		
		stateNamesArg = (c_simplechar_p*len(stateNames))(*[bytes(s,'ascii',errors='ignore') for s in stateNames])
		
		g.nodeDefinition_RenameOutcomes(nd, len(stateNames), stateNamesArg)
		
		self._clearStatesCache()
		
		# Chain
		return self
		
	def renameStates(self, newNames):
		nd = self._gNodeDef()

		if self.type()==Node.EQUATION_NODE:
			levels = self.levels()[1:]
			data = [(name,upper) for name,upper in zip(newNames,levels)]
			# Create an array of KeyValue structs
			keyValues = (KeyValue * len(data))(
				*[KeyValue(ctypes.c_char_p(k.encode("utf-8")), v) for k, v in data]
			)
			intervals = KeyValueArray(keyValues, len(data))
			g.equation_SetDiscreteIntervals(nd, intervals)
		else:
			stateNamesArg = (c_simplechar_p*len(newNames))(*[bytes(s,'ascii',errors='ignore') for s in newNames])
			
			g.nodeDefinition_RenameOutcomes(nd, len(newNames), stateNamesArg)
		
		self._clearStatesCache()
		
		# Chain
		return self
		
	def reorderStates(self, ordering):
		ordering = [self.state(s).stateNum for s in ordering]
		g.nodeDefinition_ChangeOrderOfOutcomes(self._gNodeDef(), len(ordering), (ctypes.c_int*len(ordering))(*ordering))
		
		return self
		
	def setEquation(self, equationStr):
		nd = self._gNodeDef()
		
		self.net.needsUpdate = True
		return g.equation_SetEquation(nd, equationStr)
	
	def equation(self, equationStr = None):
		# XXX
		# XXX
		# XXX Need to update equation if any variable names change
		# XXX
		# XXX Add checks if node is an equation type node
		# XXX
		if not equationStr:
			equationStr = self._equationCache.get('equation')
			if equationStr:
				return equationStr
			else:
				# Unlike equations, there is a way to get the string directly for MAUs, but this
				# is easier/consistent.
				if self.type() == Node.MAU_NODE:
					node = self.net.xdsl.find('.//nodes/maux[@id="'+self.name()+'"]/expression')
				else:
					node = self.net.xdsl.find('.//nodes/equation[@id="'+self.name()+'"]/definition')
				return node.text
		else:
			nd = self._gNodeDef()
			if self.type() == Node.MAU_NODE:
				g.mau_SetExpression(nd, equationStr)
			else:
				g.equation_SetEquation(nd, equationStr)
			self._equationCache['equation'] = equationStr
			self.net.needsUpdate = True
	
		return self

	def levels(self, levels = None):
		nd = self._gNodeDef()
		if levels is None:
			bounds = g.equation_GetBounds(nd)
			result = g.equation_GetDiscreteIntervals(nd)
			levels = [bounds[0]]
			# levels = [bounds[0]]
			for i in range(result.size):
				levels.append(result.items[i].value)
			return levels
		else:
			names = self.stateNames()
			names = names[:len(levels)]
			for i in levels[len(names):]:
				names.append('')
			g.equation_ClearDiscreteIntervals(nd)
			g.equation_SetBounds(nd, levels[0], levels[-1])
			data = [(name,upper) for name,upper in zip(names, levels[1:])]

			# Create an array of KeyValue structs
			keyValues = (KeyValue * len(data))(
				*[KeyValue(ctypes.c_char_p(k.encode("utf-8")), v) for k, v in data]
			)

			# Create a KeyValueArray
			keyValueArray = KeyValueArray(keyValues, len(data))

			g.equation_SetDiscreteIntervals(nd, keyValueArray)
		
		return self
		
	def setExperience(self, parentStates, experience):
		# It's not clear to me GeNIe supports this
		NYI()
	
	def numberStates(self):
		nodeDef = self._gNodeDef()
		return g.nodeDefinition_GetNumberOfOutcomes(nodeDef)
		
	def retractFindings(self):
		gNode = self._gNode()
		gValue = g.node_Value(gNode)
		g.nodeValue_ClearEvidence(gValue)
		self.net.needsUpdate = True
	
	def likelihoods(self, likelihoodVector = None):
		if likelihoodVector is None:
			evArr = g.nodeValue_GetVirtualEvidence(self._gNodeVal())
			size = int(evArr[0])
			retArr = []
			for i in range(size):
				retArr.append(evArr[i+1])
			return retArr
		else:
			from struct import pack
			n = len(likelihoodVector)
			dp = (ctypes.c_double*n)(*likelihoodVector)
			g.nodeValue_SetVirtualEvidence(self._gNodeVal(),
				len(likelihoodVector), dp)
			self.net.needsUpdate = True
	
	def hasFinding(self):
		gNodeValue = self._gNodeVal()
		
		return bool(g.nodeValue_IsRealEvidence(gNodeValue))
	
	def finding(self, state = None, value = None):
		gNodeValue = self._gNodeVal()
		if state is None and value is None:
			if self.hasFinding():
				ev = g.valEqEvaluation_GetEvidence(gNodeValue)
				if self.type()==Node.EQUATION_NODE:
					return ev
				return self.state(ev)
			else:
				return None
		else:
			if value is not None:
				g.valEqEvaluation_SetEvidence(gNodeValue, value)
			else:
				self.state(state).setTrueFinding()
			self.net.needsUpdate = True
			
		return self
	
	def beliefs(self):
		if self.net._autoUpdate:
			self.net.update()
		
		gNode = self._gNode()
		gNodeValue = g.node_Value(gNode)
		#rint "node value type:", g.nodeValue_GetType(gNode)
		gMat = g.nodeValue_GetMatrix(gNodeValue)
		
		gSize = g.dMatrix_GetSize(gMat)
		gDbl = g.dMatrix_GetItemsDouble(gMat)
		beliefs = []
		for i in range(gSize):
			beliefs.append(gDbl[i])
		
		return beliefs

	def samples(self):
		if self.net._autoUpdate:
			self.net.update()
		
		gNode = self._gNode()
		gNodeValue = g.node_Value(gNode)
		numSamples = g.valEqEvaluation_GetNumberOfSamples(gNodeValue)
		samplePtr = g.valEqEvaluation_GetSamples(gNodeValue)
		
		return [samplePtr[i] for i in range(numSamples)]
	
	# Use mean() or expectedValue() instead
	def _equationMean(self):
		if self.net._autoUpdate:
			self.net.update()
		
		gNode = self._gNode()
		gNodeValue = g.node_Value(gNode)
		
		return g.valEqEvaluation_GetMean(gNodeValue)
	
	# Use sd() instead
	def _equationSd(self):
		if self.net._autoUpdate:
			self.net.update()
		
		gNode = self._gNode()
		gNodeValue = g.node_Value(gNode)
		
		return g.valEqEvaluation_GetStdDev(gNodeValue)

	# SM: equationSd is preferred. In fact, can probably remove this,
	# as unlikely anything is using it
	def _equationStdDev(self):
		return self._equationSd()
	
	def expectedValue(self):
		return self.mean()
	
	def mean(self):
		return self._equationMean()
	
	def sd(self):
		return self._equationSd()
	

		
	def probs(self, parentStates):
		NYI()
	def stateNames(self):
		nodeDef = self._gNodeDef()
		if self.type()==Node.EQUATION_NODE:
			keyValues = g.equation_GetDiscreteIntervals(nodeDef)
			names = [keyValues.items[i].key.decode('UTF-8') for i in range(keyValues.size)]
			return names
		else:
			stateNameArray = g.nodeDefinition_GetOutcomesNames(nodeDef)
			charStarStar = g.stringArray_Items(stateNameArray)
			
			stateNames = []
			for i in range(self.numberStates()):
				stateNames.append(charStarStar[i].value.decode('utf-8'))
				
			return stateNames
	
	def position(self, x = None, y = None):
		node = g.GetNode(self.net.eNet, self.eId)
		nodeInfo = g.node_Info(node)
		screenInfo = g.nodeInfo_Screen(nodeInfo)
		position = g.screenInfo_position(screenInfo)
		if x:
			g.rectangle_center_X_set(position, x)
		if y:
			g.rectangle_center_Y_set(position, y)
		if x is None and y is None:
			return [g.rectangle_center_X(position), g.rectangle_center_Y(position)]
		else:
			return self

	def size(self, width = None, height = None):
		node = g.GetNode(self.net.eNet, self.eId)
		nodeInfo = g.node_Info(node)
		screenInfo = g.nodeInfo_Screen(nodeInfo)
		position = g.screenInfo_position(screenInfo)
		if width:
			g.rectangle_width_set(position, width)
		if height:
			g.rectangle_height_set(position, height)
		if width is None and height is None:
			return [g.rectangle_width(position), g.rectangle_height(position)]
		else:
			return self

	def cpt1d(self, newCpt = None):
		if newCpt is None:
			node = g.GetNode(self.net.eNet, self.eId)
			nodeDef = g.node_Definition(node)
			nodeMat = g.nodeDefinition_GetMatrix(nodeDef)
			numItems = g.dMatrix_GetSize(nodeMat)

			cptDbl = []
			dbArr = g.dMatrix_GetItemsDouble(nodeMat)
			for i in range(numItems):
				cptDbl.append(dbArr[i])

			return cptDbl
		else:
			nd = self._gNodeDef()
			
			numStates = self.numberStates()
			
			totalParams = len(newCpt)
			rows = int(totalParams/numStates)
			
			# Normalisation
			for r in range(rows):
				r *= numStates
				totalRow = sum(newCpt[r:r+numStates])
				if totalRow:
					for i in range(r, r+numStates):
						newCpt[i] = newCpt[i]/totalRow
			
			nc = (ctypes.c_double*len(newCpt))(*newCpt)
			
			g.nodeDefinition_SetDoubleDefinition(nd, len(newCpt), nc)
			self.net.needsUpdate = True
		
		# Chain
		return self
	
	def cpt(self, newCpt = None):
		if newCpt is None:
			cpt1d = self.cpt1d()
			numStates = self.numberStates()

			def chunks(l, n):
				return [l[i:i+n] for i in range(0, len(l), n)]

			cpt = chunks(cpt1d, numStates)

			return cpt
		else:
			nd = self._gNodeDef()
			
			numStates = self.numberStates()
			
			totalParams = len(newCpt)
			rows = totalParams/numStates
			
			# Normalisation
			newCpt2 = []
			for row in newCpt:
				totalRow = sum(row)
				if totalRow:
					for i in range(len(row)):
						row[i] = row[i]/totalRow
				newCpt2.extend(row)
			
			nc = (ctypes.c_double*len(newCpt2))(*newCpt2)
			
			g.nodeDefinition_SetDoubleDefinition(nd, len(newCpt2), nc)
			self.net.needsUpdate = True
		
		# Chain
		return self
	
	# Can be used with utility nodes to get/set utilities
	def utilities(self, newUtilities = None):
		# Get
		if newUtilities is None:
			node = g.GetNode(self.net.eNet, self.eId)
			nodeDef = g.node_Definition(node)
			nodeMat = g.nodeDefinition_GetMatrix(nodeDef)
			numItems = g.dMatrix_GetSize(nodeMat)

			# Utils are one dimensional (one for each parent combo)
			utilDbl = []
			dbArr = g.dMatrix_GetItemsDouble(nodeMat)
			for i in range(numItems):
				utilDbl.append(dbArr[i])

			return utilDbl
		# Set
		else:
			nd = self._gNodeDef()
			nc = (ctypes.c_double*len(newUtilities))(*newUtilities)
			
			g.nodeDefinition_SetDoubleDefinition(nd, len(newUtilities), nc)
			self.net.needsUpdate = True
	
	# For decision/utility nodes. XXX Need to add checks
	def expectedUtilities(self):
		return self.beliefs()
	
	# Get indexing parents for decision/utility nodes
	def indexingParents(self):
		self.net.update()
		indexingParentIds = g.nodeValue_GetIndexingParents(self._gNodeVal())
		parentIdItems = g.intArray_Items(indexingParentIds)
		numItems = g.intArray_NumItems(indexingParentIds)
		indexingParents = []
		for i in range(numItems):
			indexingParents.append(Node(self.net, genieNodeId=parentIdItems[i]))
		return indexingParents
		
	# Can be used with decision nodes to get/set decision options
	# No longer needed in GeNIe?, can just use set state names
	# def options(self, newOptions = None):
	# 	# Get
	# 	if newOptions is None:
	# 		node = g.GetNode(self.net.eNet, self.eId)
	# 		nodeDef = g.node_Definition(node)
	# 		stringArr = g.nodeDefinition_GetStringDefinition(nodeDef, self.numberStates())
			
	# 		stateNames = []
	# 		for i in range(self.numberStates()):
	# 			stateNames.append(stringArr[i])

	# 		return stateNames
	# 	# Set
	# 	else:
	# 		nodeDef = self._gNodeDef()
	# 		no = (ctypes.c_char_p*len(newOptions))(*newOptions)
			
	# 		g.nodeDefinition_SetStringDefinition(nodeDef, len(newOptions), no)
	# 		self.net.needsUpdate = True
	
	def parentSubmodel(self, parentSubmodel = None):
		if parentSubmodel is None:
			parentEId = g.node_GetSubmodel(self._gNode())
			if g.submodel_GetParent(self.net.eNet, parentEId) >= 0:
				return Submodel(self.net, eId = parentEId)
			return None
		else:
			submodelName = parentSubmodel if isString(parentSubmodel) else parentSubmodel.name()
			newSubmodelId = g.submodel_FindSubmodel(self.net.eNet, submodelName)
			g.node_SetSubmodel(g.GetNode(self.net.eNet, self.eId), submodelName)
		
		return self
	
	def user(self):
		nodeInfo = self._gNodeInfo()
		userProps = g.nodeInfo_UserProperties(nodeInfo)
		
		return UserProperties(self, userProps)
		
	def setUniform(self):
		numEntries = self.numberStates()*self.net.numberCombinations(self.parents())
		self.cpt1d([float(1)/self.numberStates() for i in range(numEntries)])
		
		# Chain
		return self
	
	def setRandom(self):
		newCpt = []
		for r in range(self.net.numberCombinations(self.parents())):
			randVec = normalize([random.random() for v in range(self.numberStates())])
			newCpt.append(randVec)
		
		self.cpt(newCpt)
		
		# Chain
		return self
	
	# DeMorgan node (QGeNIe) functions
	def dmPriorBelief(self, belief = None):
		dfn = self._gNodeDef()
		if belief is not None:
			g.Demorgan_SetPriorBelief(dfn, belief)
		else:
			val = g.Demorgan_GetPriorBelief(dfn)
			return val
		return self
	
	def dmParentWeights(self, weights = None):
		dfn = self._gNodeDef()
		numParents = len(self.parents())
		if weights is not None:
			# fullWeights = weights + [0]*max(0, numParents-len(weights))
			cWeights = (ctypes.c_double*numParents)(*weights)
			g.Demorgan_SetParentWeights(dfn, numParents, cWeights)
		else:
			cWeightsDblArray = g.Demorgan_GetParentWeights(dfn)
			cWeights = g.doubleArray_Items(cWeightsDblArray)
			numCWeights = g.doubleArray_NumItems(cWeightsDblArray)
			weights = [0]*numCWeights
			for i in range(numParents):
				weights[i] = cWeights[i]
			return weights

		return self

	def dmParentTypes(self, types = None):
		dfn = self._gNodeDef()
		numParents = len(self.parents())
		if types is not None:
			cTypes = (ctypes.c_int*numParents)(*types)
			g.Demorgan_SetParentTypes(dfn, numParents, cTypes)
		else:
			cTypesIntArray = g.Demorgan_GetParentTypes(dfn)
			cTypes = g.intArray_Items(cTypesIntArray)
			numCTypes = g.intArray_NumItems(cTypesIntArray)
			types = [0]*numCTypes
			for i in range(numParents):
				types[i] = cTypes[i]
			return types

		return self

	
class UserProperties(object):
	def __init__(self, node, eId):
		self.node = node
		self.eId = eId
		
	def add(self, name, value):
		g.userProperties_AddProperty(self.eId, name, value)

	def get(self, name):
		index = g.userProperties_FindProperty(self.eId, name)
		if index >= 0:
			return g.userProperties_GetPropertyValue(self.eId, index)
		else:
			return None
	
	def getAll(self):
		num = g.userProperties_GetNumberOfProperties(self.eId)
		props = {}
		for i in range(num):
			name = g.userProperties_GetPropertyName(self.eId, i)
			value = g.userProperties_GetPropertyValue(self.eId, i)
			props[name] = value
		return props
	
	def delete(self, name):
		index = g.userProperties_FindProperty(self.eId, name)
		if index >= 0:
			g.userProperties_DeleteProperty(self.eId, index)
		else:
			raise BNIError('Could not find user property \'{}\''.format(name))
	
	def clear(self):
		g.userProperties_Clear(self.eId)


class State:
	def __init__(self, node = None, stateNum = None):
		self.node = node
		self.stateNum = stateNum
	
	def name(self, _name = None, check = False):
		if _name is None:
			self.node._setupStates()

			return self.node._stateNames[self.stateNum]
		else:
			if check:
				uniqueSet = [s.name() for s in self.node.states()]
				uniqueSet.remove(self.name())
				_name = self.node.net.makeValidName(_name, uniqueSet=uniqueSet)
			self.node.renameState(self.stateNum, _name)
		
			self.node._setupStates(force = True)

	def title(self, _title = None):
		NYI()
	def setTrueFinding(self):
		gNode = self.node._gNode()
		gNodeValue = g.node_Value(gNode)
		g.nodeValue_SetEvidence(gNodeValue, self.stateNum)
		self.node.net.needsUpdate = True

	def belief(self):
		return self.node.beliefs()[self.stateNum]