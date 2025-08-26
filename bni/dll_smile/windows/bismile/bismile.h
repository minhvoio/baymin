#ifndef BISMILE_DLL_H__
#define BISMILE_DLL_H__

/**
Use this to generate random hex key: ','.join([hex(random.randint(0,255)) for i in range(16)])
**/

#define CREATEDLL

#ifdef __cplusplus
extern "C" {
#endif

#ifdef CREATEDLL
#define EXPORT(RET_TYPE,FUNC_NAME) __declspec(dllexport) RET_TYPE __stdcall FUNC_NAME
#define CALL_TYPE __stdcall
#else
#define EXPORT(RET_TYPE,FUNC_NAME) RET_TYPE FUNC_NAME
#define CALL_TYPE 
#endif

// Structures
struct KeyValue {
	const char* key;
	double value;
};

struct KeyValueArray {
	KeyValue* items;
	size_t size;
};

KeyValueArray get_key_value_array(const std::vector<std::pair<std::string, double>>& data) {
	static std::vector<KeyValue> flat_data;
	flat_data.clear();
	for (const auto& pair : data) {
		flat_data.push_back({ pair.first.c_str(), pair.second });
	}

	return { flat_data.data(), flat_data.size() };
}

std::vector<std::pair<std::string, double>> get_from_key_value_array(const KeyValueArray& kvArray) {
	std::vector<std::pair<std::string, double>> result;

	// Iterate through the KeyValue array
	for (size_t i = 0; i < kvArray.size; ++i) {
		const KeyValue& kv = kvArray.items[i];

		// Convert each KeyValue into a std::pair
		result.emplace_back(std::string(kv.key, std::strlen(kv.key)), kv.value);
	}

	return result;
}



/** The actual functions...
These DLL function names try to stick as closely as possible to the SMILE class/method names
(including case), except:
	- 'DSL_' is removed from everything
	- Functions on net objects have no 'network' prefix
Also, 'new_xxx' means 'new xxx'.
*/
/**
FIX: Have to handle memory freeing somehow. (2020-05-03 SM update: Yes, still need to do this, particularly for returned pointers.)
**/

EXPORT(const char*, test)(const char* arg);

EXPORT(void, setLicense)(const char* lic1, unsigned char lic2[64]);

EXPORT(void*, new_intArray)();
EXPORT(int, intArray_NumItems)(void* intArray);
EXPORT(int*, intArray_Items)(void* intArray);
EXPORT(int, doubleArray_NumItems)(void* doubleArray);
EXPORT(double*, doubleArray_Items)(void* doubleArray);
EXPORT(const char**, stringArray_Items)(void* stringArray);


EXPORT(void*, new_network)();
EXPORT(void*, copy_network)(void* net);
EXPORT(void, delete_network)(void* net);
EXPORT(int, AddNode)(void* net, int thisType, const char* nodeName);
EXPORT(int, DeleteNode)(void* net, int thisNode);
EXPORT(int, AddArc)(void* net, int theParent, int theChild);
EXPORT(int, RemoveArc)(void* net, int theParent, int theChild);
EXPORT(int, FindNode)(void* net, const char* withThisID);
EXPORT(int, HasNode)(void* net, const char* withThisID);
EXPORT(void*, GetNode)(void* net, int theNode); /// GetNode not very useful. Use handle instead.
EXPORT(int, GetAllNodes)(void* net, void* here);
EXPORT(void*, GetParents)(void* net, int theNode);
EXPORT(void, UpdateBeliefs)(void* net);
EXPORT(int, ClearAllEvidence)(void* net);
EXPORT(double, CalcProbEvidence)(void* net);
EXPORT(void, WriteToFile)(void* net, const char* fileName);
EXPORT(void, ReadFromFile)(void* net, const char* fileName);
EXPORT(void, SetDefaultBNAlgorithm)(void* net, int theAlgorithm);
EXPORT(int, GetDefaultBNAlgorithm)(void* net);
EXPORT(int, GetNumberOfSamples)(void* net);
EXPORT(int, SetNumberOfSamples)(void* net, int aNumber);
EXPORT(int, GetNumberOfDiscretizationSamples)(void* net);
EXPORT(void, SetNumberOfDiscretizationSamples)(void* net, int aNumber);


EXPORT(void*, net_Header)(void* net);

EXPORT(void*, node_Info)(void* node);
EXPORT(void*, node_Value)(void* node);
EXPORT(int, node_ChangeType)(void* node, int newType);
EXPORT(void*, nodeInfo_Header)(void* nodeInfo);
EXPORT(void*, nodeInfo_Screen)(void* nodeInfo);
EXPORT(void*, nodeInfo_UserProperties)(void* nodeInfo);

EXPORT(void*, nodeValue_GetMatrix)(void* nodeValue);
EXPORT(int, nodeValue_GetType)(void* nodeValue);
EXPORT(int, nodeValue_IsValueValid)(void* nodeValue);
EXPORT(int, nodeValue_IsRealEvidence)(void* nodeValue);
EXPORT(int, nodeValue_GetEvidence)(void* nodeValue);
EXPORT(int, nodeValue_SetEvidence)(void* nodeValue, int theEvidence);
EXPORT(double*, nodeValue_GetVirtualEvidence)(void* nodeValue);
EXPORT(int, nodeValue_SetVirtualEvidence)(void* nodeValue, int size, double* theEvidence);
EXPORT(int, nodeValue_ClearEvidence)(void* nodeValue);
EXPORT(void*, nodeValue_GetIndexingParents)(void* nodeValue);

EXPORT(void*, node_Definition)(void* node);
EXPORT(int, nodeDefinition_GetType)(void* nodeDefinition);
EXPORT(void*, nodeDefinition_GetMatrix)(void* nodeDefinition);
EXPORT(void*, nodeDefinition_GetDoubleDefinition)(void* nodeDefinition, int size);
//EXPORT(void*, nodeDefinition_GetStringDefinition)(void* nodeDefinition, int size);
EXPORT(void, nodeDefinition_SetDoubleDefinition)(void* nodeDefinition, int size, double* dArr);
//EXPORT(void, nodeDefinition_SetStringDefinition)(void* nodeDefinition, int size, const char** sArr);
EXPORT(int, nodeDefinition_GetNumberOfOutcomes)(void* nodeDefinition);
EXPORT(void*, nodeDefinition_GetOutcomesNames)(void* nodeDefinition);
EXPORT(int, nodeDefinition_AddOutcome)(void* nodeDefinition, const char* name);
EXPORT(int, nodeDefinition_SetNumberOfOutcomes)(void* nodeDefinition, int aNumber);
EXPORT(int, nodeDefinition_SetNumberOfOutcomesStr)(void* nodeDefinition, int size, const char** theOutcomeNames);
EXPORT(int, nodeDefinition_RenameOutcomes)(void* nodeDefinition, int size, const char** names);
EXPORT(int, nodeDefinition_RemoveOutcome)(void* nodeDefinition, int outcomeNumber);
EXPORT(int, nodeDefinition_ChangeOrderOfOutcomes)(void* nodeDefinition, int size, int* newOrder);
EXPORT(int, nodeDefinition_ChangeOrderOfOutcomesWithAddAndRemove)(void* nodeDefinition, int size1, const char** ids, int size2, int* newOrder);

EXPORT(int, equation_SetEquation)(void* equationNode, const char* eqText);
EXPORT(KeyValueArray, equation_GetDiscreteIntervals)(void* equationNode);
EXPORT(int, equation_SetDiscreteIntervals)(void* equationNode, KeyValueArray intervals);
EXPORT(int, equation_ClearDiscreteIntervals)(void* equationNode);
EXPORT(double*, equation_GetBounds)(void* equationNode);
EXPORT(int, equation_SetBounds)(void* equationNode, double low, double high);
EXPORT(int, mau_SetExpression)(void* mauDefinition, const char* expression);
// Mean is for discretized/sample, SampleMean is just for sample
EXPORT(double, valEqEvaluation_GetMean)(void* nodeValue);
EXPORT(double, valEqEvaluation_GetStdDev)(void* nodeValue);
EXPORT(double, valEqEvaluation_GetSampleMean)(void* nodeValue);
EXPORT(double, valEqEvaluation_GetSampleStdDev)(void* nodeValue);
EXPORT(double, valEqEvaluation_GetSample)(void* nodeValue, int index);
EXPORT(int, valEqEvaluation_GetNumberOfSamples)(void* nodeValue);
EXPORT(const double*, valEqEvaluation_GetSamples)(void* nodeValue);
EXPORT(void, valEqEvaluation_SetEvidence)(void* nodeValue, double val);
EXPORT(double, valEqEvaluation_GetEvidence)(void* nodeValue);

EXPORT(int, dMatrix_GetSize)(void* dMatrix);
EXPORT(double*, dMatrix_GetItemsDouble)(void* dMatrix);

EXPORT(const char*, header_GetId)(void* header);
EXPORT(void, header_SetId)(void* header, const char* id);
EXPORT(const char*, header_GetName)(void* header);
EXPORT(void, header_SetName)(void* header, const char* name);
EXPORT(const char*, header_GetComment)(void* header);
EXPORT(void, header_SetComment)(void* header, const char* name);

EXPORT(void*, screenInfo_position)(void* screenInfo);
/// Returns array of 6 ints [color, selColor, font, fontColor, borderThickness, borderColor]
EXPORT(int*, screenInfo_formatting)(void* screenInfo);
EXPORT(void, screenInfo_setFormatting)(void* screenInfo, int* fmts);

EXPORT(int, rectangle_center_X)(void* rectangle);
EXPORT(int, rectangle_center_Y)(void* rectangle);
EXPORT(void, rectangle_center_X_set)(void* rectangle, int val);
EXPORT(void, rectangle_center_Y_set)(void* rectangle, int val);

EXPORT(int, rectangle_width)(void* rectangle);
EXPORT(int, rectangle_height)(void* rectangle);
EXPORT(void, rectangle_width_set)(void* rectangle, int val);
EXPORT(void, rectangle_height_set)(void* rectangle, int val);

EXPORT(int, userProperties_AddProperty)(void* userProperties, const char *propertyName, const char *propertyValue);
EXPORT(int, userProperties_FindProperty)(void* userProperties, const char *withThisName);
EXPORT(int, userProperties_DeleteProperty)(void* userProperties, int thisOne);
EXPORT(int, userProperties_GetNumberOfProperties)(void* userProperties);
EXPORT(const char*, userProperties_GetPropertyName)(void* userProperties, int index);
EXPORT(const char*, userProperties_GetPropertyValue)(void* userProperties, int index);
EXPORT(void, userProperties_Clear)(void* userProperties);

/** Submodels **/
EXPORT(int, submodel_CreateSubmodel)(void* net, int theParent, const char *thisId);
EXPORT(int, submodel_DeleteSubmodel)(void* net, int handler);
EXPORT(int, submodel_FindSubmodel)(void* net, const char *withThisID);
EXPORT(int, submodel_GetParent)(void* net, int ofThisSubmodel);
EXPORT(int, submodel_MoveSubmodel)(void* net, int thisSubmodel, int intoThisSubmodel);
EXPORT(void*, submodel_GetSubmodel)(void* net, int handler);
EXPORT(void*, submodel_Header)(void* submodel);
EXPORT(void*, submodel_ScreenInfo)(void* submodel);
EXPORT(int, node_GetSubmodel)(void* node);
//EXPORT(int, node_SetSubmodel)(void* node, int thisSubmodel);
EXPORT(int, node_SetSubmodel)(void* node, char *thisSubmodel);
EXPORT(void*, submodel_GetIncludedSubmodels)(void* net, int inThisSubmodel);
EXPORT(void*, submodel_GetIncludedNodes)(void* net, int inThisSubmodel);
/// I thought "deep" meant recursive, but it doesn't seem to do that, and I'm not actually sure what it does
EXPORT(void*, submodel_GetDeepIncludedSubmodels)(void* net, int inThisSubmodel);
EXPORT(void*, submodel_GetDeepIncludedNodes)(void* net, int inThisSubmodel);

/** Learning **/
EXPORT(int, learn_em)(void* net, const char* fileName, float ess, bool relevance, float learningRate, int initialLearningPoint);
EXPORT(int, learn_emSimple)(void* net, const char* fileName, float ess);
EXPORT(int*, learn_pc)(void* net, const char* fileName, int maxAdjacency, int maxSearchTime, double significance, int tiersSize, int* tiers);
EXPORT(int, learn_bs)(void* net, const char* fileName, int numIterations, int maxParents);
EXPORT(int, learn_tan)(void* net, const char* fileName, const char* classvar, int maxSearchTime);

/** QGeNIe/DeMorgan **/
EXPORT(int, Demorgan_SetParentWeights)(void* obj, int size, const double* dArr);
EXPORT(int, Demorgan_SetParentWeight)(void* obj, int numParent, double value);
EXPORT(double, Demorgan_GetParentWeight)(void* obj, int numParent);
EXPORT(void*, Demorgan_GetParentWeights)(void* obj);

EXPORT(int, Demorgan_SetParentTypes)(void* obj, int size, const int* iArr);
EXPORT(int, Demorgan_SetParentType)(void* obj, int numParent, int value);
EXPORT(int, Demorgan_GetParentType)(void* obj, int numParent);
EXPORT(void*, Demorgan_GetParentTypes)(void* obj);

EXPORT(int, Demorgan_GetTemporalParentType)(void* obj, int order, int numParent);
EXPORT(int, Demorgan_SetTemporalParentType)(void* obj, int order, int numParent, int parentType);
EXPORT(double, Demorgan_GetTemporalParentWeight)(void* obj, int order, int numParent);
EXPORT(int, Demorgan_SetTemporalParentWeight)(void* obj, int order, int numParent, double weight);

EXPORT(double, Demorgan_GetPriorBelief)(void* obj);
EXPORT(int, Demorgan_SetPriorBelief)(void* obj, double value);


#ifdef __cplusplus
}
#endif

#endif