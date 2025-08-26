#include "pch.h"
#include "smile/smile.h"
#include "smile_license.h"
#include <iostream>
#include "bismile.h"

using namespace std;

class strarray : public DSL_stringArray {
public:
	strarray(int numitems, char **items) : DSL_stringArray(numitems*2) {
		//this->UseAsList(numitems);
		for (int i = 0; i < numitems; i++) {
			this->SetString(i, items[i]);
		}
	}
};

void CALL_TYPE setLicense(const char* lic1, unsigned char lic2[64]) {
	DSL_LIC1 = lic1;
	memcpy(DSL_LIC2, lic2, sizeof(unsigned char)*64);
}


void* CALL_TYPE new_intArray() {
	DSL_intArray* arr = new DSL_intArray();
	return (void*)arr;
}

void* CALL_TYPE new_network() {
	DSL_network* net = new DSL_network();
	return (void*)net;
}

void* CALL_TYPE copy_network(void* net) {
	DSL_network* newNet = new DSL_network();
	newNet->Copy(*((DSL_network*)net));
	return newNet;
}

void CALL_TYPE delete_network(void* net) {
	delete (DSL_network*)net;
}

int CALL_TYPE AddNode(void* net, int thisType, const char* nodeName) {
	((DSL_network*)net)->AddNode(thisType, nodeName);
	return FindNode(net, nodeName);
}

int CALL_TYPE DeleteNode(void* net, int thisNode) {
	return ((DSL_network*)net)->DeleteNode(thisNode);
}

int CALL_TYPE AddArc(void* net, int theParent, int theChild) {
	return ((DSL_network*)net)->AddArc(theParent, theChild);
}

int CALL_TYPE RemoveArc(void* net, int theParent, int theChild) {
	return ((DSL_network*)net)->RemoveArc(theParent, theChild);
}

int CALL_TYPE FindNode(void* net, const char* withThisID) {
	return ((DSL_network*)net)->FindNode(withThisID);
}

int CALL_TYPE HasNode(void* net, const char* withThisID) {
	return ((DSL_network*)net)->FindNode(withThisID) != -2;
}

void* CALL_TYPE GetNode(void* net, int theNode) {
	return ((DSL_network*)net)->GetNode(theNode);
}

int CALL_TYPE GetAllNodes(void* net, void* here) {
	return ((DSL_network*)net)->GetAllNodes(*((DSL_intArray*)here));
}

void* CALL_TYPE GetParents(void* net, int theNode) {
	return (void*)&(((DSL_network*)net)->GetParents(theNode));
}

void CALL_TYPE UpdateBeliefs(void* net) {
	((DSL_network*)net)->UpdateBeliefs();
}

int CALL_TYPE ClearAllEvidence(void* net) {
	return ((DSL_network*)net)->ClearAllEvidence();
}

double CALL_TYPE CalcProbEvidence(void* net) {
	double pe;
	bool ret = ((DSL_network*)net)->CalcProbEvidence(pe);

	if (ret) {
		return pe;
	}

	/// Docs aren't very clear on when 'CalcProbEvidence' returns false
	return 0;
}

int CALL_TYPE GetNumberOfSamples(void* net) {
	return ((DSL_network*)net)->GetNumberOfSamples();
}

int CALL_TYPE SetNumberOfSamples(void* net, int aNumber) {
	return ((DSL_network*)net)->SetNumberOfSamples(aNumber);
}

int CALL_TYPE GetNumberOfDiscretizationSamples(void* net) {
	return ((DSL_network*)net)->GetNumberOfDiscretizationSamples();
}

void CALL_TYPE SetNumberOfDiscretizationSamples(void* net, int aNumber) {
	((DSL_network*)net)->SetNumberOfDiscretizationSamples(aNumber);
}

void* CALL_TYPE net_Header(void* net) {
	return (void*)&((DSL_network*)net)->Header();
}

void* CALL_TYPE node_Info(void* node) {
	return (void*)&((DSL_node*)node)->Info();
}

void* CALL_TYPE node_Value(void* node) {
	return (void*)((DSL_node*)node)->Value();
}

int CALL_TYPE node_ChangeType(void* node, int newType) {
	return ((DSL_node*)node)->ChangeType(newType);
}

void* CALL_TYPE nodeValue_GetMatrix(void* nodeValue) {
	const DSL_Dmatrix* dMatrix = ((DSL_nodeValue*)nodeValue)->GetMatrix();
	return (void*)dMatrix;
}

int CALL_TYPE nodeValue_GetType(void* nodeValue) {
	return ((DSL_nodeValue*)nodeValue)->GetType();
}

int CALL_TYPE nodeValue_IsValueValid(void* nodeValue) {
	return ((DSL_nodeValue*)nodeValue)->IsValueValid();
}

int CALL_TYPE nodeValue_IsRealEvidence(void* nodeValue) {
	return ((DSL_nodeValue*)nodeValue)->IsRealEvidence();
}

int CALL_TYPE nodeValue_GetEvidence(void* nodeValue) {
	return ((DSL_nodeValue*)nodeValue)->GetEvidence();
}

int CALL_TYPE nodeValue_SetEvidence(void* nodeValue, int theEvidence) {
	return ((DSL_nodeValue*)nodeValue)->SetEvidence(theEvidence);
}

double* CALL_TYPE nodeValue_GetVirtualEvidence(void* nodeValue) {
	std::vector<double> vecEv;

	((DSL_nodeValue*)nodeValue)->GetVirtualEvidence(vecEv);
	double* ret;

	ret = new double[vecEv.size() + 1];

	ret[0] = (double)vecEv.size();

	for (unsigned i = 0; i<vecEv.size(); i++) {
		ret[i + 1] = vecEv[i];
	}

	/// First entry is size of array (not including first element).
	/// Yeah, I know, kind of crappy!
	return ret;
}

int CALL_TYPE nodeValue_SetVirtualEvidence(void* nodeValue, int size, double* theEvidence) {
	std::vector<double> vecEv;

	for (int i = 0; i<size; i++) {
		vecEv.push_back(theEvidence[i]);
	}

	return ((DSL_nodeValue*)nodeValue)->SetVirtualEvidence(vecEv);
}

int CALL_TYPE nodeValue_ClearEvidence(void* nodeValue) {
	return ((DSL_nodeValue*)nodeValue)->ClearEvidence();
}

// XXX-LEAK
void* CALL_TYPE nodeValue_GetIndexingParents(void* nodeValue) {
	DSL_intArray* pars = new DSL_intArray(((DSL_nodeValue*)nodeValue)->GetIndexingParents());
	return pars;
}

void* CALL_TYPE node_Definition(void* node) {
	return (void*)((DSL_node*)node)->Definition();
}

int CALL_TYPE nodeDefinition_GetType(void* nodeDefinition) {
	return ((DSL_nodeDefinition*)nodeDefinition)->GetType();
}

void* CALL_TYPE nodeDefinition_GetMatrix(void* nodeDefinition) {
	return (void*)((DSL_nodeDefinition*)nodeDefinition)->GetMatrix();
}

/// This returns the raw double pointer
void* CALL_TYPE nodeDefinition_GetDoubleDefinition(void* nodeDefinition, int size) {
	const DSL_Dmatrix* da = 0; //new DSL_doubleArray(size);
	((DSL_nodeDefinition*)nodeDefinition)->GetDefinition(&da);
	return (void*)&(da->GetItems());
}

///// This returns the raw char* pointer
//void* CALL_TYPE nodeDefinition_GetStringDefinition(void* nodeDefinition, int size) {
//	DSL_stringArray* sa = 0; //new DSL_stringArray(size);
//	((DSL_nodeDefinition*)nodeDefinition)->GetDefinition(&sa);
//	return (void*)sa->Items();
//}

void CALL_TYPE nodeDefinition_SetDoubleDefinition(void* nodeDefinition, int size, double* dArr) {
	DSL_doubleArray da(size);
	for (int i = 0; i<size; i++) {
		da[i] = dArr[i];
	}
	DSL_nodeDefinition* nodeDef = ((DSL_nodeDefinition*)nodeDefinition);
	nodeDef->SetDefinition(da);
}

//void CALL_TYPE nodeDefinition_SetStringDefinition(void* nodeDefinition, int size, const char** sArr) {
//	strarray sa(size, const_cast<char**>(sArr));
//	/*for (int i = 0; i<size; i++) {
//		sa.Add(sArr[i]);
//	}*/
//	((DSL_nodeDefinition*)nodeDefinition)->SetDefinition(sa);
//}

int CALL_TYPE nodeDefinition_GetNumberOfOutcomes(void* nodeDefinition) {
	return ((DSL_nodeDefinition*)nodeDefinition)->GetNumberOfOutcomes();
}

void* CALL_TYPE nodeDefinition_GetOutcomesNames(void* nodeDefinition) {
	return (void*)((DSL_nodeDefinition*)nodeDefinition)->GetOutcomesNames();
}

int CALL_TYPE nodeDefinition_AddOutcome(void* nodeDefinition, const char* name) {
	return ((DSL_nodeDefinition*)nodeDefinition)->AddOutcome(name);
}

int CALL_TYPE nodeDefinition_SetNumberOfOutcomes(void* nodeDefinition, int aNumber) {
	DSL_nodeDefinition* def = ((DSL_nodeDefinition*)nodeDefinition);
	return def->SetNumberOfOutcomes(aNumber);
}

int CALL_TYPE nodeDefinition_SetNumberOfOutcomesStr(void* nodeDefinition, int size, const char** theOutcomeNames) {
	DSL_nodeDefinition* def = ((DSL_nodeDefinition*)nodeDefinition);
	DSL_idArray namesArray;
	for (int i = 0; i<size; i++) {
		namesArray.Add(theOutcomeNames[i]);
	}
	return def->SetNumberOfOutcomes(namesArray);
}


int CALL_TYPE nodeDefinition_RenameOutcomes(void* nodeDefinition, int size, const char** names) {
	DSL_idArray namesArray;
	for (int i = 0; i<size; i++) {
		namesArray.Add(names[i]);
	}
	//strarray namesArray(size, const_cast<char**>(names));
	return ((DSL_nodeDefinition*)nodeDefinition)->RenameOutcomes(namesArray);
}

int CALL_TYPE nodeDefinition_RemoveOutcome(void* nodeDefinition, int outcomeNumber) {
	DSL_nodeDefinition* def = ((DSL_nodeDefinition*)nodeDefinition);
	return def->RemoveOutcome(outcomeNumber);
}

int CALL_TYPE nodeDefinition_ChangeOrderOfOutcomes(void* nodeDefinition, int size, int* newOrder) {
	DSL_nodeDefinition* def = ((DSL_nodeDefinition*)nodeDefinition);
	DSL_intArray iaNewOrder(size);

	for (int i = 0; i < size; i++) {
		iaNewOrder.Add(newOrder[i]);
	}
	return def->ChangeOrderOfOutcomes(iaNewOrder);
}

int CALL_TYPE nodeDefinition_ChangeOrderOfOutcomesWithAddAndRemove(void* nodeDefinition, int size1, const char** ids, int size2, int* newOrder) {
	DSL_nodeDefinition* def = ((DSL_nodeDefinition*)nodeDefinition);
	DSL_idArray _ids(size1);
	DSL_intArray _newOrder(size2);

	for (int i = 0; i < size1; i++) {
		_ids.Add(ids[i]);
	}
	for (int i = 0; i < size2; i++) {
		_newOrder.Add(newOrder[i]);
	}

	return def->ChangeOrderOfOutcomesWithAddAndRemove(_ids, _newOrder);
}


int CALL_TYPE equation_SetEquation(void* equationNode, const char* eqText) {
	return ((DSL_equation*)equationNode)->SetEquation(eqText);
}

KeyValueArray CALL_TYPE equation_GetDiscreteIntervals(void* equationNode) {
	DSL_equation* eq = (DSL_equation*)equationNode;
	const DSL_equation::IntervalVector& iv = eq->GetDiscIntervals();
	return get_key_value_array(iv);
}

int CALL_TYPE equation_SetDiscreteIntervals(void* equationNode, KeyValueArray intervals) {
	DSL_equation* eq = (DSL_equation*)equationNode;
	return eq->SetDiscIntervals(get_from_key_value_array(intervals));
}

int CALL_TYPE equation_ClearDiscreteIntervals(void* equationNode) {
	DSL_equation* eq = (DSL_equation*)equationNode;
	return eq->ClearDiscIntervals();
}

double* CALL_TYPE equation_GetBounds(void* equationNode) {
	DSL_equation* eq = (DSL_equation*)equationNode;
	double low;
	double high;
	eq->GetBounds(low, high);
	return new double[2]{ low, high };
}

int CALL_TYPE equation_SetBounds(void* equationNode, double low, double high) {
	DSL_equation* eq = (DSL_equation*)equationNode;
	return eq->SetBounds(low, high);
}


int CALL_TYPE mau_SetExpression(void* mauDefinition, const char* expression) {
	DSL_mau* mauDef = ((DSL_mau*)mauDefinition);
	vector<std::string> expressions;
	expressions.push_back(expression);
	return mauDef->SetExpressions(expressions);
}

double CALL_TYPE valEqEvaluation_GetSampleMean(void* nodeValue) {
	return ((DSL_valEqEvaluation*)nodeValue)->GetSampleMean();
}

double CALL_TYPE valEqEvaluation_GetSampleStdDev(void* nodeValue) {
	return ((DSL_valEqEvaluation*)nodeValue)->GetSampleStdDev();
}

double CALL_TYPE valEqEvaluation_GetMean(void* nodeValue) {
	double val;
	((DSL_valEqEvaluation*)nodeValue)->GetMean(val);
	return val;
}

double CALL_TYPE valEqEvaluation_GetStdDev(void* nodeValue) {
	double val;
	((DSL_valEqEvaluation*)nodeValue)->GetStdDev(val);
	return val;
}

double CALL_TYPE valEqEvaluation_GetSample(void* nodeValue, int index) {
	return ((DSL_valEqEvaluation*)nodeValue)->GetSample(index);
}

int CALL_TYPE valEqEvaluation_GetNumberOfSamples(void* nodeValue) {
	return ((DSL_valEqEvaluation*)nodeValue)->GetSamples().size();
}

const double* CALL_TYPE valEqEvaluation_GetSamples(void* nodeValue) {
	return ((DSL_valEqEvaluation*)nodeValue)->GetSamples().data();
}

void CALL_TYPE valEqEvaluation_SetEvidence(void* nodeValue, double val) {
	((DSL_valEqEvaluation*)nodeValue)->SetEvidence(val);
}

double CALL_TYPE valEqEvaluation_GetEvidence(void* nodeValue) {
	double val;
	((DSL_valEqEvaluation*)nodeValue)->GetEvidence(val);
	return val;
}

const char** CALL_TYPE stringArray_Items(void* stringArray) {
	DSL_stringArray& strArray = *((DSL_stringArray*)stringArray);
	int numItems = strArray.GetSize();
	const char** strs = new const char* [numItems];
	for (int i = 0; i < numItems; i++) {
		strs[i] = strArray[i];
	}
	return (const char**)strs;
}

double* CALL_TYPE dMatrix_GetItemsDouble(void* dMatrix) {
	return (double*)((DSL_Dmatrix*)dMatrix)->GetItems().Items();
}

int CALL_TYPE dMatrix_GetSize(void* dMatrix) {
	return ((DSL_Dmatrix*)dMatrix)->GetItems().GetSize();
}

void* CALL_TYPE nodeInfo_Header(void* nodeInfo) {
	return (void*)&((DSL_nodeInfo*)nodeInfo)->Header();
}

void* CALL_TYPE nodeInfo_Screen(void* nodeInfo) {
	return (void*)&((DSL_nodeInfo*)nodeInfo)->Screen();
}

void* CALL_TYPE nodeInfo_UserProperties(void* nodeInfo) {
	return (void*)&((DSL_nodeInfo*)nodeInfo)->UserProperties();
}

void* CALL_TYPE screenInfo_position(void* screenInfo) {
	return (void*)&((DSL_screenInfo*)screenInfo)->position;
}

int* CALL_TYPE screenInfo_formatting(void* screenInfo) {
	DSL_screenInfo* cScreenInfo = (DSL_screenInfo*)screenInfo;
	int* fmts = new int[6];
	fmts[0] = cScreenInfo->color;
	fmts[1] = cScreenInfo->selColor;
	fmts[2] = cScreenInfo->font;
	fmts[3] = cScreenInfo->fontColor;
	fmts[4] = cScreenInfo->borderThickness;
	fmts[5] = cScreenInfo->borderColor;

	return fmts;
}

void CALL_TYPE screenInfo_setFormatting(void* screenInfo, int* fmts) {
	DSL_screenInfo* cScreenInfo = (DSL_screenInfo*)screenInfo;
	if (fmts[0] != -1)  cScreenInfo->color = fmts[0];
	if (fmts[1] != -1)  cScreenInfo->selColor = fmts[1];
	if (fmts[2] != -1)  cScreenInfo->font = fmts[2];
	if (fmts[3] != -1)  cScreenInfo->fontColor = fmts[3];
	if (fmts[4] != -1)  cScreenInfo->borderThickness = fmts[4];
	if (fmts[5] != -1)  cScreenInfo->borderColor = fmts[5];
}

int CALL_TYPE rectangle_center_X(void* rectangle) {
	return ((DSL_rectangle*)rectangle)->center_X;
}

int CALL_TYPE rectangle_center_Y(void* rectangle) {
	return ((DSL_rectangle*)rectangle)->center_Y;
}

void CALL_TYPE rectangle_center_X_set(void* rectangle, int val) {
	((DSL_rectangle*)rectangle)->center_X = val;
}

void CALL_TYPE rectangle_center_Y_set(void* rectangle, int val) {
	((DSL_rectangle*)rectangle)->center_Y = val;
}

int CALL_TYPE rectangle_width(void* rectangle) {
	return ((DSL_rectangle*)rectangle)->width;
}

int CALL_TYPE rectangle_height(void* rectangle) {
	return ((DSL_rectangle*)rectangle)->height;
}

void CALL_TYPE rectangle_width_set(void* rectangle, int val) {
	((DSL_rectangle*)rectangle)->width = val;
}

void CALL_TYPE rectangle_height_set(void* rectangle, int val) {
	((DSL_rectangle*)rectangle)->height = val;
}

const char* CALL_TYPE header_GetId(void* header) {
	return ((DSL_header*)header)->GetId();
}

void CALL_TYPE header_SetId(void* header, const char* id) {
	((DSL_header*)header)->SetId(id);
}

const char* CALL_TYPE header_GetName(void* header) {
	return ((DSL_header*)header)->GetName();
}

void CALL_TYPE header_SetName(void* header, const char* name) {
	((DSL_header*)header)->SetName(name);
}

const char* CALL_TYPE header_GetComment(void* header) {
	return ((DSL_header*)header)->GetComment();
}

void CALL_TYPE header_SetComment(void* header, const char* name) {
	((DSL_header*)header)->SetComment(name);
}

void CALL_TYPE ReadFromFile(void * net, const char* fileName) {
	((DSL_network*)net)->ReadFile(fileName);
}

void CALL_TYPE WriteToFile(void* net, const char* fileName) {
	((DSL_network*)net)->WriteFile(fileName);
}

void CALL_TYPE SetDefaultBNAlgorithm(void* net, int theAlgorithm) {
	((DSL_network*)net)->SetDefaultBNAlgorithm(theAlgorithm);
}

int CALL_TYPE GetDefaultBNAlgorithm(void* net) {
	return ((DSL_network*)net)->GetDefaultBNAlgorithm();
}

int CALL_TYPE intArray_NumItems(void* intArray) {
	return ((DSL_intArray*)intArray)->NumItems();
}

int* CALL_TYPE intArray_Items(void* intArray) {
	return ((DSL_intArray*)intArray)->Items();
}

int CALL_TYPE doubleArray_NumItems(void* doubleArray) {
	return ((DSL_doubleArray*)doubleArray)->NumItems();
}

double* CALL_TYPE doubleArray_Items(void* doubleArray) {
	return ((DSL_doubleArray*)doubleArray)->Items();
}

int CALL_TYPE userProperties_AddProperty(void* userProperties, const char *propertyName, const char *propertyValue) {
	return ((DSL_userProperties*)userProperties)->AddProperty(propertyName, propertyValue);
}

int CALL_TYPE userProperties_FindProperty(void* userProperties, const char *withThisName) {
	return ((DSL_userProperties*)userProperties)->FindProperty(withThisName);
}

int CALL_TYPE userProperties_DeleteProperty(void* userProperties, int thisOne) {
	return ((DSL_userProperties*)userProperties)->DeleteProperty(thisOne);
}

int CALL_TYPE userProperties_GetNumberOfProperties(void* userProperties) {
	return ((DSL_userProperties*)userProperties)->GetNumberOfProperties();
}

const char* CALL_TYPE userProperties_GetPropertyName(void* userProperties, int index) {
	return ((DSL_userProperties*)userProperties)->GetPropertyName(index);
}

const char* CALL_TYPE userProperties_GetPropertyValue(void* userProperties, int index) {
	return ((DSL_userProperties*)userProperties)->GetPropertyValue(index);
}

void CALL_TYPE userProperties_Clear(void* userProperties) {
	((DSL_userProperties*)userProperties)->Clear();
}


/** Submodels **/
int CALL_TYPE submodel_CreateSubmodel(void* net, int theParent, const char *thisId) {
	return ((DSL_network*)net)->GetSubmodelHandler().CreateSubmodel(theParent, thisId);
}

int CALL_TYPE submodel_DeleteSubmodel(void* net, int handler) {
	return ((DSL_network*)net)->GetSubmodelHandler().DeleteSubmodel(handler);
}

int CALL_TYPE submodel_FindSubmodel(void* net, const char *withThisID) {
	return ((DSL_network*)net)->GetSubmodelHandler().FindSubmodel(withThisID);
}

int CALL_TYPE submodel_GetParent(void* net, int ofThisSubmodel) {
	return ((DSL_network*)net)->GetSubmodelHandler().GetParent(ofThisSubmodel);
}

int CALL_TYPE submodel_MoveSubmodel(void* net, int thisSubmodel, int intoThisSubmodel) {
	return ((DSL_network*)net)->GetSubmodelHandler().MoveSubmodel(thisSubmodel, intoThisSubmodel);
}


void* CALL_TYPE submodel_GetSubmodel(void* net, int handler) {
	return ((DSL_network*)net)->GetSubmodelHandler().GetSubmodel(handler);
}

void* CALL_TYPE submodel_Header(void* submodel) { return &((DSL_submodel*)submodel)->header; }
void* CALL_TYPE submodel_ScreenInfo(void* submodel) { return &((DSL_submodel*)submodel)->info; }

int CALL_TYPE node_GetSubmodel(void* node) {
	return ((DSL_node*)node)->GetSubmodel();
}
//int CALL_TYPE node_SetSubmodel(void* node, int thisSubmodel) { return ((DSL_node*)node)->SetSubmodel(thisSubmodel); }
int CALL_TYPE node_SetSubmodel(void* node, char *thisSubmodel) { return ((DSL_node*)node)->SetSubmodel(thisSubmodel); }

/// XXX-LEAK
void* CALL_TYPE submodel_GetIncludedSubmodels(void* net, int inThisSubmodel) {
	DSL_intArray* here = new DSL_intArray;
	((DSL_network*)net)->GetSubmodelHandler().GetIncludedSubmodels(inThisSubmodel, *here);
	return (void*)here;
}

/// XXX-LEAK
void* CALL_TYPE submodel_GetIncludedNodes(void* net, int inThisSubmodel) {
	DSL_intArray* here = new DSL_intArray;
	((DSL_network*)net)->GetSubmodelHandler().GetIncludedNodes(inThisSubmodel, *here);
	return (void*)here;
}

/// XXX-LEAK
void* CALL_TYPE submodel_GetDeepIncludedSubmodels(void* net, int inThisSubmodel) {
	DSL_intArray* here = new DSL_intArray;
	((DSL_network*)net)->GetSubmodelHandler().GetDeepIncludedSubmodels(inThisSubmodel, *here);
	return (void*)here;
}

/// XXX-LEAK
void* CALL_TYPE submodel_GetDeepIncludedNodes(void* net, int inThisSubmodel) {
	DSL_intArray* here = new DSL_intArray;
	((DSL_network*)net)->GetSubmodelHandler().GetDeepIncludedNodes(inThisSubmodel, *here);
	return (void*)here;
}

int CALL_TYPE learn_em(void* net, const char* fileName, float ess, bool relevance, float learningRate, int initialLearningPoint) {
	DSL_em em;
	DSL_dataset ds;
	vector<DSL_datasetMatch> matching;
	string errMsg;

	em.SetEquivalentSampleSize(ess);
	em.SetRelevance(relevance);
	em.SetLearningRate(learningRate);
	em.SetInitialLearningPoint(initialLearningPoint);

	ds.ReadFile(fileName);
	ds.MatchNetwork(*(DSL_network*)net, matching, errMsg);
	//cout << errMsg << endl;
	return em.Learn(ds, *(DSL_network*)net, matching);
}

int CALL_TYPE learn_emSimple(void* net, const char* fileName, float ess) {
	return learn_em(net, fileName, ess, (bool)0, 0.6f, 1);
}

/// Tiers is a pairwise list of node handles and tier indexes. e.g. [<size_of_list>,<node_handle1>,<tier1>,<node_handle2>,<tier1>,...]
int* CALL_TYPE learn_pc(void* net, const char* fileName, int maxAdjacency, int maxSearchTime, double significance,
		int tiersSize, int* tiers) {
	DSL_pc pc;
	pc.maxAdjacency = maxAdjacency;
	pc.maxSearchTime = maxSearchTime;
	pc.significance = significance;
	DSL_bkgndKnowledge& priors = pc.bkk;
	if (tiers && tiersSize) {
		for (int i = 0; i < tiersSize; i += 2) {
			priors.tiers.push_back(pair<int, int>(tiers[i], tiers[i + 1]));
		}
	}

	DSL_dataset ds;
	ds.ReadFile(fileName);

	DSL_pattern pat;
	int res = pc.Learn(ds, pat);

	/// Return the adjacency matrix as learned for the pattern (which may have
	/// undirected arcs). First element is size.
	int* matrix = new int[pat.GetSize()*pat.GetSize() + 1];
	matrix[0] = pat.GetSize()*pat.GetSize();
	for (int i = 0; i < pat.GetSize(); i++) {
		for (int j = 0; j < pat.GetSize(); j++) {
			if (i != j) {
				matrix[i*pat.GetSize() + j + 1] = pat.GetEdge(i, j);
			}
			else {
				matrix[i*pat.GetSize() + j + 1] = -1;
			}
		}
	}

	//cout << "DAG:" << pat.IsDAG() << endl;
	pat.ToNetwork(ds, *(DSL_network*)net);
	//cout << "DAG:" << pat.IsDAG() << endl;

	return matrix;
}

int CALL_TYPE learn_bs(void* net, const char* fileName, int numIterations, int maxParents) {
	DSL_bs bayesSearch;
	bayesSearch.nrIteration = numIterations;
	bayesSearch.maxParents = maxParents;

	DSL_dataset ds;
	ds.ReadFile(fileName);

	int res = bayesSearch.Learn(ds, *(DSL_network*)net);
	return res;
}

int CALL_TYPE learn_tan(void* net, const char* fileName, const char* classvar, int maxSearchTime) {
	DSL_tan tan;
	tan.classvar = classvar;
	tan.maxSearchTime = maxSearchTime;

	DSL_dataset ds;
	ds.ReadFile(fileName);

	int res = tan.Learn(ds, *(DSL_network*)net);
	return res;
}

/** QGeNIe/DeMorgan **/

int CALL_TYPE Demorgan_SetParentWeights(void* obj, int size, const double* dArr) {
	DSL_doubleArray da(size);
	for (int i = 0; i < size; i++) {
		da[i] = dArr[i];
	}
	return ((DSL_demorgan*)obj)->SetParentWeights(da);
}

int CALL_TYPE Demorgan_SetParentWeight(void* obj, int numParent, double value) {
	return ((DSL_demorgan*)obj)->SetParentWeight(numParent, value);
}

double CALL_TYPE Demorgan_GetParentWeight(void* obj, int numParent) {
	return ((DSL_demorgan*)obj)->GetParentWeight(numParent);
}

void* CALL_TYPE Demorgan_GetParentWeights(void* obj) {
    return (void*)&((DSL_demorgan*)obj)->GetParentWeights();
}

int CALL_TYPE Demorgan_SetParentTypes(void* obj, int size, const int* iArr) {
	DSL_intArray ia(size);
	for (int i = 0; i < size; i++) {
		ia[i] = iArr[i];
	}
	return ((DSL_demorgan*)obj)->SetParentTypes(ia);
}

int CALL_TYPE Demorgan_SetParentType(void* obj, int numParent, int value) {
	return ((DSL_demorgan*)obj)->SetParentType(numParent, value);
}

int CALL_TYPE Demorgan_GetParentType(void* obj, int numParent) {
	return ((DSL_demorgan*)obj)->GetParentType(numParent);
}

void* CALL_TYPE Demorgan_GetParentTypes(void* obj) {
	return (void*)&((DSL_demorgan*)obj)->GetParentTypes();
}

int CALL_TYPE Demorgan_GetTemporalParentType(void* obj, int order, int numParent) {
	return ((DSL_demorgan*)obj)->GetTemporalParentType(order, numParent);
}

int CALL_TYPE Demorgan_SetTemporalParentType(void* obj, int order, int numParent, int parentType) {
	return ((DSL_demorgan*)obj)->SetTemporalParentType(order, numParent, parentType);
}

double CALL_TYPE Demorgan_GetTemporalParentWeight(void* obj, int order, int numParent) {
	return ((DSL_demorgan*)obj)->GetTemporalParentWeight(order, numParent);
}

int CALL_TYPE Demorgan_SetTemporalParentWeight(void* obj, int order, int numParent, double weight) {
	return ((DSL_demorgan*)obj)->SetTemporalParentWeight(order, numParent, weight);
}

double CALL_TYPE Demorgan_GetPriorBelief(void* obj) {
	return ((DSL_demorgan*)obj)->GetPriorBelief();
}

int CALL_TYPE Demorgan_SetPriorBelief(void* obj, double value) {
	return ((DSL_demorgan*)obj)->SetPriorBelief(value);
}


const char* CALL_TYPE test(const char* arg) {
	cout << arg;
	return arg;
}

int main() {
	DSL_network net;

	int success = net.AddNode(DSL_CPT, "Yay");
	cout << "Woot!";

	return 0;
}
