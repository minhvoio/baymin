smoking_lung_cancer.dne"
filePath2 = "outputs/test.dne"

myNet = Net(netDir+filePath1)
print("Re-opened network structure:")
for node in myNet.nodes():
		print(f"{node.name()} -> {[child.name() for child in node.children()]}")