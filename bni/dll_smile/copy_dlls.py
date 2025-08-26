import shutil, os, time, re

ARCHIVE = True

def copyDlls():
	# We just archive the old version on the first run of the day
	archiveDir = 'dll_archive/{}'.format(time.strftime('%Y-%m-%d'))
	
	if not os.path.exists(archiveDir):
		os.makedirs(archiveDir)

		dllFns = ['bismile.dll', 'bismile64.dll']

		if ARCHIVE:
			for dllFn in dllFns:
				shutil.copy2(dllFn, archiveDir)
			
	shutil.copy2('windows/bismile/x86/bismile.dll', 'bismile.dll')
	shutil.copy2('windows/bismile/x64/bismile64.dll', 'bismile64.dll')

	# If the 'inactive' folder is academic, then this is the business build
	try:
		os.unlink('bismile-academic-dlls.txt')
		os.unlink('bismile-business-dlls.txt')
	except: pass
	isBusiness = os.path.exists('windows/bismile/smile-academic')
	type = 'business' if isBusiness else 'academic'
	open(f'bismile-{type}-dlls.txt', 'w').close()
	shutil.copy2('bismile.dll', os.path.join(f'latest-{type}', 'bismile.dll'))
	shutil.copy2('bismile64.dll', os.path.join(f'latest-{type}', 'bismile64.dll'))

	today = time.strftime('%Y-%m-%d')

	# Release to GitHub:
	if input('Release to github? [yn]')=='y':
		# Delete old release, if there
		os.system(f'gh release delete {type}-{today} --yes')
		os.system(f'gh release create {type}-{today} bismile.dll bismile64.dll --title "{type}-{today}" --notes "Latest {type} release"')

def mapType(typeStr):
	typeMap = {
		'void': 'VOID',
		'int': 'ctypes.c_int',
		'float': 'ctypes.c_float',
		'double': 'ctypes.c_double',
		'char': 'ctypes.c_char',
		'bool': 'ctypes.c_bool',
	}
	numPointers = [0]
	def count(m): numPointers[0] += 1
	unsigned = [False]
	def doUnsigned(m): unsigned[0] = True
	re.sub(r'\*', count, typeStr)
	typeStr = re.sub(r'\s*\*\s*', '', typeStr)
	typeStr = re.sub(r'\bconst\b', '', typeStr)
	typeStr = re.sub(r'\bunsigned\b', doUnsigned, typeStr)
	typeStr = typeStr.strip()
	
	mappedType = typeMap.get(typeStr, typeStr)
	
	# Update any unsigned
	if unsigned[0]:
		mappedType = re.sub(r'ctypes\.c_', r'\g<0>u', mappedType)
	
	# Add pointers
	for i in range(numPointers[0]):
		mappedType = 'ctypes.POINTER('+mappedType+')'
		
	# Special case for unsigned chars
	mappedType = re.sub(r'ctypes\.c_uchar', 'ctypes.c_ubyte', mappedType)
	
	# Replace char* with c_simplechar_p
	mappedType = re.sub(r'ctypes\.POINTER\(ctypes\.c_char\)', 'c_simplechar_p', mappedType)
	
	return mappedType

def makePyBindings():
	hdr = 'windows/bismile/bismile.h'
	with open(hdr) as bismileHdr, open('bindings.py','w') as bindingsFile:
		for ln in bismileHdr:
			m = re.match(r'\s*EXPORT\s*\(\s*(?P<returnType>[^,]*),\s*(?P<functionName>\w+)\s*\)\(\s*(?P<args>.*?)\s*\)\s*;', ln)
			if m:
				returnType = m.group('returnType')
				functionName = m.group('functionName')
				argSpecs = m.group('args')
				
				args = []
				for argSpec in re.split(r'\s*,\s*', argSpecs):
					if not argSpec:  continue
					# Strip off any dimensions at end
					dimensions = ''
					def rep(m):
						dimensions = m.group(1)
						return ''
					argSpec = re.sub(r'([\d\[\]]+)$', rep,  argSpec)
					#print('argSpec',argSpec)
					m = re.match(r'^\s*(.*?)\s*(\w+)\s*$', argSpec)
					varType = m.group(1)
					varName = m.group(2)
					args.append([varType, varName])
				
				returnTypeStr = mapType(returnType)
				bindingStr = ''
				if returnTypeStr == 'VOID':
					bindingStr += '#'
				bindingStr += 'g.'+functionName+'.restype = '+returnTypeStr
				bindingStr += '\n'
				bindingStr += 'g.'+functionName+'.argtypes = ['
				bindingStr += ', '.join(mapType(arg[0]) for arg in args)
				bindingStr += ']'
				bindingStr += '\n'
				
				# print(bindingStr)
				
				bindingsFile.write(bindingStr)

def updateBniSmile():
	with open('bindings.py') as bindingsFile:
		bindingsStr = bindingsFile.read()
		
	with open('bni_smile.py') as bniSmileFile:
		bniSmileStr = bniSmileFile.read()
	
	bniSmileStr = re.sub(r'(# \[\[BINDINGS START\]\].*)([\w\W]*)(# \[\[BINDINGS END\]\].*)', r'\1\n'+bindingsStr.replace('\\', '\\\\')+r'\3', bniSmileStr)
	
	with open('bni_smile.py', 'w') as bniSmileFile:
		bniSmileFile.write(bniSmileStr)

copyDlls()
makePyBindings()
updateBniSmile()