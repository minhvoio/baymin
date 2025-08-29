import shutil, os, re

def copy(fromDir, files, toDir):
	for fn in files:
		if type(fn) == list:
			sourceFn = os.path.join(fromDir, fn[0])
			destFn = os.path.join(toDir, fn[1])
		else:
			sourceFn = os.path.join(fromDir, fn)
			destFn = os.path.join(toDir, fn)
		if os.path.isdir(sourceFn):
			shutil.copytree(sourceFn, destFn, dirs_exist_ok=True)
		else:
			shutil.copyfile(sourceFn, destFn)

print('Git checkin first? (Ctrl-C to do so)')
input()
copy(fromDir = '../windows/bismile/', files = ['bismile.cpp', 'bismile.h', 'smile_license_business.h', ['smile_license.h','smile_license_academic.h']], toDir = '.')
copy(fromDir = '..', files = ['bni_smile.py', 'test_bni_smile.py'], toDir = '.')
copy(fromDir = '../..', files = ['nets'], toDir = '.')

# Turn off the define (windows only)
txt = re.sub(r'#define CREATEDLL', '', open('bismile.h').read())
open('bismile.h', 'w').write(txt)

# Remove pch.h, because the darn ifdef/endif wrapping macro doesn't work in vc++
txt = re.sub(r'#include "pch.h"', '', open('bismile.cpp').read())
open('bismile.cpp', 'w').write(txt)

# Update 'nets' folder location
txt = re.sub(r'\nnetDir = .*', '\nnetDir = "nets/"', open('test_bni_smile.py').read())
open('test_bni_smile.py', 'w').write(txt)

print('Done')
input()