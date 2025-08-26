## bni_smile - dynamic library version

This is a dynamic library version of bni_smile that can be used pretty much anywhere that dynamic libraries are supported (and has the supporting runtime libraries, etc.)

There is a Windows and Linux version of the library.

## Building on Windows

You will need Visual Studio 2022 (community edition is fine) with the Desktop for C/C++ development package installed. Make sure that you open the .vcxproj file with VS2022 --- don't just open the folder, as that won't work.

(You can also get things working in other environments too.)

To build:
- Open bismile.vcxproj in VS2022
- Run Build -> Batch build... -> Build (or Rebuild)
  (Outputs will go into x64/bismile64.dll and x86/bismile.dll)
- In the base dll_smile/ folder, run copy_dlls.py
  (This will archive the old dlls, and then copy in the new ones)
- To check, run test_bni_smile.py

### Licenses

The version of SMILE in this repository is the *academic* version. To build, obtain an academic license from https://download.bayesfusion.com/files.html?category=Academia, then extract the `smile_license.h` into the `windows/bismile` folder.

SMILE needs a different package for business versions. You can go to https://download.bayesfusion.com/files.html?category=Business to obtain the packge. You will need to rename the `windows/bismile/smile` folder (to, say, `smile-academic`) and then extract the contents into `windows/bismile/smile`. You need to download the version for VS2022 (e.g. `smile-academic-2.3.2-win-vs2022.zip`).