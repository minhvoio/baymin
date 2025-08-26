@echo off
echo Academic version...
ren smile-academic smile
ren smile_license_academic.h smile_license.h
call make_libbismile.bat
ren smile smile-academic
ren smile_license.h smile_license_academic.h
copy libbismile.so ..\latest-academic

echo Business version...
ren smile-business smile
ren smile_license_business.h smile_license.h
call make_libbismile.bat
ren smile smile-business
ren smile_license.h smile_license_business.h
copy libbismile.so ..\latest-business
