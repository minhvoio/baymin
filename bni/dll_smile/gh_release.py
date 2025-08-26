import shutil, os, time, re

def ghRelease(type):
    today = time.strftime('%Y-%m-%d')
    shutil.make_archive(f'bni_smile_lib-{type}-{today}', 'zip', f'latest-{type}')
    # Delete old release, if there
    os.system(f'gh release delete {type}-{today} --yes')
    os.system(f'gh release create {type}-{today} bni_smile_lib-{type}-{today}.zip --title "{type}-{today}" --notes "bni_smile {type}-{today} release"')

def ghReleaseOptions():
    # Release to GitHub:
    print('This will release what\'s in latest-academic and/or latest-business to github')
    print()
    print('1. Release both academic and business to github')
    print('2. Release academic to github')
    print('3. Release business to github')
    print('<enter to quit>')
    opt = input('Option [1-3]?')
    if opt=='1':
        ghRelease('academic')
        ghRelease('business')
    elif opt=='2':
        ghRelease('academic')
    elif opt=='3':
        ghRelease('business')

ghReleaseOptions()