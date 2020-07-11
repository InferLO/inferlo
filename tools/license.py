import os

license_notice = \
    """# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
"""

license_name = 'Apache License'


def needs_license(file_name):
    if not file_name.endswith('.py'):
        return False
    if file_name.endswith('__init__.py'):
        return False
    with open(file_name, encoding='utf-8') as f:
        content = f.read()
    return not license_name in content


def append_license(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        content = ''.join(f.readlines())
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(license_notice + content)


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), 'inferlo')
    files_need_license = []
    for dir, _, files in os.walk(path):
        for file in files:
            file_name = os.path.join(dir, file)
            if (needs_license(file_name)):
                files_need_license.append(file_name)
    for file_name in files_need_license:
        append_license(file_name)
