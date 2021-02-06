import os

LICENSE_NOTICE = ''.join([
    '# Copyright (c) The InferLO authors. All rights reserved.\n'
    '# Licensed under the Apache License, Version 2.0 - see LICENSE.\n'
])

LICENSE_NAME = 'Apache License'
assert LICENSE_NAME in LICENSE_NOTICE


def needs_license(file_name):
    if not file_name.endswith('.py'):
        return False
    if file_name.endswith('__init__.py'):
        return False
    with open(file_name, encoding='utf-8') as f:
        content = f.read()
    return not LICENSE_NAME in content


def append_license(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        content = ''.join(f.readlines())
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(LICENSE_NOTICE + content)


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
        print("Appended license to %s" % file_name)
