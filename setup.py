import setuptools

with open("README.md", "r", encoding='utf-8') as readme_file:
    long_description = readme_file.read()

with open('requirements.txt', 'r', encoding='utf-8') as req_file:
    requirements = [r.strip() for r in req_file.readlines()]

setuptools.setup(
    name="inferlo",
    version="0.3.1",
    author="The InferLO Developers",
    description="Inference, Learning & Optimization with Graphical Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache 2',
    url="https://github.com/InferLO/inferlo",
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=requirements,
)
