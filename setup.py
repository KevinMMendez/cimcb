from setuptools import setup


def readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()
  
setup(
    name="cimcb",
    version="1.1.0",
    description="This is a pre-release.",
    long_description=readme(),
    long_description_content_type='text/markdown',
    license="http://www.apache.org/licenses/LICENSE-2.0.html",
    url="https://github.com/KevinMMendez/cimcb",
    packages=["cimcb", "cimcb.bootstrap", "cimcb.cross_val", "cimcb.model", "cimcb.plot", "cimcb.utils"],
    python_requires=">=3.5",
    install_requires=["bokeh>=1.0.0",
                      "keras",
                      "numpy>=1.12",
                      "pandas",
                      "scipy",
                      "scikit-learn",
                      "statsmodels",
                      "tensorflow",
                      "tqdm",
                      "xlrd",
                      "joblib"],
    author="Kevin Mendez, David Broadhurst",
    author_email="k.mendez@ecu.edu.au, d.broadhurst@ecu.edu.au",
)
