## Contributing to InferLO

### First time setup

You will need [Git](https://git-scm.com/) and [Python](python.org) 3.7+. It's
recommended, but not necessary, to do all development in the virtual environment.

Clone the repository:

```
git clone https://github.com/InferLO/inferlo.git
cd inferlo
```

Install the requirements:

```
pip install -r requirements.txt
pip install -r docs/requirements.txt
pip install -r tools/dev_requirements.txt
```


Check that tests pass:

```
pytest
```


### Development cycle

For making any changes you will have to create new branch, make your changes 
on that branch and then merge it with `master` branch.

*Process below applies to those who has commit access to repository.
If you don't have it, you should fork this repository and make all changes to a
branch in your own repository. Then you should create pull request to merge
your branch with ```master``` branch of the main repository.*

Create new branch (replace ```my-branch``` with descriptive name):
```
git checkout -b my-branch
```

If you are getting back to work on existing branch, switch to that branch
and sync it with master:

```
git checkout my-branch
git merge master
```

Make your changes.

When you are ready to commit, check that tests pass and style check passes:

```
pytest
pycodestyle inferlo 
pylint --rcfile=tools/pylintrc.txt inferlo
```

To save some time, you can run ```tools/check_style.sh```
or ```tools/check_style.bat``` to perform style checks.

If checks don't pass - make them pass.

Commit your changes.

```
git add .
git commmit -m 'Commit description'
git push origin HEAD
```

If you get error at this point saying "Updates were rejected", then remote 
version of your branch was updated, you have to pull the changes:

```git pull origin my-branch```

Now, go to GitHub and find your branch. Make sure that continuous integration
passes for your last commit. Now, 
[create a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

Your pull request will be reviewed and you might be asked to make sme changes.

Finally, when your PR is approved,
[merge it](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/merging-a-pull-request).

### Testing 

* **Unit tests**. All unit tests must pass. 
That can be checked by running ```pytest``` in root of the repository. 
When you are working on a new feature, to save time, you can run tests only for
the test file affected by your changes: ```pytest path_to_file.py```, 
also PyCharm can run tests in one file or one package for you. 
Before pushing chages, please run global ```pytest```.

* **Coverage**. You should cover all functionality with tests. In other words, any
change to existing code which makes it incorrect must make some test to fail. We 
don't enforce coverage automatically, but we might do it in the future.
Code reviewers should ask contributors to add tests if some behaviors are not
covered with tests.

* **Style**. We perform 2 independent style checks: 
[PEP8](https://www.python.org/dev/peps/pep-0008/) and 
[pylint](https://www.pylint.org/). PEP8 is set of simple rules about code layout.
We enforce them with ```pycodestyle```. Most PEP8 violations can be automatically 
fixed by ```autopep8```, and we have a simple script ```tools/fix_style.sh``` to
do that. Pylint performs series of more clever checks. Both
checks are mandatory - if they fail, continuous integration won't let
yuo tom mege code to master.

## Documentation

We use [sphinx](https://www.sphinx-doc.org/en/master/) to automatically 
build the documentation. Documentation is authomatically exported to 
ReadTheDocs.

In particular, pydocs are automatically converted to "API reference" section in
documentation. If you are adding new public class or function to the library,
you should add it to ```docs/api.rst```.

To build documentation locally, run:

```
cd docs
make html
```

And then open ```docs/_build/html/index.html```.

We have continuous integration check requiring that documentation builds. 
So, if you made any changes to documentation, you should build it locally to 
check that it builds.

To build the docs you will have to additionally install 
[pandoc](https://pandoc.org/). On Linux you can do that by running
```sudo apt-get intsall pandoc```.

## Adding new algorithms

If you are adding new algorithm to solve a problem fo particular kind of problem,
for which we already have at least one algorithm, you should follow the following 
process.

Write your algorithm as a function, which takes graphical model object
as first parameter. If your algorithm has configurable parameters, they
should be keyword arguments with default values.

Then add call to your function from a method on model class. It should be 
conditioned on user passing your algorithm name as ```algorithm``` argument.