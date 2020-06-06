## Contributing to InferLO

If you are member of InferLO organization, you can work in inferlo repository 
directly. You still have to create branches for your ork (see below). Guidelines below 
apply mostly for this case.

If you want to contribute not being member, you should fork repository, make changes
on a branch of your repository and then make a pull request.

Described workflow should work both on Windows and Linux.

### First time setup

You will need [Git](https://git-scm.com/) and [Python](python.org) 3.7+.

Clone the repository:

```
git clone git@github.com:InferLO/inferlo.git
cd inferlo
```

Install the requirements:

```
pip install -r requirements.txt
pip install -r docs/requirements.txt
pip install -r tools/requirements.txt
```


Check that tests pass:

```
pytest
```


### Development cycle

For making any changes you will have to create new branch, make your changes 
on that branch and then merge it with `master` branch.

Create a branch (replace ```my-branch``` with descriptive name):

```git checkout -b my-branch```

Make your changes.

When you are ready to commit, check that tests pass and style check passes:

```
pytest
tools/check_style.sh
```

On Windows use ".bat" instead of ".sh". If it doesn't work, just run commands from 
that script (pycodestyle and pylint). If style check fails, some of errors can 
be automatically fixed by ```tools/fix_style.sh```.

Optionally, you can check that documentation builds, and also see up-to-date documentation 
locally:

```
cd docs
make html
```

