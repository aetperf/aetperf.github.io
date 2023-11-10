
Suppose you've developed a Python package called `MyPackage` on Linux, with specific package requirements, and need to install it on a Windows machine that lacks internet access, on which you may not have any specific priviledges. This blog post will show you one way to do that, which involves downloading Wheel files on a similar Windows machine with internet access, then transferring them to the isolated machine and install them in a virtual environment.

## On a machine that has internet access

Begin by preparing your working environment.

- Create a Main Directory

On your internet-connected Windows machine, open PowerShell and create a main directory for your installation:

```powershell
PS C:\Users\Francois> mkdir MyPackageFolder
PS C:\Users\Francois> cd MyPackageFolder
```
- Download Python

Download the Python embedded package from the official Python [Python web site](https://www.python.org/downloads/release/python-3116/). Specifically, choose the "Windows embeddable package 64-bit" from [this](https://www.python.org/ftp/python/3.11.6/python-3.11.6-embed-amd64.zip) link. The file size is about 10.7 MB for Python 3.11. Once downloaded, copy the zip archive - `python-3.11.6-embed-amd64.zip` - into your directory, extract and delete it: 

```powershell
PS C:\Users\Francois\MyPackageFolder> ls
```

	    Directory: C:\Users\Francois\MyPackageFolder


	Mode                 LastWriteTime         Length Name
	----                 -------------         ------ ----
	d-----         11/1/2023   8:03 PM                python-3.11.6-embed-amd64


- Add Your Python Package

Copy your Python package folder - named `MyPackage` - into this directory, excluding any hidden or useless folders like `.git`, `.github`, `__pycache__`, `.ruff_cache`...

- Get pip

To install packages, you'll need pip. Download `get-pip.py` from https://bootstrap.pypa.io/ and place it in your Python folder `python-3.11.6-embed-amd64\`. Then, install pip with the following command:

```powershell
PS C:\Users\Francois\MyPackageFolder> cd .\python-3.11.6-embed-amd64\
PS C:\Users\Francois\MyPackageFolder\python-3.11.6-embed-amd64> .\python.exe .\get-pip.py
```

You can confirm the installation by checking the `Scripts` folder within your Python directory:

```powershell
PS C:\Users\Francois\MyPackageFolder\python-3.11.6-embed-amd64> ls .\Scripts\
```
	    Directory: C:\Users\Francois\MyPackageFolder\python-3.11.6-embed-amd64\Scripts


	Mode                 LastWriteTime         Length Name
	----                 -------------         ------ ----
	-a----         11/1/2023   8:13 PM         108424 pip.exe
	-a----         11/1/2023   8:13 PM         108424 pip3.11.exe
	-a----         11/1/2023   8:13 PM         108424 pip3.exe
	-a----         11/1/2023   8:13 PM         108411 wheel.exe

- Update the `python37._pth` File

Open the `python37._pth` file and uncomment the following line by removing the '#' symbol:

```python
#import site
```

- Create a Wheelhouse Directory

*Wheel* is a built distribution format: a wheel file is a zipped archive with the `.whl` extension, usually containing a pre-built binary package.

In the root folder, create a directory named `wheelhouse` to store all the wheel files:

```powershell
PS C:\Users\Francois\MyPackageFolder\python-3.11.6-embed-amd64> cd ..
PS C:\Users\Francois\MyPackageFolder> mkdir wheelhouse
```

	    Directory: C:\Users\Francois\MyPackageFolder


	Mode                 LastWriteTime         Length Name
	----                 -------------         ------ ----
	d-----         11/1/2023   8:24 PM                wheelhouse

- Download Required Packages

Use your internet-connected Windows machine to download all the required packages and save them into the `wheelhouse` folder. You can do this by running the following commands:

```powershel
PS C:\Users\Francois\MyPackageFolder> .\python-3.11.6-embed-amd64\python.exe -m pip download -r .\MyPackage\requirements.txt -d .\wheelhouse
```

Repeat the process for any other requirements, such as for virtual environments, with a similar command:

```powershell
PS C:\Users\Francois\MyPackageFolder> .\python-3.11.6-embed-amd64\python.exe -m pip download -r .\MyPackage\requirements_venv.txt -d .\wheelhouse
```

These virtual environment requirements are the following ones in our case:

	virtualenv
	distlib<1,>=0.3.7
	filelock<4,>=3.12.2
	platformdirs<4,>=3.9.1

Verify that the downloaded wheels are in the "wheelhouse" directory:

```powershell
PS C:\Users\Francois\MyPackageFolder> ls .\wheelhouse\
```

	    Directory: C:\Users\Francois\MyPackageFolder\wheelhouse


	Mode                 LastWriteTime         Length Name
	----                 -------------         ------ ----
	-a----         11/1/2023   8:27 PM         181495 cffi-1.16.0-cp311-cp311-win_amd64.whl
	-a----         11/1/2023   8:27 PM          25335 colorama-0.4.6-py2.py3-none-any.whl
	-a----         11/1/2023   8:27 PM          13277 contextlib2-21.6.0-py2.py3-none-any.whl
	-a----         11/1/2023   8:27 PM        2669522 cryptography-41.0.5-cp37-abi3-win_amd64.whl
	-a----         11/1/2023   8:28 PM         468869 distlib-0.3.7-py2.py3-none-any.whl
	-a----         11/1/2023   8:28 PM          11740 filelock-3.13.1-py3-none-any.whl
	-a----         11/1/2023   8:27 PM         288398 greenlet-3.0.1-cp311-cp311-win_amd64.whl
	-a----         11/1/2023   8:27 PM          62549 loguru-0.7.2-py3-none-any.whl
	-a----         11/1/2023   8:27 PM       15799803 numpy-1.26.1-cp311-cp311-win_amd64.whl
	-a----         11/1/2023   8:27 PM       10647686 pandas-2.1.2-cp311-cp311-win_amd64.whl
	-a----         11/1/2023   8:28 PM          17579 platformdirs-3.11.0-py3-none-any.whl
	-a----         11/1/2023   8:27 PM        1163579 psycopg2-2.9.9-cp311-cp311-win_amd64.whl
	-a----         11/1/2023   8:27 PM         118697 pycparser-2.21-py2.py3-none-any.whl
	-a----         11/1/2023   8:27 PM        1978695 pymssql-2.2.10-cp311-cp311-win_amd64.whl
	-a----         11/1/2023   8:27 PM         247702 python_dateutil-2.8.2-py2.py3-none-any.whl
	-a----         11/1/2023   8:27 PM         502454 pytz-2023.3.post1-py2.py3-none-any.whl
	-a----         11/1/2023   8:27 PM           5924 scramp-1.1.0-py3-none-any.whl
	-a----         11/1/2023   8:27 PM          11053 six-1.16.0-py2.py3-none-any.whl
	-a----         11/1/2023   8:27 PM          31584 typing_extensions-4.8.0-py3-none-any.whl
	-a----         11/1/2023   8:27 PM         341835 tzdata-2023.3-py2.py3-none-any.whl
	-a----         11/1/2023   8:28 PM        3768405 virtualenv-20.24.6-py3-none-any.whl
	-a----         11/1/2023   8:27 PM           3604 win32_setctime-1.1.0-py3-none-any.whl

- Zip and Transfer

Note that you could also include your own package in this `wheelhouse` folder, or a package you edited. In the case of a pure python package, here is the process:

```powershell
PS C:\Users\Francois\MyPackageFolder> cd ..\MyOwnRequiredPackage\  
PS C:\Users\Francois\MyOwnRequiredPackage> python setup.py bdist_wheel  
PS C:\Users\Francois\MyOwnRequiredPackage> cp .\dist\myownrequiredpackage-1.0-py3-none-any.whl ..\wheelhouse\
```

Now, zip the entire `MyPackageFolder` folder and move it to the machine without internet access. 

With this zipped package in hand, you can install your Python package and its dependencies on the isolated Windows machine.

## On a machine that does not have internet access

- Extract the zipped folder wherever you want and navigate into this directory using PowerShell. In the following, the same path as above is used:

```powershell
PS C:\Users\Francois\MyPackageFolder> ls
```

	    Directory: C:\Users\Francois\MyPackageFolder


	Mode                 LastWriteTime         Length Name
	----                 -------------         ------ ----
	d-----         11/1/2023   8:43 PM                MyPackage
	d-----         11/1/2023   8:43 PM                python-3.11.6-embed-amd64
	d-----         11/1/2023   8:43 PM                wheelhouse


- install virtualenv:

```powershell
PS C:\Users\Francois\MyPackageFolder> .\python-3.11.6-embed-amd64\python.exe -m pip install virtualenv --no-index --find-links .\wheelhouse
```

- create a virtualenv, e.g. `testenv`:

```powershell
PS C:\Users\Francois\MyPackageFolder>  .\python-3.11.6-embed-amd64\python.exe -m virtualenv testenv
```

You can verify that a new subfolder, `testenv`, has been created:

```powershell
PS C:\Users\Francois\MyPackageFolder> ls
```

	    Directory: C:\Users\Francois\MyPackageFolder


	Mode                 LastWriteTime         Length Name
	----                 -------------         ------ ----
	d-----         11/1/2023   8:43 PM                MyPackage
	d-----         11/1/2023   8:43 PM                python-3.11.6-embed-amd64
	d-----         11/1/2023   8:49 PM                testenv
	d-----         11/1/2023   8:43 PM                wheelhouse

- activate the virtualenv:

```powershell
PS C:\Users\Francois\MyPackageFolder> .\testenv\Scripts\activate.ps1
```

- install all the package requirements

```powershell
(testenv) PS C:\Users\Francois\MyPackageFolder> python -m pip install -r .\MyPackage\requirements.txt --no-index --find-links .\wheelhouse
```

- now we can install our package:

```powershell
(testenv) PS C:\Users\Francois\MyPackageFolder> python -m pip install -e .\MyPackage\ --no-build-isolation
```


Here you go! Note that the package is installed in *editable* mode in this case with the `-e` option.


```powershell
(testenv) PS C:\Users\Francois\MyPackageFolder> python
Python 3.11.6 (tags/v3.11.6:8b6ee5b, Oct  2 2023, 14:57:12) [MSC v.1935 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import MyPackage
>>> quit()
```

Additionally, you can verify that a command line entry point `myentrypoint` - that does nothing here - is available in the shell:

```powershell
(testenv) PS C:\Users\Francois\MyPackageFolder> myentrypoint
(testenv) PS C:\Users\Francois\MyPackageFolder> deactivate
PS C:\Users\Francois\MyPackageFolder>
```

The entry point has been defined has follows in a module names `cli.py`:

```python
def myentrypoint():
    pass
```
and in `setup.py`:

```python
setup(
    entry_points={
        "console_scripts": [
            "myentrypoint=mypackage.cli:myentrypoint",
        ],
    },
)
```

Now, your Python package is successfully installed and ready to use.

In summary, this post shows how to install your Python package in a virtual environment on a Windows machine without internet access, using the embedded Python version, which does not require any special privileges.

## References

- Microsoft - [Install Python in an offline Windows environment](https://learn.microsoft.com/bs-latn-ba/azure-data-studio/notebooks/notebooks-python-offline-installation?view=sql-server-2016)

- Bojan Nikolic - [Installing Python on Windows using the embedded package (no privileges needed)](https://bnikolic.co.uk/blog/python/2022/03/14/python-embedwin.html)
