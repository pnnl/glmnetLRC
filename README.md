## glmnetLRC

### Lasso and Elastic-Net Logistic Regression Classification with an Arbitrary Loss Function

The [package vignette](http://pnnl.github.io/glmnetLRC) includes examples to help you get started, as well as mathematical 
details of the algorithms used by the package.

#### To cite:

Sego LH, Venzin AM, Ramey JA. 2016. glmnetLRC: Lasso and Elastic-Net Logistic Regression Classification (LRC) 
with an Arbitrary Loss Function in R. Pacific Northwest National Laboratory. http://pnnl.github.io/glmnetLRC.

#### To install:

Begin by installing dependencies from [CRAN](http://cran.r-project.org):

    install.packages(c("devtools", "glmnet", "plyr"))

The `Smisc` package (which is imported by `glmnetLRC`) contains C code and requires compilation. To do this
* on a Mac, you'll need [Xcode](http://developer.apple.com/xcode/) 
* on Windows, you'll need to install [R tools](http://cran.r-project.org/bin/windows/Rtools/)
* on Linux, compilation should take place "automatically"

With the compilation tools in place, you can now install the `Smisc` and `glmnetLRC` packages
from [the PNNL github site](http://github.com/pnnl):

    devtools::install_github("pnnl/Smisc")
    devtools::install_github("pnnl/glmnetLRC")

#### Contributing

We welcome contributions to this package.  Please follow [these steps](http://pnnl.github.io/prepPackage) when contributing.
